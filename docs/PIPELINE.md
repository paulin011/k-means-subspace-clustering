# PIPELINE.md — how the scripts chain together

`README.md` documents each script on its own. This file explains the **flow**: what
produces what, why each stage exists, and the invariants that let the stages check each
other. Metric definitions live in `docs/METRICS.md`; this is the architecture.

```
latents_2/  (13,021 files x [12288, 2048], 1.2 TB)
    │
    ├─1─ cluster_io.load_tokens        sample files -> [T, 2048] fp16 + sample.json
    │        │
    ├─2─ subspace_kmeans.py            K-subspaces (or MPPCA EM with --soft)
    │        └── model.pt + assignments.pt          <- the frozen model
    │             │
    │             ├─3─ holdout_eval.py       same kernel, unseen files -> holdout.json
    │             │
    │             ├─4─ file_signature.py     same kernel, ALL 13,021 files
    │             │       └── signatures.npz, {residual,label,margin}_map.npy,
    │             │           cluster_mix.csv, cluster_margin.csv
    │             │
    │             ├─5─ analyze_clusters.py   -> report.md   (+ --margins from stage 4)
    │             └─5─ temporal_spatial.py   -> temporal_report.md + maps/
    │
    └─6─ persistence_error.py          err_persist.npy  [13020, 12288]
             └── analyze_forecast_error.py -> weights.csv (goal 1) + attribution
```

Stages 3, 4 and 6 all consume stage 2's output and never modify it.

---

## 1. Sampling — `src/common/cluster_io.py`

**Why it exists:** 1.2 TB does not fit in 512 GB RAM, and I/O dominates everything.

`load_tokens()` samples **files**, not tokens. Touching a single token costs a ~100 MB
file read, so once a file is open you take all 12,288 of its cells. It allocates one flat
`torch.empty(T, 2048, dtype=float16)` up front (352 GB for 7000 files) and fills it from
`--load-workers` threads. fp16 halves RAM; the loss is irrelevant because the covariances
downstream accumulate in fp32.

Two mechanisms make runs comparable:

- **`sample_fingerprint`** = `sha1(tokens_per_file | seed | sorted file ids)[:12]`. Runs
  sharing a fingerprint saw the *identical* token set, so their objectives compare
  directly. v2/v3/v4/v6/v7/v8/v10 all share `82ca602ed7e7`.
- **`sample.json`** is written on every run; `--files-from` replays it exactly. File order
  is preserved in the manifest (init seeds depend on it) but sorted inside the hash.

`--max-ram-gb` guards the allocation but counts **only** the fp16 buffer. True peak also
holds the int32 file/cell id arrays and ~`load_workers` x 100 MB of transient decode
buffers, so leave headroom.

## 2. Clustering — `src/clustering/subspace_kmeans.py`

**The model.** Cluster *j* is an affine subspace: an anchor `mu_j` in R^2048 plus an
orthonormal basis `U_j` in R^{2048 x d}. Assignment is by orthogonal residual

    R_j(x) = ||x - mu_j||^2 - ||U_j^T (x - mu_j)||^2

k-means is exactly the **d = 0** case (no basis, second term vanishes), so one script
covers both algorithms.

**The kernel.** See `docs/KERNEL.md` — the residual is computed entirely with matrix
multiplies, never materialising the `[B, K, 2048]` difference tensor. This is the single
most confusing part of the codebase: the expression *looks* like it only measures distance
to centroids, but it is the exact orthogonal residual to the subspace. Every later stage
(holdout, signatures, margins) reuses this same kernel verbatim.

**Initialisation.** K random tokens become the means with `U = 0`. Because U=0, the first
sweep is a plain k-means step. This is deliberate: seeding with a random *partition*
instead would give every cluster nearly the same PCA basis (each would see the global
covariance) and the symmetry would never break.

**E-step:** assign each token to its argmin residual.

**M-step:** per cluster, build the covariance from streamed moments (`S = sum x x^T`,
`msum = sum x`, `cnt`), form `C = S/n - mu mu^T`, then **exact `eigh`** and keep the top-d
eigenvectors. PCA is the provably optimal d-dim subspace for a point set, so the objective
cannot increase — convergence is monotone. No power iteration, no approximation.

**Both GPUs**, one thread each, each accumulating its own `S [K, 2048, 2048]` (2.1 GB).
GPU peer-to-peer is **silently broken on this machine** (D2D copies return zeros with no
error), so accumulators are copied to CPU *from the thread that launched the kernels*,
merged there, and pushed back to GPU 0 for the eigendecomposition.

**The re-seed guard.** Clusters below `max(2d, 64)` tokens cannot support a d-dim basis.
The obvious repair — reseed from a random token with `U = 0` — **cannot work at large d**:
a point-seed's residual is the full `||x - mu||^2` while its rivals' residuals are tiny, so
it never wins a token back. That stranded 1 cluster at d=64 (v4) and 2 at d=128 (v5).
The working fix **splits the largest healthy cluster along its top PC**: both halves
inherit the parent's subspace, offset +/-1 std along PC1, so the next assignment divides
the parent's tokens between them. v6 has all 128 clusters healthy (min 227,886 tokens).

**Final relabel sweep.** After the loop, one more pass relabels every token under the
*saved* bases. Without it `model['counts']` would lag one step behind `assignments.pt`.
This sweep's objective becomes `final_obj_per_token` — the true residual of the saved
model. `history[-1]` is the *previous* model's objective, one step behind.

**Soft mode (`--soft`).** Swaps the argmin for EM on the matching **MPPCA** mixture, whose
components are

    C_j = U_j diag(lam_j) U_j^T + sigma2_j (I - U_j U_j^T)

That structure makes the Mahalanobis distance and log-determinant separable into terms the
kernel already computes:

    (x-mu)^T C_j^-1 (x-mu) = R_j / sigma2_j + sum_i z_i^2 / lam_i      (z = U_j^T(x-mu_j))
    log det C_j            = sum_i log lam_i + (DIM - d) log sigma2_j

so the only extra cost is two `[B, K, d]` einsums. Three details that matter:

- **Iteration 1 is always hard**: `use_soft` is gated on `"sigma2" in model`, and sigma2
  does not exist until the first M-step — there is no covariance to score a likelihood
  with yet.
- **`P.pow_(2)` is in place**: P is `[B, K, d]` (~8.6 GB at the default chunk size), so the
  `pcl` einsum consumes P *before* it is squared, avoiding a second copy and an OOM.
- **Truncation to `--soft-topm`** (default 4) caps the M-step at ~m x rather than K x cost,
  via a sort-and-group so each cluster does exactly one `addmm` over its own rows.

**`--soft-temp` is not optional in practice.** At T=1 (the literal likelihood) the D=2048
score is so peaked that responsibilities come out one-hot (measured mean top-1 = 0.993) and
the run is just slow hard k-subspaces. The likelihood counts 2048 dimensions as independent
evidence when the data has ~121 effective ones, so **T ~ 2048/121 ~ 17** de-overcounts it.
**The empirical validation of T=17 has been retracted (2026-08-04).** An earlier smoke test
reported mean top-1 = 0.877 with 37.2% of tokens below 0.9 and claimed this "matched" the
independently measured ~31-33% near-tie fraction. The production run v10 (7000 files, 25
iterations) measures **mean top-1 = 0.950, 15.1% below 0.9, 0.9% below 0.5** — considerably
more peaked. Two reasons the smoke test misled:

- It ran 300 files for 3 iterations with dozens of freshly re-seeded clusters. Tokens
  genuinely *were* ambiguous under that badly-fit model. A converged model discriminates
  much better, so T=17 under-tempers at production scale.
- The two quantities were never measuring the same thing. The near-tie margin uses only the
  residual `R`; the MPPCA likelihood additionally uses within-subspace position
  (`z^2/lam`), per-cluster `sigma2`, the log-det volume term, and the mixing weights —
  strictly more information, so it *should* be more decisive. Their agreement in the smoke
  test was a coincidence.

T=17 still stands on the effective-dimension argument (2048/121), which is a statement
about the data, not about that run. But it is now an unvalidated prior, not a calibrated
choice. If responsibilities matching the measured ~33% margin ambiguity are wanted, T needs
to be higher — roughly 30-40 — which is cheap to test by **rescoring the saved v10 model at
several T without refitting**. The reported objective is always the *untempered* likelihood,
so runs stay comparable regardless.

Everything soft is **additive**: `label` stays the argmax, `counts` stays the integer
bincount, and `sigma2` / `mixing` / `soft_counts` / `resp_idx` / `resp_w` are new optional
keys. Every existing reader works unchanged on a soft run's output.

## 3. Validation — `src/clustering/holdout_eval.py`

Freezes the trained means/bases and replays the identical kernel on tokens from files
**disjoint from `sampled_files`**. If held-out residual >> in-sample, the bases were
fitting noise. v6: **31.5% vs 31.5%** — the subspaces generalise. The threshold
(`OVERFIT_GAP_THRESHOLD = 0.03`) lives in `cluster_io.py` so the script's verdict and the
report's verdict cannot drift apart.

## 4. Full-dataset replay — `src/analysis/file_signature.py`

The same frozen kernel a third time, now over **all 13,021 files**, reduced *per file*
instead of per set. Streams in batches of `--batch-files` (default 500, ~25 GB fp16)
because the full dataset is ~656 GB as fp16. Single GPU — it is I/O-bound at ~12 files/s,
so the broken P2P never enters.

Per file: `mix [K]` (regime fingerprint), `mean_residual`, `residual_frac`,
`rare_exposure`, `mean_margin`, `near_frac`, and the reconstructed date
(`2014-01-01 + idx x 6h`). Beyond the reduction it keeps three things:

- **`residual_map [N, 12288]`** — per-cell residual: outlier detection at
  per-cell-per-timestep instead of per-timestep resolution. **Always normalise before
  ranking**: per-cell mean residual spans 15.7x (273…4276), so use
  `z = (R - cell_mean)/cell_std` from the climatology vectors in `signatures.npz`.
- **`label_map [N, 12288]`** — the dynamic label map over the whole dataset
  (`assignments.pt` covers only the sampled files).
- **`margin_map` + `cluster_margin.csv`** — `topk(2)` instead of `min(1)` makes the
  runner-up free, giving `margin = (R2-R1)/R1` and the near-tie statistics.

Rows follow the **sorted file list**, not file id, so `--limit` / `--files-from` subsets
stay dense. Both maps are `.npy` memmaps; load with `mmap_mode="r"`.

## 5. Reporting — `analyze_clusters.py`, `temporal_spatial.py`, `worldmap.py`

`analyze_clusters.py` is algorithm-agnostic: it reads anything following the `cluster_io`
schema. Its central result is the **law of total variance** decomposition:

    total    = between + within
    between  = sum_j w_j ||mu_j - mu_g||^2      8.4%   variance BETWEEN regimes
    within   = sum_j w_j trace_j               91.6%   variance INSIDE regimes
      captured = sum_j w_j sum_i lam_ji        60.1%   ...explained by the subspaces
      resid    = within - captured             31.5%   ...left over

That 60.1% is the justification for the whole approach: **two thirds of the structure is
within-regime variation** that a centroid-only model discards entirely.

`worldmap.py` holds pure-numpy HEALPix geometry (no healpy / cartopy / shapely).
`affinity_ordered_colors` does spectral seriation — Fiedler vector of the affinity matrix
mapped through `turbo` — so subspace-similar clusters get similar hues, which is what makes
interpolating *RGB* onto the map legitimate rather than an artefact.
`temporal_spatial.py` reuses the run's existing `assignments.pt`, so the monthly/seasonal
maps need no recomputation (~10 s).

## 6. The forecast-error track

`persistence_error.py` computes the model-free baseline `||x_t - x_{t+1}||^2` per cell
(CPU-only, ~15 min) — the skill-score denominator and a 6h-tendency measure.
`extract_forecast_error.py` is the supercomputer-side twin that runs the actual
WeatherGenerator forecast engine; its self-check aborts if `skill = 1 - mean_fc/mean_ps <= 0`,
which is what would reveal a wrong input / rope / config.
`analyze_forecast_error.py` joins either onto the clusters and is currently in **bridge
mode** (persistence only), pending that supercomputer run.

---

## The invariants that hold it together

The stages are written independently but cross-check each other. These are the checks that
catch a regression:

| invariant | status |
|---|---|
| count-weighted mean of `mean_residual` == `final_obj_per_token` | v6: 1888.12 vs 1887.83 |
| `residual_map` row mean == that row's `mean_residual` | 6e-08 rel |
| `label_map` row histogram == that row's `mix` | 6e-10 |
| `margin_map` row mean == `mean_margin` | 4e-06 rel |
| `cluster_tokens` == `bincount(label_map)` | exact |
| runner-up confusion row-sums == `cluster_tokens`, diagonal == 0 | exact |
| `model['counts']` == `bincount(assignments['label'])` | enforced by the final relabel sweep |
| held-out residual ~= in-sample residual | 31.5% vs 31.5% |

Plus the seed sweep (v6/v7/v8): objective spread 0.24%, variance split reproducing to
~0.5%, pairwise NMI 0.72 against 0.13 chance. That is what makes d=64 / K=128 the final
config rather than a lucky initialisation.

## Where it stands

The partition is seed-stable, generalises to unseen files, and captures ~60% of
within-regime variance. The margin pass adds that **33.4% of tokens are near-ties**, spread
very unevenly: ~1% for crisp clusters (c21, c66, c27) against ~70% for c106/c45, which are
each other's runner-up and are one continuum cut in half; 29 of 128 clusters have most of
their own tokens contested. That is the gap the soft/MPPCA run fills — not a better
partition (expect ~3.7% of labels to move) but per-token responsibilities that state the
ambiguity instead of hiding it.

Open: the supercomputer forecast-error extraction (`docs/wgen_architecture.md` §4).
