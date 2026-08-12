# Over-cluster then merge: K=256 → merged, at fixed d

*Plan written 2026-08-12. Branch `feat/overcluster-merge`.*

**Idea.** Fit K-subspaces with **more clusters than we want** (K=256) at the **same subspace
dimension** (d=64), then **merge** clusters agglomerately until a **configurable threshold** stops
the cascade. Goal: a better partition than fitting the target K directly, because over-parameterised
seeding escapes local minima that a direct fit at the target K cannot.

---

## 1. Feasibility (literature)

Over-cluster-then-merge is a standard, well-supported construction, and each of the three pieces we
need has direct precedent:

- **Over-parameterisation helps k-means escape local minima.** Initialising with more centres than
  the true cluster count is a known robustness device; the cost is that one true cluster gets split
  across several centres, which is exactly what the merge step repairs
  ([Multi-Prototypes Convex Merging k-means](https://arxiv.org/pdf/2302.07045),
  [Structures of Spurious Local Minima in k-means](https://arxiv.org/pdf/2002.06694)).
- **The hybrid "partition then agglomerate" pipeline is established.** Run k-means with a
  deliberately high k, then merge the resulting groups hierarchically using a data-driven stopping
  criterion rather than a pre-fixed final count
  ([Merging K-means with hierarchical clustering](https://pmc.ncbi.nlm.nih.gov/articles/PMC5935272/),
  [Clustering Large Datasets by Merging K-Means Solutions](https://link.springer.com/article/10.1007/s00357-019-09314-8)).
- **It is specifically endorsed for *subspace* clustering.** The recommended recipe is to start from
  many clusters — enough that each one is likely pure, i.e. drawn from a single subspace — and then
  merge, which also removes the need to fix the cluster count up front
  ([Subspace clustering without knowing the number of clusters](https://arxiv.org/pdf/1909.04406),
  [Subspace Clustering through Sub-Clusters](https://www.jmlr.org/papers/volume22/18-780/18-780.pdf)).
  Related: [ensembles of K-subspaces](https://academic.oup.com/imaiai/article/10/1/73/5999493) attacks
  the same initialisation sensitivity by consensus instead of merging.

**Merge criterion.** The literature's default for "which pair to merge" is
[Ward's](https://en.wikipedia.org/wiki/Ward's_method) minimum-variance linkage: merge the pair whose
union causes the **smallest increase in the total within-cluster residual**,
`Δ(a,b) = ESS(a∪b) − ESS(a) − ESS(b)`, with a distance threshold as the stopping rule. We adopt
Ward's *shape* but with our own objective substituted for ESS (§2) — that makes the merge criterion
**the same quantity the clustering already minimises**, so a merge's cost is denominated in the
units the run reports (`obj_per_token`).

Subspace **affinity** (mean squared cosine of principal angles, already in
`worldmap.build_affinity_matrix`) is the other obvious candidate and we keep it as a *secondary*
criterion — but it must not be the default: this repo has already measured that affinity
**sees orientation only, and is blind to mean placement and data density** (CLAUDE.md: c123 maxAff
0.697 / near-tie 1.3% vs c122 maxAff 0.668 / near-tie 45.7%). Two clusters can share a subspace
orientation and sit far apart along it; Ward-on-our-objective sees that, affinity does not.

**Verdict: feasible and standard.** The merge itself needs **no data pass** — it runs off the
per-cluster second moments the fit already computes every iteration (§2.3) — so the only new cost on
the clustering side is keeping one copy of them (`--save-moments`).

---

## 2. The merge criterion, and what it costs

### 2.1 Residual from stored fields

The K-subspaces objective for cluster *j* with `n_j` tokens, mean `μ_j`, member covariance `C_j`:

```
R_j = Σ_{x∈j} ( ‖x−μ_j‖² − ‖U_jᵀ(x−μ_j)‖² ) = n_j · ( tr(C_j) − Σ_{i≤d} λ_ji )
```

`model.pt` already stores `counts` (`n_j`), `trace` (`tr(C_j)`), `eigvals` (`λ_j`), `means`, `U`.
**So `R_j` is free**, and so is the run's objective `Σ_j R_j / T`.

### 2.2 The merged cluster's moments are exact

For a merged pair with `n = n_a+n_b`, `δ = μ_a−μ_b`:

```
μ   = (n_a μ_a + n_b μ_b) / n
C   = (n_a C_a + n_b C_b)/n + (n_a n_b/n²) δδᵀ
tr(C) = (n_a tr_a + n_b tr_b)/n + (n_a n_b/n²) ‖δ‖²
```

The trace — the "total variance" half of `R` — is therefore **exact and free**. The only hard term
is `Σ_{i≤d} λ_i(C)`, the top-*d* eigenvalues of the merged 2048×2048 covariance. That is what §2.3
is about.

### 2.3 Exact top-d eigenvalues from the stored second moments

`R_j` needs `Σ_{i≤d} λ_i(C)`, and the merged `C` is only known through `S`. Two routes were
implemented and **measured against fp64 brute force on real tokens** (S3 in the smoke tests):

**The PPCA surrogate (first attempt, now the fallback).** Model each parent as
`C_j ≈ U_j diag(λ_j) U_jᵀ + s_j (I − U_j U_jᵀ)`, `s_j = (tr_j − Σλ_j)/(DIM−d)` — the same
covariance model `--soft` already commits to. Then `C ≈ W + sI` with `W` PSD of rank ≤ 2d+1,
so the whole spectrum comes from a `q×q` eigh, `q ≤ 129`. Needs nothing but `model.pt`.

**It is not accurate enough to set a threshold by.** Measured error in `ΔR`: **7% on
synthetic data with a genuinely isotropic tail, but ~35% on real tokens** — and it picked
the *wrong cheapest pair*. The reason is exactly the property this project already
documented: the tokens have **~121 effective dimensions**, so at d=64 the unmodelled tail is
nowhere near isotropic, which is the surrogate's one assumption. Pair *ranking* survives
(Spearman 0.94), so it remains usable as a fallback for runs predating `--save-moments`
(v6 et al.), but a threshold denominated in "percent of the objective" would be fiction.

**Exact, from second moments (the implemented default).** Second moments are *additive over
a merge*, so `C = (S_a + S_b)/n − μμᵀ` is exact. `subspace_kmeans.py` already builds
`S [K,2048,2048]` every iteration and throws it away; a new `--save-moments` flag keeps the
final sweep's copy (4.3 GB at K=256, accumulated during the *final relabel* sweep so it
matches the labels in `assignments.pt`). Then `Σ_{i≤d} λ_i(C)` comes from **subspace
iteration warm-started at `V = orth([U_a, U_b, δ])`** — already close to the merged top-d.
Crucially `C` is never materialised: `C@V = (S_a@V + S_b@V)/n − μ(μᵀV)` uses the stored `S`
directly, so this is a `bmm` at a **measured 10.3 TFLOPS** rather than 46.6 ms/pair of full
`eigh` (which would be 25 min for one pass over K=256's pairs).

Measured convergence of the median `ΔR` error on real tokens, by iteration count:

| power-iters | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| real tokens (d=16) | 4.5% | 0.50% | 0.099% | 0.023% | **0.0016%** |
| synthetic worst case | 7.1% | 2.7% | 1.34% | 0.78% | 0.33% |

`--power-iters` defaults to **5**. The synthetic row is a deliberately near-degenerate
spectrum (one huge eigenvalue over a band of near-equal ones) and converges slower; real
token spectra decay smoothly. The **merge order is stable long before the value is** — the
cheapest pair is identified correctly even at 0 iterations, in every case tested. The fp32
moments themselves contribute only ~2e-8 relative error, so the iteration is the only error
term.

### 2.4 Cost

`K=256` → 32,640 initial pairs, plus ~K rescored per accepted merge (~98k evaluations for a
full cascade). Dominated by the `bmm`s and the tall-skinny QR. Measured at production K in
§4. Memory: `S` is 4.3 GB on GPU at K=256, and each scored pair gathers 2×34 MB, so pairs
are batched at 32.

Because the cascade is cheap relative to the 5 h fit it is **always run to completion**, and
the threshold is applied afterwards as a *cut* through the recorded dendrogram. Re-tuning
the threshold therefore costs nothing and needs no refit; `merge_log.json` holds every step.

### 2.5 What is approximate

Counts, means and traces are exact identities in both modes (verified in S2 to 1e-9). Each
*accepted* merge is additionally re-fitted with a full `eigh` of the merged covariance, and
`--refit-iters > 0` re-derives every surviving basis from real data — so the saved model
contains no iterated quantity at all; the iteration only ever chooses *which* pairs to
merge. `d=0` (plain k-means) collapses to Ward's classic exact formula
`ΔR = (n_a n_b/n)‖δ‖²`, which is the identity that pins down the δ term's factor.

## 3. Implementation

### Stage A — the K=256 fit (one new flag: `--save-moments`)

```bash
python3 src/clustering/subspace_kmeans.py \
  --files-from runs/clustering/v2_subspace_big/sample.json --seed 0 \
  --clusters 256 --dim 64 --iters 25 --chunk-size 131072 --save-moments \
  --max-ram-gb 420 --out runs/clustering/v11_k256_d64
```

Reusing v2's sample (fingerprint `82ca602ed7e7`, 7000 files) is the whole point: it makes the result
**directly comparable to v6** (K=128, d=64, same tokens, `final_obj_per_token` 1887.83).

**`--chunk-size` must be halved.** The assignment kernel holds `P = [B,K,d]` and, in hard mode,
materialises a second copy in `(P*P).sum(-1)`. At the default `B=262144`, K=256, d=64 that is
2 × 17.2 GB, plus `S [256,2048,2048]` = 4.3 GB and `X` = 2.1 GB → **~41 GB, over the A40's 44.4 GB
once fragmentation is counted**. `--chunk-size 131072` halves both copies to 8.6 GB (~24 GB peak,
the same headroom v6 ran with). Total FLOPs are unchanged.

**Measured peak GPU memory at K=256, d=64, chunk 131072: 21.9 GiB of 44.4 GiB** (smoke run,
both GPUs) — comfortable.

**Runtime estimate ~4.5–5 h.** Per-sweep cost is dominated by `B·2048·K·d` (assignment) plus
`B·2048·2048` (moment accumulation); K=128→256 takes that ratio from 10240 to 18432, i.e. **≈1.8×**
v6's 367 s/iter → ~660 s/iter → ~275 min for 25 iters, plus ~7 min load. Inside CLAUDE.md's ≤6 h
budget, but with little margin — drop to `--iters 20` if the estimate proves optimistic (v6 had
already crossed `obj/token` 1892 by iter 20).

### Stage B — `src/clustering/merge_clusters.py` (new)

Reads a finished run dir, writes a new one. No GPU strictly needed for the cascade, but it uses one
if present.

```
--dir              input run directory (model.pt; assignments.pt only for --refit-iters 0)
--out              output run directory
--criterion        residual (default) | affinity
--merge-threshold  τ: stop when the cheapest merge costs more than τ (default 0.002)
--target-k         alternatively/additionally merge down to exactly this K
--refit-iters      real assignment/update sweeps after the cut (default 3; 0 = no data pass)
--min-count        never merge below this size guard / always merge clusters under it
--dump-dendrogram  write the full 256→1 trace regardless of where the cut lands
```

**Threshold semantics (the configurable knob).** Cost is reported **relative to the run's own
objective**:

```
cost(a,b) = ΔR(a,b) / R_total_initial          ΔR = R_{a∪b} − R_a − R_b  ≥ 0
```

so `τ = 0.002` means *"accept a merge that degrades the objective by less than 0.2%"*. This is
directly interpretable against the numbers the project already quotes: the v6/v7/v8 seed spread is
**0.24%**, so `τ ≈ 0.002` is "merge anything whose cost is below run-to-run seed noise" — a
defensible default, and the natural reference point for choosing any other value. Under
`--criterion affinity` the threshold is instead a principal-angle affinity in [0,1] (merge while the
best pair exceeds it, e.g. 0.9, matching the existing `maxAff` health flag).

`merge_log.json` records every step (`pair`, `cost`, `cumulative_cost`, `sizes`, resulting K) for the
**whole** cascade, so the threshold can be re-chosen from the log without recomputation. Greedy Ward
on a surrogate is not guaranteed inversion-free, so the cut is taken at the **first** step whose cost
exceeds τ (not the last), and any inversions are flagged in the log.

### Stage C — refit (data pass, reuses existing kernels)

`merge_clusters.py` imports `run_sweep` / `update_model` from `subspace_kmeans.py` (same directory,
so a plain `import subspace_kmeans` works alongside the standard `src/` bootstrap) and runs
`--refit-iters` sweeps on the **identical token sample** (`--files-from <dir>/sample.json`, same seed
and `--tokens-per-file`), then a final relabel sweep exactly as `subspace_kmeans.main()` does. This:

- replaces every surrogate basis with an exact PCA basis from real data,
- reconciles `counts` to `bincount(label)` (the cross-algorithm invariant),
- produces a truthful `final_obj_per_token` for the merged model.

Output is a **schema-valid `model.pt` + `assignments.pt`** via `cluster_io.save_model` /
`save_assignments`, with `config["method"] = "subspace_kmeans_merged"` and a `merge` block recording
the parent run, fingerprint, criterion, τ, and merge count. Because the schema holds, the entire
existing analysis stack — `analyze_clusters.py`, `holdout_eval.py`, `temporal_spatial.py`,
`file_signature.py` — works on the merged run unchanged. That is a hard requirement, not a bonus.

---

## 4. Smoke tests (before the 5 h run)

| # | Test | Passes if |
|---|---|---|
| S1 | **Synthetic split/merge.** Build tokens on known subspaces, split one cluster artificially in two, run the cascade. | The artificial split is the *first* merge, at cost ≈ 0; genuinely distinct clusters cost orders of magnitude more. |
| S2 | **Algebra.** Merged `μ`, `tr(C)`, `n` from the update formulas vs. brute-force recomputation over the member tokens. | Match to float tolerance (exact identities). |
| S3 | **Surrogate accuracy.** PPCA-surrogate `ΔR` vs exact `ΔR` from a full 2048×2048 `eigh` on real tokens (small K, small d). | Small relative error, and — what actually matters — **preserved pair ranking**. |
| S4 | **Schema + tooling.** Merged output through `save_model` asserts, then `analyze_clusters.py --dir`. | Report generates; `counts == bincount(label)`; `U` orthonormal; `explained_var_ratio ∈ [0,1]`. |
| S5 | **Degenerate cuts.** `--target-k K` (no-op) and `--target-k 1`. | No-op reproduces the input model; K=1 terminates cleanly. |
| S6 | **End-to-end mini.** Full pipeline on ~30 files, K=16→8, d=8, with refit. | Runs; merged objective ≥ parent's; refit lowers it monotonically. |

S1/S2/S5 are data-free and run in seconds. S3/S6 use a small real sample (~30 files, ~1.5 GB).

---

## 5. Experiment

The scientific payoff is a **controlled A/B against v6 on the identical token sample**:

| run | K | d | how |
|---|---|---|---|
| v6 (existing) | 128 | 64 | direct fit | 
| v11 (new) | 256 | 64 | direct fit |
| v12 (new) | 128 | 64 | **v11 merged 256→128, then refit** |

`v12` vs `v6` is the whole hypothesis, and both are measured by `final_obj_per_token` on the same
7000-file sample. Reference points already established: v6 = 1887.83; the seed spread across
v6/v7/v8 is 1887.8/1892.3/1888.2, i.e. **0.24%** — so **v12 must beat v6 by more than ~0.25% to be
a real effect and not a lucky init.** Secondary reads: NMI(v12, v6) against the known seed-to-seed
baseline (v6↔v7 0.683, v6↔v8 0.688); cluster health flags; and the merge-cost curve itself, whose
knee is an *empirical estimate of the natural cluster count* — the first thing in this project to
speak to "is K=128 right?" rather than assuming it.

Threshold-driven cuts (τ = 0.002) run from the same dendrogram at no extra cost, so we get both the
"merge to exactly 128" comparison and the "merge until it hurts" answer.

## 6. Order of work

1. `merge_clusters.py` + smoke tests S1–S6. ← cheap, no long run blocked on it
2. Launch stage A (K=256, ~5 h) as soon as the smoke tests pass.
3. Merge → refit → `analyze_clusters.py` → compare against v6.
4. Update `README.md` + `CLAUDE.md` (repo rule: every script documented).
