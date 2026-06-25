# Weather-Encoder Latent Analysis

Structure analysis of token embeddings produced by a weather encoder trained on ERA5.
The dataset (`latents_2/`, 1.2 TB) holds 13,021 files `latent_{i}.pt`, one per sample
(time step), each a dict `{idx: int64 scalar, latent: float32 [1, 12288, 2048]}` —
12,288 HEALPix cells (nside=32, **NESTED ordering**) × 2048-dim token per cell, ≈160 M tokens total.

**Time axis:** the files are ERA5 states at **6-hourly cadence (4/day: 00/06/12/18 UTC)**.
Each file stores only an integer `idx` (its sample index), **not a timestamp**, so the
calendar date is reconstructed positionally as `datetime = 2014-01-01 00:00 UTC + idx×6h`
(file 0 = 2014-01-01 00:00; file 13020 = 2022-11-30 00:00 — a full 9-year 2014→2022 ERA5
range would run to 2022-12-31 18:00, ~127 more steps, so the last ~month of 2022 is not
fully covered). `temporal_spatial.py` groups files into calendar months this way.

## Scripts

### `cluster_io.py` — shared sampling/IO for all clustering algorithms

Every clustering script (`subspace_kmeans.py`, and any k-means / k-center scripts) is
expected to import this module rather than reimplement sampling or output, so that runs
from different algorithms can be compared directly:

- `load_tokens(args)` — samples `--num-files` latent files uniformly at random (or
  reuses an exact file list via `--files-from`), loads them in parallel, and writes
  `sample.json` (the reproducible manifest: fingerprint, seed, tokens-per-file, file
  list in load order).
- `sample_fingerprint(files, tokens_per_file, seed)` / `load_file_list(path)` — used to
  identify/reuse a token sample.
- `save_model(out_dir, **fields)` — validates shapes and writes `model.pt` under the
  shared schema below.
- `save_assignments(out_dir, file_id, cell_id, label)` — writes `assignments.pt`.

**`model.pt` contract** (required regardless of algorithm): `U [K, 2048, d]` (`d=0` for
plain point clusters — k-means/k-center have no subspace, just a centroid), `means
[K, 2048]`, `eigvals [K, d]`, `trace [K]` (mean squared distance to centroid — what
k-means/subspace_kmeans optimize), `counts [K]`, `explained_var_ratio [K]` (0 when
`d=0`), `config` (must include `config["method"]` ∈ `{"kmeans", "kcenter",
"subspace_kmeans"}`), per-iteration `history`, `sampled_files`, `sample_fingerprint`.
Optional: `radius [K]` — k-center's native minimax objective (max distance from
centroid to any member), populated only by k-center.

K-means is literally the `d=0` case of K-subspaces clustering: the orthogonal-residual
assignment formula collapses to plain squared distance to centroid when there's no
basis. `subspace_kmeans.py --dim 0` exercises this path directly.

`assignments.pt` contract (all algorithms): `file_id` / `cell_id` / `label` (int32) for
every sampled token.

### `subspace_kmeans.py` — K-subspaces clustering

K-means generalized to affine subspaces: each cluster is a mean μⱼ plus an orthonormal
basis `Uⱼ [2048, d]`, and each token is assigned to the cluster with the smallest
orthogonal residual ‖x−μⱼ‖² − ‖Uⱼᵀ(x−μⱼ)‖².

Algorithm details:

- **One streaming sweep per iteration** over the sampled tokens (held in RAM as fp16),
  split across both A100s (one worker thread per GPU). Assignment and the per-cluster
  second-moment accumulation happen in the same sweep.
- **Exact per-cluster PCA**: bases come from batched `torch.linalg.eigh` of the
  2048×2048 cluster covariances — no randomized SVD or power iterations needed,
  because the covariances are cheap to accumulate on GPU during the sweep.
- **Init**: K random tokens as seeds with zero bases (first pass = nearest-seed k-means
  step). A random partition does *not* work — all K bases collapse onto the global PCA.
- **Tiny/empty clusters are re-seeded by splitting the largest healthy cluster** along
  its top principal axis (both halves inherit that cluster's subspace, offset ±1 std
  along PC1). A plain random-token re-seed (`U=0`) cannot recover at large `d` — a fresh
  point-seed has residual `‖x−μ‖²` while every rival owns a `d`-dim subspace with near-zero
  residual, so it never wins more than its one starter token (observed: `d=64` strands 1
  cluster, `d=128` strands 2 at size 1 forever). The split keeps all K subspaces populated.
  The `d=0` (plain k-means) path has no subspace to split, so it keeps the random-token re-seed.

Token selection: `--num-files` files drawn uniformly at random, `--tokens-per-file`
random cells from each (default: all 12,288). Every file covers the whole globe, so
file subsampling only thins the time axis. RAM = `num_files × tokens_per_file × 4096` bytes
(the fp16 `data` buffer; `--max-ram-gb` guards against this). True peak is a little higher —
the int32 `file_id`/`cell_id` arrays (~8 bytes/token) plus transient float32 decode buffers
(~`--load-workers` × 100 MB) — so leave headroom.

```bash
# defaults: 1500 files (~75 GB RAM), K=64, d=16, ≤25 iterations  — ~7 min wall time
nohup python3 subspace_kmeans.py --out subspace_kmeans_runs/v1_subspace_out \
    > subspace_kmeans_runs/v1_subspace_out/run.log 2>&1 &

# large run: 7000 files (~352 GB RAM), K=128, d=32  — ~1-1.5 h wall time
nohup python3 subspace_kmeans.py --num-files 7000 --clusters 128 --dim 32 \
    --iters 40 --max-ram-gb 420 --out subspace_kmeans_runs/v2_subspace_big \
    > subspace_kmeans_runs/v2_subspace_big/run.log 2>&1 &
```

**Reproducible sampling & cross-run comparison.** Every run writes `sample.json`
(a manifest with seed, tokens-per-file, the file-id list in load order, and a short
`fingerprint`). To cluster a *different* configuration on the **exact same tokens** —
the only fair way to compare, e.g. K or `d` or iteration count — reuse that sample:

```bash
python3 subspace_kmeans.py --files-from subspace_kmeans_runs/v2_subspace_big/sample.json \
    --clusters 128 --dim 32 --iters 100 --out subspace_kmeans_runs/v3_subspace_big_i100
```

`--files-from` accepts a `sample.json` or a `model.pt`; it overrides `--num-files`.
Runs that share a fingerprint were clustered on identical tokens and are directly
comparable (the report prints the fingerprint), even across algorithms. The token set is
order-independent, but load order — and therefore the random init — is preserved, so
reusing a sample with the same `--seed`/`--tokens-per-file` reproduces a run *exactly*.
`analyze_clusters.py` backfills `sample.json` for older runs that predate the manifest.

Outputs (in `--out`), via `cluster_io.py`'s `save_model`/`save_assignments` — see the
shared contract above:

| file | contents |
|---|---|
| `model.pt` | `U [K, 2048, d]` orthonormal bases (descending eigenvalue order), `means [K, 2048]`, `eigvals [K, d]`, `trace [K]`, `counts [K]`, `explained_var_ratio [K]`, `config`, per-iteration `history`, `sampled_files`, `sample_fingerprint` |
| `assignments.pt` | `file_id` / `cell_id` / `label` (int32) for every sampled token — maps each assignment back to its HEALPix cell and time step |
| `sample.json` | reproducible manifest (fingerprint, seed, tokens-per-file, file-id list in load order) — feed to `--files-from` |

Project a token onto its cluster subspace with `(x - means[j]) @ U[j]`.

### `analyze_clusters.py` — Markdown report generator

Algorithm-agnostic: reads any `model.pt`/`assignments.pt` following the `cluster_io.py`
schema, from `subspace_kmeans.py` or from k-means/k-center (the `d=0` point-cluster
case). Turns a result directory into a self-describing report (`report.md`):
configuration, convergence table, global variance decomposition (between-cluster /
subspace-captured / residual), a per-cluster table (size, spatial concentration over
HEALPix cells, temporal coverage and variation, plus EVR/effective-dimensionality when
`d>0`, plus k-center's `radius` when present), and pairwise subspace affinity (`d>0`
only — mean squared principal-angle cosines, flags merge candidates).

```bash
python3 analyze_clusters.py --dir subspace_kmeans_runs/v1_subspace_out \
    --out subspace_kmeans_runs/v1_subspace_out/report.md
```

The **world map and the temporal/seasonal analysis live in a separate report** — see
`temporal_spatial.py` below; `analyze_clusters.py` keeps the compact per-cluster
spatial/temporal columns and a pointer to `temporal_report.md`. Every metric in
`report.md` is defined and interpreted in **[METRICS.md](METRICS.md)**.

### `worldmap.py` — shared HEALPix geometry, cluster coloring, and map renderer

Imported by both `analyze_clusters.py` and `temporal_spatial.py`, so the two reports
share one geometry, one cluster-color assignment, and one map renderer:

- Pure-numpy `healpix_nest2ring` / `healpix_ring_lonlat` — cell id → lon/lat (no healpy).
  Validated data-free: `nest2ring` is a bijection of `[0, 12288)`, lon/lat are clean
  (lat ∈ [−88.5°, 88.5°]), and the equal-area cell distribution matches analytics (50.0%
  at |lat|<30°; 13.7% at |lat|>60° vs 13.4% theoretical).
- `build_affinity_matrix` (subspace affinity for `d>0`, centroid cosine for `d=0`) and
  `affinity_ordered_colors` — **spectral seriation** of the affinity matrix via its Fiedler
  vector, mapped through the `turbo` colormap, so subspace-similar clusters get similar
  hues. Real structure then reads as smooth gradients, not the false "scatter" a random
  hue shuffle produces at large K.
- `get_coastlines()` — fetches + caches the Natural Earth 110m coastline GeoJSON (parsed
  with stdlib `json`; no cartopy/shapely, which aren't installed on this box).
- `render_world_map()` — a Mollweide map of the dominant cluster per cell with two
  readability features: **continent outlines**, and a smooth **heatmap** (the per-cell RGB
  is interpolated with `scipy.griddata` onto a regular grid, lightly Gaussian-smoothed, and
  painted with `pcolormesh`) that replaces the old speckled 12,288-pixel scatter with a
  continuous, coast-aligned field. Interpolating RGB (not the categorical cluster id) is
  valid precisely because `affinity_ordered_colors` already makes neighbors similar. ~0.8 s/map.

### `temporal_spatial.py` — temporal & spatial report

Dedicated report (`temporal_report.md`) for the spatial and temporal structure at calendar
(monthly) resolution:

- an **annual** dominant-cluster world map + **12 monthly** maps (continent outlines +
  smooth heatmap via `worldmap.py`), all on one shared color scale so months are comparable;
- a **monthly enrichment** table flagging the most seasonal clusters (1.0 = year-round,
  ≫1 = concentrated in those months);
- a **month-to-month stability** curve — the share of cells whose dominant cluster flips
  between consecutive months — plus a Jan↔Jul (winter vs summer) shift.

```bash
python3 temporal_spatial.py --dir subspace_kmeans_runs/v6_subspace_big_d64
```

It reconstructs each file's calendar month from `datetime = 2014-01-01 00:00 + idx×6h` and
reuses the run's **existing** `assignments.pt` (v6 covers all 12,288 cells in every
calendar month, ~540–620 files/month) — so there is **no recomputation**: it's a ~10 s
rendering/reporting pass. For a run that under-samples some month, re-cluster with more
`--num-files` and re-run.

### `holdout_eval.py` — generalization check on unseen files

The variance decomposition in the report is measured on the tokens the model was *fit*
on. With `d`-dim per-cluster bases the model has many free parameters (K·2048·d), so an
in-sample residual could be optimistic. This script freezes the trained means and bases
and replays the **exact** assignment rule (`‖x−μ_j‖² − ‖Uⱼᵀ(x−μ_j)‖²`) on tokens from
latent files the run never saw (sampled disjoint from `sampled_files`), then reports the
held-out residual fraction next to the in-sample one. A small gap ⇒ the subspaces capture
reusable structure; a large positive gap ⇒ overfitting (lower `--dim` or add tokens).

```bash
python3 holdout_eval.py --dir subspace_kmeans_runs/v6_subspace_big_d64 --num-files 200
```

Writes `<dir>/holdout.json`; the next `analyze_clusters.py` run renders a **Held-out
generalization** section from it automatically. (v6, d=64: held-out 31.3% vs in-sample
31.5% — the subspaces generalise.)

### Chained run (fire and forget)

```bash
out=subspace_kmeans_runs/v2_subspace_big; mkdir -p "$out"
nohup bash -c "python3 subspace_kmeans.py --num-files 7000 --clusters 128 --dim 32 \
  --iters 40 --max-ram-gb 420 --out $out && \
  python3 analyze_clusters.py --dir $out --out $out/report.md" \
  > $out/run.log 2>&1 &
```

## Results so far

Runs live under `subspace_kmeans_runs/`, each in a `vI_<name>` directory numbered in
chronological order (v1 → v5 below).

- `v1_subspace_out/` — first full run (1500 files, K=64, d=16,
  6.7 min). Key findings: clusters are **spatially localized but temporally universal**
  (geographic regimes — each present in ~100% of time steps but concentrated in a few
  hundred of the 12,288 cells); cluster identity alone explains 8.5% of token variance,
  the top-16 subspaces a further 33.5%; within-cluster spectra are fairly flat
  (d80 ≈ 10–12 of 16), motivating larger `--dim`.
- **The encoder's HEALPix cell indexing is NESTED** (established from
  `v1_subspace_out/dominant_cluster_map.png`: coherent continent-scale regions under the
  NESTED interpretation, incoherent stripes under RING).
- `v2_subspace_big/` — larger run (7000 files ≈ 86 M tokens, K=128, d=32, 65 min,
  fingerprint `82ca602ed7e7`). Within-cluster EVR rose to 0.49 (d=32 captures more);
  geographic structure sharpened into clear latitude bands + continents.
- `v3_subspace_big_i100/` — same sample as v2 (fingerprint `82ca602ed7e7`),
  K=128 d=32 but **100 iterations**, to see how much further the objective settles past
  iter 40 (it was still at ~0.9% labels-changed). Queued 2026-06-12.
- **The K=128 map looked "more scattered" than K=64 — but that was a visualization
  artifact, not worse clustering.** Measured neighbor-agreement (share of adjacent cells
  with the same dominant cluster) was 0.72 at K=64 vs 0.60 at K=128, yet *relative to
  chance* the big run is more structured (46× vs 77× the 1/K baseline), and its per-cell
  mode-purity is higher (0.31 → 0.35). Doubling K just doubles region boundaries and a
  random hue shuffle aliased the finer sub-regions; the affinity-ordered colormap (above)
  restores smooth gradients.
- `v4_subspace_big_d64/` — same sample as v2 (fingerprint `82ca602ed7e7`), K=128 but
  d=64. Count-weighted within-cluster EVR(top-64) rose to 0.664 and residual variance
  dropped to 31.5%; d80 (min/median/max 28/37/65) still close to d, so the spectrum stays
  fairly flat — larger `--dim` keeps paying off.
- `v5_subspace_big_d128/` — same sample as v2 (fingerprint `82ca602ed7e7`), K=128, d=128.
  EVR(top-128) reached 0.816, residual down to 17.8%; d80 (min/median/max 47/63/129) is
  again close to d, so within-cluster structure is still not fully captured even at
  d=128. **Note:** 2 clusters were stranded at size 1 — a degeneracy of large `d` with the
  old random-token re-seed, *not* of large K (d=32 at the same K is singleton-free); see v6.
- `v6_subspace_big_d64/` — **supersedes v4**: identical config (same sample, K=128, d=64)
  but with the split-largest re-seed guard. The one cluster v4 stranded at size 1 is
  rescued: all 128 clusters healthy (min 227,886 tokens), and `max d80` drops 65→**40**
  (every cluster now needs ≤40 of its 64 dims for 80% of captured variance — the spectrum
  is no longer truncated, unlike d=32 where d80≈d). Variance decomposition is otherwise
  identical to v4 (8.4% between / 60.1% within / 31.5% residual), confirming the guard only
  fixed the degenerate cluster. **This is the recommended config**: d=64 captures the
  within-cluster structure without the singleton degeneracy and washed-out cluster identity
  (between-cluster share) that d=128 causes.
- The **temporal & spatial report** (`temporal_spatial.py` → `temporal_report.md`) breaks
  v6 down by calendar month: clusters 24/82/73/104 peak in NH summer, 7/29/67 in winter;
  month-to-month dominant-cluster flips range 6.2% (Jul→Aug) to 21.5% (Apr→May), and
  Jan↔Jul differ in 45% of cells — a clear hemispheric seasonal cycle. Maps reuse v6's
  existing assignments (no re-clustering).

## Hardware notes (this machine)

- 48-core CPU, 512 GB RAM, 2× A100 40 GB, `/usr/bin/python3` + PyTorch 2.6.0 (no venv).
- **GPU↔GPU peer copies are silently broken**: `tensor.to()` between `cuda:0` and
  `cuda:1` returns zeros/garbage with no error, although `can_device_access_peer`
  reports True. Route all inter-GPU transfers through CPU, and do device-to-host copies
  from the thread that launched the producing kernels. Both scripts follow this rule.

## Legacy

`JL-Downscaling/` (Johnson–Lindenstrauss projection experiments) and
`latents_downscaled/` (its output) predate this work and are unrelated.
