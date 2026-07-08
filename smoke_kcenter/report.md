# Clustering report (kcenter) — `smoke_kcenter`

*Generated 2026-07-08 15:35 by `analyze_clusters.py`. K=16 point clusters in 2048-dim token space, 40,000 tokens.*

## Overview

The model groups the 40,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 16 point clusters, each summarised by a single centroid. A token is assigned to the nearest centroid (plain squared distance).

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
- **trace** (`model['trace'][j]`): mean squared distance of cluster *j*'s tokens to its centroid μⱼ — the cluster's total within-cluster variance.
- **counts** (`model['counts'][j]`): number of tokens in cluster *j*; **wⱼ = counts[j] / Σcounts** is its population share, used to weight every global average.

## Configuration

*How to read this: these are the run's input settings, taken from `model['config']` (plus `model['sampled_files']` for the true file count). `clusters` is K, `dim` is the subspace dimension d (`dim=0` ⇒ plain k-means). `iters` is the maximum number of training iterations (each one reassigns every token to its best cluster, then refits the centroids and subspaces); `tol` is the convergence threshold: once the fraction of tokens that change cluster in an iteration falls below it, the run stops early instead of using all `iters`. Together they bound how long the run takes. `seed` fixes both the token sample and the cluster initialisation so a run is reproducible.*

| parameter | value |
|---|---|
| src | /home/mremane/embeddings/latents_2 |
| num_files | 20 |
| tokens_per_file | 2000 |
| clusters | 16 |
| seed | 0 |
| chunk_size | 262144 |
| gpu | 0 |
| tokens analyzed | 40,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `478a7befe373`
- **Files:** 20 latent files, 2000 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 kcenter.py --files-from smoke_kcenter/sample.json --seed 0 --tokens-per-file 2000 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 20, full list in `smoke_kcenter/sample.json`): 673, 1741, 1973, 2253, 2927, 3623, 3804, 4023, 4960, 5768, 5790, 6802, 7971, 8500, 8900, 9478, 9519, 12075, 12538, 12856

## Convergence

*How to read this: one row per centre added by the Gonzalez farthest-point greedy, from `model['history']`. **covering radius** is the max distance from any token to its nearest chosen centre so far — k-center's native minimax objective; it shrinks (non-strictly) as centres are added. **objective/token** is the mean squared distance to the nearest centre so far, shown for comparison with k-means/subspace_kmeans's objective.*

| centre | covering radius | objective/token |
|---|---|---|
| 1 | 155.94 | 12542.11 |
| 2 | 155.60 | 12540.43 |
| 3 | 152.56 | 12533.25 |
| 4 | 152.26 | 12528.79 |
| 5 | 150.90 | 12525.83 |
| 6 | 150.05 | 12519.95 |
| 7 | 148.41 | 12517.96 |
| 8 | 146.70 | 12512.57 |
| 9 | 146.12 | 12508.64 |
| 10 | 145.85 | 12498.48 |
| 11 | 144.77 | 12495.39 |
| 12 | 144.56 | 12475.79 |
| 13 | 142.83 | 12472.58 |
| 14 | 142.30 | 12468.27 |
| 15 | 141.36 | 12465.04 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **13426**, split into:

- **7.2%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **92.8%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 40,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files` = share of the 20 sampled time steps (latent files, 6-hourly) in which the cluster appears at least once. ≈100% ⇒ always present in time.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*
- *`radius` = k-center's native objective: the max distance from the centroid to any member (a worst-case, not an average like `trace`).*

Spatial columns are over the 11936 HEALPix cells with data; `cells@50%` = number of cells holding half the cluster's tokens (low = localized); `owned` = cells where this cluster is the most common label; `files` = share of the 20 sampled time steps where the cluster appears; `tCV` = coefficient of variation of its share across time deciles (0 = constant in time). `radius` = k-center's native objective, the max distance from centroid to any member (vs. `tCV`-adjacent `trace`, the mean squared distance k-means/subspace optimize).

| cluster | tokens | share | cells@50% | owned | files | tCV | radius |
|---|---|---|---|---|---|---|---|
| 0 | 38,177 | 95.4% | 3822 | 11826 | 100% | 0.01 | 141.36 |
| 12 | 462 | 1.2% | 186 | 35 | 100% | 0.20 | 139.58 |
| 10 | 271 | 0.7% | 126 | 13 | 95% | 0.22 | 138.01 |
| 6 | 154 | 0.4% | 60 | 13 | 85% | 0.63 | 139.79 |
| 13 | 137 | 0.3% | 67 | 4 | 100% | 0.61 | 134.65 |
| 14 | 127 | 0.3% | 60 | 4 | 100% | 0.22 | 139.74 |
| 15 | 127 | 0.3% | 62 | 5 | 100% | 0.36 | 138.29 |
| 9 | 126 | 0.3% | 60 | 7 | 95% | 0.48 | 139.71 |
| 8 | 120 | 0.3% | 54 | 8 | 100% | 0.50 | 141.04 |
| 4 | 84 | 0.2% | 42 | 6 | 95% | 0.29 | 135.42 |
| 11 | 74 | 0.2% | 36 | 3 | 80% | 0.51 | 140.71 |
| 3 | 72 | 0.2% | 36 | 4 | 100% | 0.54 | 137.19 |
| 5 | 32 | 0.1% | 15 | 2 | 50% | 1.66 | 137.84 |
| 7 | 17 | 0.0% | 9 | 3 | 40% | 1.48 | 139.64 |
| 2 | 11 | 0.0% | 6 | 2 | 50% | 0.96 | 135.91 |
| 1 | 9 | 0.0% | 5 | 1 | 35% | 0.83 | 136.91 |

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 temporal_spatial.py --dir smoke_kcenter --out smoke_kcenter/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *Large `radius` relative to `trace`* ⇒ an outlier-driven cluster; k-center minimizes the worst case, so a few far points can still leave `radius` high even with low average (`trace`) variance.
