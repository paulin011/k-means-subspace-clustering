# Clustering report (subspace_kmeans) — `smoke_kmeans`

*Generated 2026-07-08 10:55 by `analyze_clusters.py`. K=16 point clusters in 2048-dim token space, 40,000 tokens.*

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
| dim | 0 |
| iters | 5 |
| tol | 0.001 |
| linear | False |
| seed | 0 |
| chunk_size | 262144 |
| gpus | 2 |
| tokens analyzed | 40,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `478a7befe373`
- **Files:** 20 latent files, 2000 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 subspace_kmeans.py --files-from smoke_kmeans/sample.json --seed 0 --tokens-per-file 2000 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 20, full list in `smoke_kmeans/sample.json`): 673, 1741, 1973, 2253, 2927, 3623, 3804, 4023, 4960, 5768, 5790, 6802, 7971, 8500, 8900, 9478, 9519, 12075, 12538, 12856

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 8651.29 | 100.00% | 1 | 14,633 |
| 2 | 5703.66 | 36.86% | 1 | 9,941 |
| 3 | 5636.92 | 17.61% | 1 | 7,886 |
| 4 | 5613.30 | 10.74% | 1 | 7,224 |
| 5 | 5602.01 | 7.47% | 2 | 7,095 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **6051**, split into:

- **7.3%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **92.7%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 40,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files` = share of the 20 sampled time steps (latent files, 6-hourly) in which the cluster appears at least once. ≈100% ⇒ always present in time.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 11936 HEALPix cells with data; `cells@50%` = number of cells holding half the cluster's tokens (low = localized); `owned` = cells where this cluster is the most common label; `files` = share of the 20 sampled time steps where the cluster appears; `tCV` = coefficient of variation of its share across time deciles (0 = constant in time).

| cluster | tokens | share | cells@50% | owned | files | tCV |
|---|---|---|---|---|---|---|
| 7 | 7,139 | 17.8% | 1489 | 2290 | 100% | 0.07 |
| 2 | 6,057 | 15.1% | 1213 | 2203 | 100% | 0.06 |
| 8 | 5,815 | 14.5% | 1376 | 1412 | 100% | 0.04 |
| 0 | 5,255 | 13.1% | 1327 | 2598 | 100% | 0.07 |
| 13 | 4,865 | 12.2% | 1177 | 860 | 100% | 0.16 |
| 12 | 4,283 | 10.7% | 977 | 870 | 100% | 0.12 |
| 9 | 4,130 | 10.3% | 1090 | 750 | 100% | 0.10 |
| 1 | 2,427 | 6.1% | 778 | 950 | 100% | 0.08 |
| 15 | 15 | 0.0% | 7 | 1 | 30% | 1.50 |
| 4 | 3 | 0.0% | 2 | 0 | 5% | 2.83 |
| 11 | 3 | 0.0% | 2 | 0 | 5% | 2.83 |
| 3 | 2 | 0.0% | 1 | 1 | 5% | 2.83 |
| 6 | 2 | 0.0% | 1 | 0 | 10% | 1.87 |
| 14 | 2 | 0.0% | 1 | 0 | 5% | 2.83 |
| 5 | 1 | 0.0% | 1 | 1 | 5% | 2.83 |
| 10 | 1 | 0.0% | 1 | 0 | 5% | 2.83 |

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 temporal_spatial.py --dir smoke_kmeans --out smoke_kmeans/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
