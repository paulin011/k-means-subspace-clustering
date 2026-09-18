# Clustering report (kcenter) — `runs/clustering/v16_kcenter_seed1`

*Generated 2026-09-14 16:47 by `analyze_clusters.py`. K=128 point clusters in 2048-dim token space, 86,016,000 tokens.*

## Overview

The model groups the 86,016,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 128 point clusters, each summarised by a single centroid. A token is assigned to the nearest centroid (plain squared distance).

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
- **trace** (`model['trace'][j]`): mean squared distance of cluster *j*'s tokens to its centroid μⱼ — the cluster's total within-cluster variance.
- **counts** (`model['counts'][j]`): number of tokens in cluster *j*; **wⱼ = counts[j] / Σcounts** is its population share, used to weight every global average.

## Configuration

*How to read this: these are the run's input settings, taken from `model['config']` (plus `model['sampled_files']` for the true file count). `clusters` is K, `dim` is the subspace dimension d (`dim=0` ⇒ plain k-means). `iters` is the maximum number of training iterations (each one reassigns every token to its best cluster, then refits the centroids and subspaces); `tol` is the convergence threshold: once the fraction of tokens that change cluster in an iteration falls below it, the run stops early instead of using all `iters`. Together they bound how long the run takes. `seed` fixes both the token sample and the cluster initialisation so a run is reproducible.*

| parameter | value |
|---|---|
| src | latents_2 |
| num_files | 7000 *(--files-from reused the sample; --num-files=1500 ignored)* |
| tokens_per_file | 12288 |
| clusters | 128 |
| seed | 0 |
| chunk_size | 262144 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `82ca602ed7e7`
- **Files:** 7000 latent files, 12288 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/kcenter.py --files-from runs/clustering/v16_kcenter_seed1/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v16_kcenter_seed1/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 9744.21 | 100.00% | 196 | 84,949,015 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **3624**, split into:

- **6.8%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **93.2%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*
- *`radius` = k-center's native objective: the max distance from the centroid to any member (a worst-case, not an average like `trace`).*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.00%–98.76% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | cells@50% | owned | files@50% | tCV | radius |
|---|---|---|---|---|---|---|---|
| 0 | 84,949,016 | 98.76% | 6085 | 12288 | 49.8% | 0.001 | 146.37 |
| 85 | 165,703 | 0.19% | 1090 | 0 | 25.8% | 0.073 | 140.45 |
| 44 | 75,299 | 0.09% | 447 | 0 | 16.3% | 0.106 | 137.45 |
| 25 | 54,873 | 0.06% | 507 | 0 | 15.3% | 0.133 | 138.37 |
| 115 | 40,787 | 0.05% | 678 | 0 | 22.5% | 0.043 | 136.73 |
| 126 | 31,697 | 0.04% | 657 | 0 | 18.1% | 0.099 | 136.65 |
| 121 | 31,571 | 0.04% | 283 | 0 | 17.7% | 0.085 | 141.49 |
| 123 | 30,257 | 0.04% | 504 | 0 | 20.7% | 0.050 | 137.12 |
| 109 | 28,070 | 0.03% | 88 | 0 | 8.2% | 0.264 | 138.86 |
| 107 | 22,392 | 0.03% | 379 | 0 | 16.1% | 0.193 | 138.51 |
| 122 | 19,810 | 0.02% | 330 | 0 | 7.9% | 0.227 | 137.49 |
| 67 | 19,723 | 0.02% | 86 | 0 | 6.0% | 0.335 | 138.74 |
| 112 | 18,750 | 0.02% | 203 | 0 | 7.1% | 0.160 | 139.46 |
| 71 | 18,641 | 0.02% | 232 | 0 | 12.4% | 0.127 | 143.80 |
| 15 | 17,695 | 0.02% | 492 | 0 | 19.4% | 0.040 | 138.03 |
| 88 | 14,905 | 0.02% | 332 | 0 | 13.7% | 0.336 | 138.17 |
| 118 | 13,816 | 0.02% | 35 | 0 | 6.9% | 0.319 | 139.66 |
| 65 | 13,584 | 0.02% | 176 | 0 | 10.5% | 0.121 | 136.03 |
| 62 | 12,647 | 0.01% | 393 | 0 | 15.2% | 0.182 | 138.76 |
| 59 | 12,612 | 0.01% | 268 | 0 | 10.5% | 0.139 | 139.28 |
| 79 | 12,308 | 0.01% | 73 | 0 | 8.1% | 0.217 | 143.54 |
| 60 | 11,802 | 0.01% | 157 | 0 | 10.6% | 0.117 | 137.00 |
| 100 | 11,405 | 0.01% | 294 | 0 | 18.2% | 0.183 | 137.52 |
| 55 | 11,363 | 0.01% | 532 | 0 | 14.8% | 0.095 | 137.34 |
| 120 | 10,727 | 0.01% | 170 | 0 | 8.3% | 0.242 | 140.68 |
| 57 | 10,681 | 0.01% | 330 | 0 | 9.7% | 0.232 | 137.46 |
| 98 | 10,645 | 0.01% | 6 | 0 | 6.7% | 0.132 | 135.23 |
| 39 | 10,186 | 0.01% | 200 | 0 | 9.0% | 0.172 | 138.65 |
| 8 | 9,801 | 0.01% | 341 | 0 | 12.5% | 0.178 | 136.83 |
| 38 | 9,323 | 0.01% | 344 | 0 | 11.8% | 0.136 | 141.88 |
| 84 | 8,811 | 0.01% | 233 | 0 | 10.3% | 0.153 | 139.67 |
| 86 | 7,935 | 0.01% | 318 | 0 | 10.6% | 0.203 | 144.44 |
| 70 | 7,757 | 0.01% | 113 | 0 | 13.5% | 0.090 | 137.35 |
| 91 | 7,424 | 0.01% | 356 | 0 | 14.8% | 0.117 | 138.55 |
| 34 | 7,385 | 0.01% | 205 | 0 | 12.3% | 0.137 | 138.19 |
| 13 | 7,357 | 0.01% | 426 | 0 | 5.2% | 0.165 | 137.37 |
| 43 | 7,219 | 0.01% | 205 | 0 | 7.3% | 0.273 | 138.67 |
| 3 | 6,689 | 0.01% | 376 | 0 | 8.9% | 0.188 | 136.40 |
| 27 | 6,621 | 0.01% | 297 | 0 | 10.5% | 0.164 | 139.07 |
| 36 | 6,265 | 0.01% | 307 | 0 | 8.3% | 0.244 | 141.06 |
| 76 | 6,202 | 0.01% | 195 | 0 | 10.8% | 0.145 | 135.77 |
| 75 | 6,191 | 0.01% | 135 | 0 | 8.0% | 0.234 | 138.26 |
| 74 | 6,014 | 0.01% | 231 | 0 | 7.2% | 0.151 | 136.37 |
| 61 | 5,891 | 0.01% | 330 | 0 | 6.1% | 0.186 | 136.25 |
| 127 | 5,860 | 0.01% | 190 | 0 | 10.6% | 0.172 | 140.55 |
| 105 | 5,672 | 0.01% | 237 | 0 | 6.0% | 0.463 | 139.73 |
| 26 | 5,365 | 0.01% | 223 | 0 | 7.5% | 0.185 | 137.95 |
| 10 | 5,287 | 0.01% | 157 | 0 | 6.8% | 0.290 | 139.15 |
| 17 | 5,186 | 0.01% | 93 | 0 | 8.5% | 0.344 | 140.74 |
| 49 | 5,099 | 0.01% | 225 | 0 | 8.4% | 0.166 | 141.54 |
| 97 | 5,001 | 0.01% | 7 | 0 | 4.9% | 0.233 | 136.77 |
| 50 | 4,963 | 0.01% | 189 | 0 | 6.3% | 0.202 | 138.34 |
| 16 | 4,888 | 0.01% | 42 | 0 | 5.7% | 0.254 | 139.62 |
| 69 | 4,845 | 0.01% | 437 | 0 | 8.6% | 0.120 | 138.06 |
| 94 | 4,800 | 0.01% | 193 | 0 | 7.2% | 0.234 | 140.73 |
| 81 | 4,698 | 0.01% | 257 | 0 | 7.0% | 0.361 | 136.20 |
| 72 | 4,463 | 0.01% | 65 | 0 | 4.5% | 0.422 | 137.46 |
| 23 | 4,413 | 0.01% | 277 | 0 | 11.6% | 0.166 | 136.36 |
| 119 | 4,352 | 0.01% | 19 | 0 | 5.4% | 0.220 | 136.91 |
| 73 | 4,144 | 0.00% | 141 | 0 | 10.7% | 0.122 | 138.92 |
| 12 | 3,962 | 0.00% | 425 | 0 | 9.7% | 0.141 | 139.34 |
| 99 | 3,826 | 0.00% | 127 | 0 | 4.0% | 0.380 | 136.52 |
| 6 | 3,789 | 0.00% | 296 | 0 | 9.1% | 0.254 | 135.91 |
| 18 | 3,763 | 0.00% | 184 | 0 | 4.2% | 0.366 | 136.46 |
| 28 | 3,593 | 0.00% | 278 | 0 | 10.9% | 0.184 | 137.30 |
| 7 | 3,588 | 0.00% | 213 | 0 | 10.3% | 0.217 | 139.68 |
| 45 | 3,538 | 0.00% | 237 | 0 | 10.5% | 0.268 | 136.44 |
| 48 | 3,502 | 0.00% | 189 | 0 | 8.4% | 0.240 | 135.54 |
| 87 | 3,339 | 0.00% | 25 | 0 | 3.2% | 0.466 | 136.52 |
| 80 | 3,298 | 0.00% | 307 | 0 | 9.1% | 0.120 | 138.03 |
| 83 | 3,161 | 0.00% | 9 | 0 | 4.1% | 0.360 | 136.62 |
| 41 | 3,068 | 0.00% | 208 | 0 | 5.7% | 0.149 | 140.35 |
| 124 | 3,051 | 0.00% | 223 | 0 | 9.6% | 0.240 | 138.53 |
| 1 | 3,038 | 0.00% | 207 | 0 | 6.2% | 0.318 | 138.69 |
| 37 | 3,032 | 0.00% | 296 | 0 | 8.7% | 0.137 | 136.81 |
| 104 | 3,019 | 0.00% | 75 | 0 | 4.8% | 0.502 | 139.20 |
| 92 | 2,926 | 0.00% | 232 | 0 | 3.4% | 0.232 | 138.34 |
| 103 | 2,806 | 0.00% | 15 | 0 | 3.5% | 0.503 | 137.62 |
| 54 | 2,758 | 0.00% | 244 | 0 | 8.0% | 0.097 | 138.57 |
| 9 | 2,748 | 0.00% | 148 | 0 | 4.8% | 0.354 | 139.38 |
| 66 | 2,742 | 0.00% | 79 | 0 | 0.9% | 0.661 | 134.45 |
| 52 | 2,622 | 0.00% | 240 | 0 | 10.2% | 0.169 | 137.23 |
| 29 | 2,560 | 0.00% | 18 | 0 | 3.2% | 0.376 | 137.21 |
| 5 | 2,510 | 0.00% | 178 | 0 | 3.8% | 0.202 | 139.44 |
| 125 | 2,487 | 0.00% | 86 | 0 | 3.0% | 0.204 | 139.70 |
| 40 | 2,436 | 0.00% | 134 | 0 | 6.8% | 0.321 | 137.54 |
| 64 | 2,408 | 0.00% | 172 | 0 | 8.7% | 0.184 | 136.31 |
| 19 | 2,280 | 0.00% | 143 | 0 | 5.9% | 0.452 | 137.10 |
| 2 | 2,245 | 0.00% | 160 | 0 | 6.3% | 0.302 | 141.13 |
| 51 | 2,221 | 0.00% | 97 | 0 | 5.3% | 0.205 | 137.79 |
| 42 | 2,178 | 0.00% | 103 | 0 | 5.2% | 0.230 | 137.91 |
| 68 | 2,101 | 0.00% | 148 | 0 | 5.3% | 0.476 | 140.99 |
| 77 | 2,092 | 0.00% | 146 | 0 | 5.6% | 0.472 | 136.96 |
| 108 | 2,069 | 0.00% | 125 | 0 | 3.5% | 0.490 | 136.76 |
| 20 | 2,058 | 0.00% | 70 | 0 | 2.4% | 0.412 | 139.87 |
| 14 | 1,984 | 0.00% | 173 | 0 | 2.5% | 0.292 | 139.24 |
| 82 | 1,793 | 0.00% | 51 | 0 | 2.5% | 0.387 | 138.05 |
| 96 | 1,783 | 0.00% | 239 | 0 | 5.9% | 0.238 | 137.72 |
| 24 | 1,766 | 0.00% | 207 | 0 | 8.7% | 0.141 | 138.68 |
| 113 | 1,726 | 0.00% | 182 | 0 | 4.1% | 0.246 | 135.14 |
| 58 | 1,724 | 0.00% | 157 | 0 | 5.6% | 0.169 | 140.70 |
| 32 | 1,702 | 0.00% | 113 | 0 | 4.4% | 0.139 | 138.47 |
| 95 | 1,561 | 0.00% | 178 | 0 | 3.7% | 0.243 | 135.55 |
| 47 | 1,492 | 0.00% | 148 | 0 | 5.8% | 0.135 | 137.04 |
| 30 | 1,475 | 0.00% | 91 | 0 | 3.4% | 0.333 | 140.88 |
| 116 | 1,446 | 0.00% | 84 | 0 | 3.2% | 0.406 | 137.60 |
| 11 | 1,427 | 0.00% | 168 | 0 | 4.9% | 0.272 | 137.00 |
| 102 | 1,425 | 0.00% | 86 | 0 | 3.5% | 0.232 | 136.77 |
| 93 | 1,391 | 0.00% | 143 | 0 | 3.3% | 0.370 | 137.06 |
| 117 | 1,360 | 0.00% | 135 | 0 | 6.8% | 0.274 | 136.29 |
| 33 | 1,347 | 0.00% | 8 | 0 | 3.6% | 0.295 | 140.48 |
| 4 | 1,304 | 0.00% | 114 | 0 | 2.5% | 0.212 | 137.68 |
| 89 | 1,211 | 0.00% | 271 | 0 | 3.6% | 0.156 | 135.73 |
| 46 | 1,205 | 0.00% | 174 | 0 | 2.2% | 0.144 | 136.63 |
| 63 | 1,106 | 0.00% | 21 | 0 | 2.2% | 0.460 | 140.94 |
| 35 | 1,090 | 0.00% | 132 | 0 | 3.7% | 0.238 | 137.98 |
| 106 | 1,037 | 0.00% | 140 | 0 | 2.6% | 0.256 | 136.89 |
| 78 | 903 | 0.00% | 78 | 0 | 2.5% | 0.385 | 134.40 |
| 31 | 860 | 0.00% | 134 | 0 | 2.8% | 0.452 | 137.97 |
| 53 | 817 | 0.00% | 43 | 0 | 1.8% | 0.334 | 140.02 |
| 21 | 624 | 0.00% | 13 | 0 | 1.1% | 0.654 | 136.99 |
| 56 | 623 | 0.00% | 40 | 0 | 2.8% | 0.351 | 137.99 |
| 110 | 513 | 0.00% | 73 | 0 | 1.6% | 0.284 | 136.38 |
| 22 | 428 | 0.00% | 37 | 0 | 0.4% | 0.820 | 141.95 |
| 90 | 393 | 0.00% | 33 | 0 | 1.8% | 0.488 | 136.66 |
| 101 | 382 | 0.00% | 99 | 0 | 1.2% | 0.272 | 140.46 |
| 111 | 207 | 0.00% | 28 | 0 | 0.4% | 0.862 | 136.31 |
| 114 | 196 | 0.00% | 9 | 0 | 0.7% | 0.604 | 135.63 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127** under a quarter of the mean cluster size: **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v16_kcenter_seed1 --out runs/clustering/v16_kcenter_seed1/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *Large `radius` relative to `trace`* ⇒ an outlier-driven cluster; k-center minimizes the worst case, so a few far points can still leave `radius` high even with low average (`trace`) variance.
