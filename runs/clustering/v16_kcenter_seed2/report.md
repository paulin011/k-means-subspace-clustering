# Clustering report (kcenter) — `runs/clustering/v16_kcenter_seed2`

*Generated 2026-09-14 16:48 by `analyze_clusters.py`. K=128 point clusters in 2048-dim token space, 86,016,000 tokens.*

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
  python3 src/clustering/kcenter.py --files-from runs/clustering/v16_kcenter_seed2/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v16_kcenter_seed2/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 13723.11 | 100.00% | 1,244 | 57,382,242 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **14191**, split into:

- **41.6%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **58.4%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*
- *`radius` = k-center's native objective: the max distance from the centroid to any member (a worst-case, not an average like `trace`).*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.00%–66.71% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | cells@50% | owned | files@50% | tCV | radius |
|---|---|---|---|---|---|---|---|
| 0 | 57,382,240 | 66.71% | 4901 | 12274 | 48.8% | 0.009 | 145.29 |
| 111 | 1,908,863 | 2.22% | 2054 | 0 | 41.2% | 0.055 | 142.12 |
| 112 | 1,908,179 | 2.22% | 1681 | 1 | 41.9% | 0.099 | 142.57 |
| 101 | 1,725,119 | 2.01% | 2204 | 0 | 42.9% | 0.074 | 140.44 |
| 109 | 1,700,205 | 1.98% | 2296 | 0 | 41.5% | 0.086 | 143.63 |
| 126 | 1,596,121 | 1.86% | 1677 | 7 | 36.6% | 0.044 | 143.35 |
| 79 | 1,455,671 | 1.69% | 2590 | 0 | 42.2% | 0.052 | 140.70 |
| 121 | 1,111,611 | 1.29% | 2160 | 0 | 38.3% | 0.106 | 141.13 |
| 106 | 1,000,804 | 1.16% | 2203 | 0 | 39.2% | 0.037 | 144.91 |
| 19 | 981,255 | 1.14% | 1675 | 0 | 40.3% | 0.037 | 141.63 |
| 110 | 863,654 | 1.00% | 2358 | 1 | 41.1% | 0.048 | 141.94 |
| 77 | 813,513 | 0.95% | 2217 | 0 | 40.9% | 0.026 | 142.70 |
| 23 | 810,508 | 0.94% | 2099 | 0 | 39.5% | 0.091 | 143.07 |
| 72 | 742,935 | 0.86% | 1960 | 0 | 39.5% | 0.065 | 140.70 |
| 95 | 711,390 | 0.83% | 1307 | 1 | 38.0% | 0.119 | 141.88 |
| 123 | 618,836 | 0.72% | 2029 | 0 | 37.7% | 0.036 | 141.99 |
| 127 | 600,825 | 0.70% | 1145 | 3 | 33.9% | 0.087 | 142.72 |
| 102 | 590,159 | 0.69% | 1557 | 0 | 38.5% | 0.050 | 140.75 |
| 114 | 578,258 | 0.67% | 1700 | 0 | 39.1% | 0.078 | 142.31 |
| 107 | 564,623 | 0.66% | 883 | 0 | 33.9% | 0.041 | 138.67 |
| 45 | 548,521 | 0.64% | 821 | 0 | 29.1% | 0.094 | 145.17 |
| 125 | 492,987 | 0.57% | 1557 | 0 | 36.5% | 0.054 | 141.11 |
| 90 | 450,692 | 0.52% | 1689 | 0 | 38.3% | 0.055 | 140.12 |
| 86 | 422,127 | 0.49% | 1059 | 0 | 34.6% | 0.027 | 142.59 |
| 63 | 404,894 | 0.47% | 607 | 0 | 35.2% | 0.109 | 143.25 |
| 54 | 402,503 | 0.47% | 1115 | 0 | 37.5% | 0.033 | 143.33 |
| 49 | 367,527 | 0.43% | 1211 | 0 | 36.7% | 0.055 | 140.74 |
| 66 | 344,096 | 0.40% | 1732 | 0 | 37.6% | 0.055 | 139.87 |
| 62 | 305,758 | 0.36% | 965 | 0 | 35.2% | 0.077 | 140.38 |
| 118 | 272,127 | 0.32% | 1466 | 0 | 38.4% | 0.067 | 142.02 |
| 78 | 244,490 | 0.28% | 1271 | 0 | 27.6% | 0.074 | 140.48 |
| 93 | 241,397 | 0.28% | 688 | 0 | 33.6% | 0.093 | 141.65 |
| 64 | 190,910 | 0.22% | 953 | 0 | 31.1% | 0.142 | 140.60 |
| 25 | 180,140 | 0.21% | 1206 | 0 | 33.3% | 0.120 | 141.48 |
| 94 | 174,998 | 0.20% | 814 | 0 | 32.7% | 0.143 | 142.52 |
| 53 | 163,321 | 0.19% | 345 | 0 | 18.9% | 0.106 | 142.06 |
| 58 | 126,341 | 0.15% | 1114 | 0 | 30.1% | 0.106 | 143.74 |
| 51 | 120,491 | 0.14% | 1384 | 0 | 31.8% | 0.051 | 145.79 |
| 61 | 113,011 | 0.13% | 376 | 0 | 23.8% | 0.084 | 143.46 |
| 60 | 112,567 | 0.13% | 567 | 0 | 26.4% | 0.147 | 141.24 |
| 98 | 100,152 | 0.12% | 956 | 0 | 28.8% | 0.103 | 141.00 |
| 84 | 98,932 | 0.12% | 1182 | 0 | 28.8% | 0.080 | 141.99 |
| 108 | 94,590 | 0.11% | 1002 | 0 | 20.8% | 0.085 | 140.41 |
| 33 | 91,465 | 0.11% | 777 | 0 | 29.2% | 0.092 | 146.14 |
| 20 | 83,886 | 0.10% | 605 | 0 | 28.7% | 0.114 | 142.76 |
| 40 | 82,691 | 0.10% | 789 | 0 | 29.0% | 0.059 | 142.37 |
| 80 | 78,493 | 0.09% | 512 | 0 | 25.9% | 0.083 | 140.97 |
| 39 | 77,557 | 0.09% | 586 | 0 | 32.1% | 0.102 | 143.41 |
| 83 | 77,229 | 0.09% | 687 | 0 | 26.0% | 0.059 | 141.42 |
| 100 | 76,792 | 0.09% | 740 | 0 | 29.3% | 0.121 | 140.86 |
| 116 | 73,527 | 0.09% | 437 | 1 | 26.9% | 0.091 | 147.52 |
| 69 | 72,845 | 0.08% | 442 | 0 | 27.1% | 0.068 | 142.61 |
| 27 | 65,822 | 0.08% | 545 | 0 | 22.5% | 0.126 | 140.66 |
| 57 | 63,015 | 0.07% | 718 | 0 | 25.8% | 0.134 | 144.66 |
| 119 | 59,063 | 0.07% | 687 | 0 | 23.7% | 0.133 | 140.60 |
| 65 | 58,327 | 0.07% | 544 | 0 | 27.3% | 0.047 | 140.33 |
| 120 | 57,171 | 0.07% | 792 | 0 | 26.7% | 0.063 | 139.18 |
| 75 | 56,397 | 0.07% | 847 | 0 | 23.2% | 0.069 | 141.31 |
| 26 | 52,450 | 0.06% | 974 | 0 | 24.3% | 0.091 | 141.24 |
| 92 | 52,354 | 0.06% | 367 | 0 | 24.4% | 0.123 | 142.49 |
| 17 | 51,283 | 0.06% | 450 | 0 | 20.7% | 0.123 | 142.35 |
| 43 | 51,122 | 0.06% | 903 | 0 | 24.3% | 0.072 | 141.36 |
| 44 | 50,861 | 0.06% | 593 | 0 | 22.9% | 0.128 | 143.15 |
| 124 | 49,819 | 0.06% | 90 | 0 | 10.3% | 0.194 | 140.44 |
| 85 | 49,344 | 0.06% | 186 | 0 | 23.8% | 0.123 | 143.12 |
| 74 | 48,446 | 0.06% | 778 | 0 | 26.1% | 0.099 | 140.71 |
| 47 | 42,895 | 0.05% | 516 | 0 | 22.0% | 0.100 | 140.39 |
| 104 | 41,341 | 0.05% | 350 | 0 | 14.8% | 0.199 | 142.57 |
| 9 | 40,011 | 0.05% | 166 | 0 | 10.2% | 0.292 | 141.65 |
| 117 | 38,266 | 0.04% | 600 | 0 | 25.2% | 0.038 | 140.83 |
| 70 | 38,020 | 0.04% | 256 | 0 | 15.2% | 0.347 | 140.27 |
| 7 | 36,222 | 0.04% | 764 | 0 | 15.6% | 0.092 | 140.81 |
| 73 | 34,972 | 0.04% | 30 | 0 | 14.9% | 0.104 | 141.16 |
| 48 | 30,395 | 0.04% | 603 | 0 | 18.9% | 0.098 | 142.31 |
| 14 | 28,335 | 0.03% | 390 | 0 | 22.5% | 0.051 | 140.86 |
| 24 | 25,508 | 0.03% | 413 | 0 | 18.6% | 0.177 | 141.05 |
| 29 | 24,814 | 0.03% | 110 | 0 | 8.1% | 0.210 | 142.36 |
| 34 | 24,156 | 0.03% | 455 | 0 | 16.1% | 0.145 | 142.53 |
| 12 | 23,683 | 0.03% | 450 | 0 | 20.2% | 0.093 | 145.80 |
| 3 | 23,418 | 0.03% | 393 | 0 | 26.1% | 0.093 | 142.44 |
| 68 | 21,385 | 0.02% | 603 | 0 | 21.4% | 0.093 | 142.24 |
| 32 | 20,832 | 0.02% | 430 | 0 | 21.4% | 0.093 | 141.03 |
| 38 | 20,697 | 0.02% | 232 | 0 | 22.0% | 0.097 | 142.44 |
| 35 | 19,427 | 0.02% | 124 | 0 | 21.0% | 0.098 | 141.95 |
| 55 | 18,134 | 0.02% | 762 | 0 | 21.6% | 0.126 | 143.48 |
| 41 | 17,454 | 0.02% | 426 | 0 | 23.3% | 0.048 | 140.80 |
| 46 | 15,561 | 0.02% | 259 | 0 | 13.9% | 0.141 | 142.13 |
| 91 | 15,258 | 0.02% | 496 | 0 | 13.5% | 0.131 | 141.04 |
| 96 | 15,232 | 0.02% | 339 | 0 | 16.8% | 0.279 | 141.17 |
| 122 | 14,737 | 0.02% | 699 | 0 | 18.2% | 0.136 | 140.79 |
| 82 | 14,428 | 0.02% | 345 | 0 | 15.3% | 0.137 | 142.95 |
| 15 | 13,331 | 0.02% | 310 | 0 | 16.2% | 0.248 | 144.21 |
| 56 | 13,212 | 0.02% | 410 | 0 | 15.7% | 0.147 | 143.14 |
| 103 | 12,963 | 0.02% | 532 | 0 | 18.6% | 0.109 | 141.99 |
| 28 | 11,853 | 0.01% | 216 | 0 | 14.1% | 0.166 | 140.24 |
| 52 | 11,708 | 0.01% | 324 | 0 | 18.1% | 0.058 | 142.26 |
| 16 | 11,520 | 0.01% | 554 | 0 | 15.4% | 0.104 | 142.77 |
| 50 | 10,458 | 0.01% | 325 | 0 | 14.9% | 0.207 | 141.18 |
| 30 | 9,865 | 0.01% | 495 | 0 | 14.0% | 0.120 | 140.89 |
| 21 | 9,491 | 0.01% | 234 | 0 | 17.2% | 0.133 | 142.42 |
| 115 | 9,278 | 0.01% | 363 | 0 | 11.1% | 0.138 | 141.96 |
| 76 | 9,172 | 0.01% | 72 | 0 | 13.4% | 0.189 | 141.75 |
| 1 | 8,440 | 0.01% | 283 | 0 | 10.8% | 0.250 | 141.06 |
| 71 | 8,398 | 0.01% | 288 | 0 | 15.4% | 0.103 | 141.76 |
| 113 | 8,131 | 0.01% | 147 | 0 | 13.0% | 0.193 | 139.13 |
| 37 | 7,387 | 0.01% | 384 | 0 | 14.5% | 0.076 | 143.42 |
| 89 | 7,218 | 0.01% | 235 | 0 | 12.6% | 0.108 | 145.68 |
| 97 | 6,672 | 0.01% | 536 | 0 | 13.3% | 0.118 | 141.78 |
| 10 | 6,516 | 0.01% | 343 | 0 | 8.4% | 0.132 | 142.63 |
| 8 | 6,456 | 0.01% | 440 | 0 | 12.6% | 0.116 | 143.06 |
| 81 | 6,093 | 0.01% | 380 | 0 | 9.2% | 0.118 | 143.29 |
| 18 | 5,812 | 0.01% | 157 | 0 | 8.4% | 0.344 | 142.92 |
| 87 | 5,727 | 0.01% | 282 | 0 | 15.9% | 0.073 | 143.38 |
| 42 | 5,256 | 0.01% | 472 | 0 | 13.5% | 0.089 | 141.51 |
| 22 | 4,729 | 0.01% | 180 | 0 | 7.9% | 0.312 | 140.58 |
| 105 | 4,401 | 0.01% | 269 | 0 | 11.6% | 0.142 | 139.31 |
| 13 | 4,356 | 0.01% | 304 | 0 | 9.5% | 0.155 | 141.06 |
| 4 | 4,308 | 0.01% | 355 | 0 | 6.8% | 0.182 | 144.49 |
| 67 | 4,104 | 0.00% | 320 | 0 | 10.5% | 0.145 | 141.66 |
| 88 | 4,037 | 0.00% | 310 | 0 | 10.8% | 0.114 | 141.23 |
| 36 | 3,888 | 0.00% | 251 | 0 | 4.9% | 0.330 | 141.95 |
| 2 | 3,549 | 0.00% | 325 | 0 | 6.8% | 0.202 | 140.33 |
| 59 | 3,236 | 0.00% | 208 | 0 | 5.4% | 0.362 | 142.61 |
| 11 | 3,000 | 0.00% | 178 | 0 | 4.5% | 0.313 | 145.27 |
| 99 | 2,835 | 0.00% | 122 | 0 | 6.4% | 0.209 | 141.07 |
| 31 | 2,754 | 0.00% | 168 | 0 | 3.7% | 0.235 | 142.19 |
| 5 | 2,119 | 0.00% | 194 | 0 | 5.0% | 0.302 | 142.28 |
| 6 | 1,244 | 0.00% | 202 | 0 | 6.1% | 0.215 | 143.61 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 113, 114, 115, 117, 118, 119, 120, 121, 122, 123, 124, 125** under a quarter of the mean cluster size: **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 65, 67, 68, 69, 70, 71, 73, 74, 75, 76, 80, 81, 82, 83, 84, 85, 87, 88, 89, 91, 92, 96, 97, 98, 99, 100, 103, 104, 105, 108, 113, 115, 116, 117, 119, 120, 122, 124**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v16_kcenter_seed2 --out runs/clustering/v16_kcenter_seed2/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *Large `radius` relative to `trace`* ⇒ an outlier-driven cluster; k-center minimizes the worst case, so a few far points can still leave `radius` high even with low average (`trace`) variance.
