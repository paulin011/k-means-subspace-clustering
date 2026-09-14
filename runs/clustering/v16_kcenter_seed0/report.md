# Clustering report (kcenter) — `runs/clustering/v16_kcenter_seed0`

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
  python3 src/clustering/kcenter.py --files-from runs/clustering/v16_kcenter_seed0/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v16_kcenter_seed0/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 12498.89 | 100.00% | 1,190 | 76,218,426 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **7286**, split into:

- **31.5%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **68.5%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*
- *`radius` = k-center's native objective: the max distance from the centroid to any member (a worst-case, not an average like `trace`).*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.00%–88.61% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | cells@50% | owned | files@50% | tCV | radius |
|---|---|---|---|---|---|---|---|
| 0 | 76,218,424 | 88.61% | 5655 | 12287 | 49.2% | 0.007 | 146.81 |
| 24 | 999,247 | 1.16% | 1010 | 1 | 33.8% | 0.048 | 141.86 |
| 64 | 975,084 | 1.13% | 968 | 0 | 37.9% | 0.042 | 141.41 |
| 106 | 619,209 | 0.72% | 2008 | 0 | 37.7% | 0.067 | 142.58 |
| 100 | 410,147 | 0.48% | 1311 | 0 | 31.3% | 0.054 | 143.73 |
| 103 | 378,025 | 0.44% | 1434 | 0 | 32.4% | 0.065 | 141.03 |
| 127 | 305,748 | 0.36% | 233 | 0 | 17.0% | 0.197 | 139.89 |
| 122 | 292,632 | 0.34% | 1129 | 0 | 26.8% | 0.173 | 141.49 |
| 91 | 285,702 | 0.33% | 737 | 0 | 31.7% | 0.128 | 141.65 |
| 115 | 264,093 | 0.31% | 711 | 0 | 26.9% | 0.121 | 141.35 |
| 93 | 244,456 | 0.28% | 1605 | 0 | 30.1% | 0.171 | 141.92 |
| 113 | 242,830 | 0.28% | 1489 | 0 | 30.7% | 0.066 | 141.95 |
| 79 | 222,057 | 0.26% | 1020 | 0 | 28.6% | 0.103 | 142.67 |
| 77 | 166,862 | 0.19% | 1225 | 0 | 31.8% | 0.063 | 141.33 |
| 75 | 161,595 | 0.19% | 527 | 0 | 29.9% | 0.093 | 142.75 |
| 42 | 158,457 | 0.18% | 1244 | 0 | 27.6% | 0.220 | 142.26 |
| 111 | 146,713 | 0.17% | 1198 | 0 | 28.8% | 0.171 | 141.90 |
| 72 | 140,410 | 0.16% | 608 | 0 | 26.9% | 0.094 | 139.93 |
| 124 | 130,239 | 0.15% | 1208 | 0 | 31.9% | 0.066 | 141.48 |
| 92 | 130,000 | 0.15% | 351 | 0 | 21.5% | 0.097 | 141.65 |
| 66 | 127,936 | 0.15% | 693 | 0 | 18.3% | 0.084 | 143.81 |
| 45 | 123,365 | 0.14% | 821 | 0 | 31.2% | 0.126 | 142.25 |
| 121 | 119,992 | 0.14% | 455 | 0 | 22.4% | 0.145 | 140.60 |
| 96 | 119,701 | 0.14% | 617 | 0 | 30.7% | 0.050 | 140.68 |
| 126 | 118,692 | 0.14% | 977 | 0 | 27.3% | 0.200 | 143.45 |
| 84 | 117,977 | 0.14% | 340 | 0 | 25.9% | 0.120 | 142.98 |
| 125 | 116,915 | 0.14% | 658 | 0 | 26.7% | 0.059 | 143.70 |
| 95 | 114,132 | 0.13% | 656 | 0 | 32.6% | 0.111 | 140.08 |
| 54 | 108,907 | 0.13% | 747 | 0 | 32.6% | 0.074 | 141.66 |
| 61 | 101,461 | 0.12% | 604 | 0 | 34.2% | 0.112 | 142.45 |
| 68 | 95,451 | 0.11% | 450 | 0 | 22.4% | 0.094 | 141.63 |
| 101 | 84,827 | 0.10% | 340 | 0 | 23.2% | 0.194 | 140.81 |
| 88 | 75,702 | 0.09% | 1443 | 0 | 26.7% | 0.148 | 140.98 |
| 48 | 75,683 | 0.09% | 816 | 0 | 27.2% | 0.078 | 144.29 |
| 108 | 74,447 | 0.09% | 712 | 0 | 27.6% | 0.069 | 142.16 |
| 74 | 72,848 | 0.08% | 613 | 0 | 24.5% | 0.032 | 139.97 |
| 80 | 71,175 | 0.08% | 586 | 0 | 21.4% | 0.117 | 140.19 |
| 120 | 69,416 | 0.08% | 856 | 0 | 25.8% | 0.059 | 144.09 |
| 13 | 68,261 | 0.08% | 498 | 0 | 21.2% | 0.272 | 141.86 |
| 44 | 67,759 | 0.08% | 313 | 0 | 22.2% | 0.087 | 142.91 |
| 67 | 67,180 | 0.08% | 565 | 0 | 19.3% | 0.281 | 142.10 |
| 86 | 61,364 | 0.07% | 321 | 0 | 9.4% | 0.323 | 142.50 |
| 99 | 47,825 | 0.06% | 534 | 0 | 25.0% | 0.104 | 140.32 |
| 81 | 47,647 | 0.06% | 855 | 0 | 15.9% | 0.207 | 141.91 |
| 104 | 47,626 | 0.06% | 550 | 0 | 25.2% | 0.273 | 141.72 |
| 65 | 44,684 | 0.05% | 122 | 0 | 13.0% | 0.144 | 140.35 |
| 83 | 44,127 | 0.05% | 355 | 0 | 18.5% | 0.088 | 140.69 |
| 87 | 43,160 | 0.05% | 857 | 0 | 18.6% | 0.073 | 141.17 |
| 114 | 40,353 | 0.05% | 666 | 0 | 26.2% | 0.163 | 139.99 |
| 55 | 37,015 | 0.04% | 262 | 0 | 13.5% | 0.124 | 140.06 |
| 62 | 36,851 | 0.04% | 416 | 0 | 24.6% | 0.071 | 142.21 |
| 90 | 35,629 | 0.04% | 489 | 0 | 20.8% | 0.245 | 142.31 |
| 39 | 35,439 | 0.04% | 529 | 0 | 22.0% | 0.086 | 142.37 |
| 107 | 31,420 | 0.04% | 549 | 0 | 20.8% | 0.163 | 141.68 |
| 25 | 31,419 | 0.04% | 122 | 0 | 11.1% | 0.167 | 145.94 |
| 117 | 31,322 | 0.04% | 399 | 0 | 25.2% | 0.102 | 140.67 |
| 57 | 30,035 | 0.03% | 154 | 0 | 20.6% | 0.114 | 143.21 |
| 102 | 29,962 | 0.03% | 441 | 0 | 21.8% | 0.073 | 139.62 |
| 105 | 29,714 | 0.03% | 288 | 0 | 18.9% | 0.154 | 140.97 |
| 119 | 28,939 | 0.03% | 598 | 0 | 25.9% | 0.088 | 139.50 |
| 23 | 28,285 | 0.03% | 642 | 0 | 22.4% | 0.052 | 143.02 |
| 69 | 27,719 | 0.03% | 570 | 0 | 17.0% | 0.134 | 140.35 |
| 41 | 26,593 | 0.03% | 541 | 0 | 15.9% | 0.379 | 141.29 |
| 97 | 26,121 | 0.03% | 33 | 0 | 11.1% | 0.118 | 139.55 |
| 73 | 25,463 | 0.03% | 589 | 0 | 16.7% | 0.133 | 139.49 |
| 15 | 25,121 | 0.03% | 330 | 0 | 12.8% | 0.160 | 140.60 |
| 26 | 24,052 | 0.03% | 381 | 0 | 20.6% | 0.058 | 141.28 |
| 43 | 24,042 | 0.03% | 293 | 0 | 17.9% | 0.167 | 141.77 |
| 40 | 23,759 | 0.03% | 464 | 0 | 23.9% | 0.067 | 139.85 |
| 30 | 23,374 | 0.03% | 171 | 0 | 16.5% | 0.236 | 141.53 |
| 50 | 20,374 | 0.02% | 535 | 0 | 20.7% | 0.085 | 142.71 |
| 56 | 19,769 | 0.02% | 449 | 0 | 15.6% | 0.102 | 141.64 |
| 52 | 19,747 | 0.02% | 219 | 0 | 18.5% | 0.208 | 142.20 |
| 3 | 18,688 | 0.02% | 710 | 0 | 17.1% | 0.109 | 139.88 |
| 49 | 18,393 | 0.02% | 348 | 0 | 18.6% | 0.131 | 142.17 |
| 17 | 17,841 | 0.02% | 458 | 0 | 18.0% | 0.123 | 143.66 |
| 59 | 17,816 | 0.02% | 303 | 0 | 19.6% | 0.152 | 139.92 |
| 28 | 17,615 | 0.02% | 447 | 0 | 13.8% | 0.098 | 145.09 |
| 18 | 16,778 | 0.02% | 478 | 0 | 19.9% | 0.188 | 145.71 |
| 19 | 16,456 | 0.02% | 335 | 0 | 14.0% | 0.115 | 139.91 |
| 116 | 15,402 | 0.02% | 394 | 0 | 18.0% | 0.148 | 142.66 |
| 31 | 15,229 | 0.02% | 131 | 0 | 7.7% | 0.362 | 144.93 |
| 76 | 15,124 | 0.02% | 293 | 0 | 14.0% | 0.165 | 139.21 |
| 118 | 12,724 | 0.01% | 190 | 0 | 19.6% | 0.100 | 141.32 |
| 63 | 12,399 | 0.01% | 211 | 0 | 17.6% | 0.144 | 142.46 |
| 109 | 11,494 | 0.01% | 55 | 0 | 8.2% | 0.287 | 139.84 |
| 38 | 11,452 | 0.01% | 61 | 0 | 10.8% | 0.164 | 138.86 |
| 10 | 11,117 | 0.01% | 317 | 0 | 20.4% | 0.150 | 142.83 |
| 8 | 11,110 | 0.01% | 469 | 0 | 18.2% | 0.148 | 143.59 |
| 98 | 10,699 | 0.01% | 438 | 0 | 17.7% | 0.086 | 142.71 |
| 58 | 10,491 | 0.01% | 525 | 0 | 13.1% | 0.134 | 138.73 |
| 60 | 10,465 | 0.01% | 474 | 0 | 12.4% | 0.150 | 140.26 |
| 16 | 10,259 | 0.01% | 548 | 0 | 17.4% | 0.081 | 140.19 |
| 22 | 9,695 | 0.01% | 441 | 0 | 12.8% | 0.146 | 140.21 |
| 82 | 9,502 | 0.01% | 22 | 0 | 10.7% | 0.147 | 140.45 |
| 71 | 9,088 | 0.01% | 98 | 0 | 9.6% | 0.219 | 145.11 |
| 123 | 9,048 | 0.01% | 331 | 0 | 19.3% | 0.129 | 140.65 |
| 46 | 8,958 | 0.01% | 387 | 0 | 10.2% | 0.151 | 142.08 |
| 5 | 8,907 | 0.01% | 328 | 0 | 7.8% | 0.137 | 140.32 |
| 110 | 8,883 | 0.01% | 116 | 0 | 7.9% | 0.417 | 142.08 |
| 51 | 8,676 | 0.01% | 630 | 0 | 13.2% | 0.085 | 140.91 |
| 85 | 8,379 | 0.01% | 232 | 0 | 9.7% | 0.321 | 142.45 |
| 33 | 7,834 | 0.01% | 219 | 0 | 9.2% | 0.218 | 142.85 |
| 9 | 7,497 | 0.01% | 106 | 0 | 5.9% | 0.170 | 141.11 |
| 35 | 7,482 | 0.01% | 324 | 0 | 10.4% | 0.123 | 140.33 |
| 112 | 6,990 | 0.01% | 217 | 0 | 11.0% | 0.192 | 139.27 |
| 21 | 6,708 | 0.01% | 189 | 0 | 13.4% | 0.098 | 140.21 |
| 11 | 6,615 | 0.01% | 202 | 0 | 7.0% | 0.321 | 140.51 |
| 94 | 6,429 | 0.01% | 220 | 0 | 11.7% | 0.093 | 142.41 |
| 34 | 5,992 | 0.01% | 393 | 0 | 11.8% | 0.182 | 140.57 |
| 47 | 5,968 | 0.01% | 295 | 0 | 15.7% | 0.123 | 140.83 |
| 89 | 5,932 | 0.01% | 234 | 0 | 10.6% | 0.424 | 141.77 |
| 78 | 5,928 | 0.01% | 482 | 0 | 8.4% | 0.146 | 139.73 |
| 70 | 5,583 | 0.01% | 380 | 0 | 10.4% | 0.166 | 141.13 |
| 14 | 5,010 | 0.01% | 445 | 0 | 8.5% | 0.104 | 141.50 |
| 4 | 4,281 | 0.00% | 281 | 0 | 5.8% | 0.236 | 139.88 |
| 27 | 3,975 | 0.00% | 271 | 0 | 8.0% | 0.380 | 142.62 |
| 53 | 3,823 | 0.00% | 88 | 0 | 4.7% | 0.263 | 137.32 |
| 37 | 3,601 | 0.00% | 223 | 0 | 10.2% | 0.213 | 141.54 |
| 6 | 3,552 | 0.00% | 208 | 0 | 4.1% | 0.203 | 142.05 |
| 2 | 3,113 | 0.00% | 159 | 0 | 4.6% | 0.192 | 141.84 |
| 1 | 2,903 | 0.00% | 234 | 0 | 6.5% | 0.239 | 139.71 |
| 20 | 2,307 | 0.00% | 64 | 0 | 3.6% | 0.276 | 143.04 |
| 29 | 2,255 | 0.00% | 240 | 0 | 6.2% | 0.157 | 140.82 |
| 12 | 2,016 | 0.00% | 133 | 0 | 3.1% | 0.388 | 140.53 |
| 36 | 1,850 | 0.00% | 247 | 0 | 7.8% | 0.146 | 139.01 |
| 32 | 1,769 | 0.00% | 192 | 0 | 5.1% | 0.211 | 138.30 |
| 7 | 1,190 | 0.00% | 203 | 0 | 5.9% | 0.272 | 142.22 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127** under a quarter of the mean cluster size: **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 107, 108, 109, 110, 111, 112, 114, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v16_kcenter_seed0 --out runs/clustering/v16_kcenter_seed0/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *Large `radius` relative to `trace`* ⇒ an outlier-driven cluster; k-center minimizes the worst case, so a few far points can still leave `radius` high even with low average (`trace`) variance.
