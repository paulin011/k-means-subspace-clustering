# Clustering report (subspace_kmeans) — `runs/clustering/v2_subspace_big/`

*Generated 2026-07-31 17:58 by `analyze_clusters.py`. K=128 affine subspaces of dim 32 in 2048-dim token space, 86,016,000 tokens.*

## Overview

The model groups the 86,016,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 128 clusters, and fits a 32-dimensional flat (an *affine subspace*: a centroid plus a basis of directions) through each one. A token is assigned to whichever cluster leaves the smallest **orthogonal residual** — the part of the token that its cluster's subspace cannot reconstruct.

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
- **Uⱼ** (`model['U'][j]`, shape `[2048, 32]`): an orthonormal basis for cluster *j*'s subspace; its columns are PC directions in descending eigenvalue order.
- **Orthogonal residual** of a token *x* under cluster *j*: `‖x − μⱼ‖² − ‖Uⱼᵀ(x − μⱼ)‖²`. The first term is the squared distance to the centroid; the second is the part of that distance the subspace *captures*. What's left is the unexplained residual that the assignment minimises.
- **eigvals** (`model['eigvals'][j]`): the top-32 eigenvalues of cluster *j*'s within-cluster covariance — variance along each kept PC direction.
- **trace** (`model['trace'][j]`): mean squared distance of cluster *j*'s tokens to its centroid μⱼ — the cluster's total within-cluster variance.
- **counts** (`model['counts'][j]`): number of tokens in cluster *j*; **wⱼ = counts[j] / Σcounts** is its population share, used to weight every global average.

## Configuration

*How to read this: these are the run's input settings, taken from `model['config']` (plus `model['sampled_files']` for the true file count). `clusters` is K, `dim` is the subspace dimension d (`dim=0` ⇒ plain k-means). `iters` is the maximum number of training iterations (each one reassigns every token to its best cluster, then refits the centroids and subspaces); `tol` is the convergence threshold: once the fraction of tokens that change cluster in an iteration falls below it, the run stops early instead of using all `iters`. Together they bound how long the run takes. `seed` fixes both the token sample and the cluster initialisation so a run is reproducible.*

| parameter | value |
|---|---|
| src | latents_2 |
| num_files | 7000 |
| tokens_per_file | 12288 |
| clusters | 128 |
| dim | 32 |
| iters | 40 |
| tol | 0.001 |
| linear | False |
| seed | 0 |
| chunk_size | 262144 |
| gpus | 2 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `82ca602ed7e7`
- **Files:** 7000 latent files, 12288 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v2_subspace_big//sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v2_subspace_big//sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7659.20 | 100.00% | 127 | 23,702,502 |
| 2 | 3092.84 | 74.64% | 2,108 | 5,171,010 |
| 3 | 2834.43 | 40.86% | 31,953 | 2,312,301 |
| 4 | 2765.63 | 22.33% | 125,563 | 1,977,172 |
| 5 | 2737.22 | 14.81% | 156,615 | 1,676,012 |
| 6 | 2722.19 | 10.95% | 171,476 | 1,581,330 |
| 7 | 2712.85 | 8.58% | 182,258 | 1,531,793 |
| 8 | 2706.53 | 6.97% | 188,586 | 1,491,608 |
| 9 | 2701.97 | 5.87% | 193,709 | 1,457,367 |
| 10 | 2698.40 | 5.10% | 198,848 | 1,444,461 |
| 11 | 2695.57 | 4.51% | 205,626 | 1,421,788 |
| 12 | 2693.21 | 4.09% | 210,482 | 1,395,454 |
| 13 | 2691.12 | 3.74% | 214,633 | 1,380,268 |
| 14 | 2689.34 | 3.40% | 219,147 | 1,375,497 |
| 15 | 2687.80 | 3.11% | 220,110 | 1,376,128 |
| 16 | 2686.40 | 2.87% | 219,939 | 1,378,148 |
| 17 | 2685.20 | 2.65% | 219,637 | 1,379,993 |
| 18 | 2684.18 | 2.46% | 219,318 | 1,381,436 |
| 19 | 2683.30 | 2.29% | 219,277 | 1,382,534 |
| 20 | 2682.55 | 2.13% | 219,580 | 1,383,572 |
| 21 | 2681.86 | 2.01% | 220,328 | 1,384,443 |
| 22 | 2681.22 | 1.89% | 221,170 | 1,385,269 |
| 23 | 2680.70 | 1.78% | 221,624 | 1,385,951 |
| 24 | 2680.22 | 1.67% | 221,293 | 1,386,489 |
| 25 | 2679.79 | 1.58% | 221,704 | 1,386,832 |
| 26 | 2679.41 | 1.49% | 222,946 | 1,387,125 |
| 27 | 2679.08 | 1.40% | 223,158 | 1,387,333 |
| 28 | 2678.79 | 1.34% | 223,219 | 1,387,562 |
| 29 | 2678.52 | 1.29% | 223,006 | 1,387,824 |
| 30 | 2678.27 | 1.24% | 222,595 | 1,388,035 |
| 31 | 2678.04 | 1.20% | 222,159 | 1,388,176 |
| 32 | 2677.80 | 1.16% | 221,891 | 1,388,363 |
| 33 | 2677.58 | 1.13% | 221,652 | 1,388,315 |
| 34 | 2677.37 | 1.08% | 221,451 | 1,388,330 |
| 35 | 2677.18 | 1.03% | 220,864 | 1,388,221 |
| 36 | 2677.02 | 1.00% | 220,097 | 1,388,220 |
| 37 | 2676.87 | 0.97% | 219,465 | 1,388,135 |
| 38 | 2676.71 | 0.95% | 218,948 | 1,388,079 |
| 39 | 2676.55 | 0.94% | 218,530 | 1,388,031 |
| 40 | 2676.39 | 0.92% | 218,161 | 1,388,056 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5998**, split into:

- **10.2%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **45.2%** is `captured / total`. It is the chunk of `within` that the top-32 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5998, the same denominator as the other two, which is what lets all three add to 100%.
- **44.6%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-32): 0.512** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 32 subspace directions recover about 51%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 16 / median 20 / max 23** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=33 = a truncation warning. Your max is 23, below the cap → no cluster truncated, d=32 has headroom.

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`EVR(top-32)` = `Σ eigvals[j] / trace[j]` — fraction of this cluster's own variance captured by its subspace (1.0 = the subspace explains the cluster perfectly; near the global average ⇒ d truncates the spectrum).*
- *`d80` = smallest number of leading PC directions whose eigenvalues reach 80% of `Σ eigvals[j]` (capped at d+1=33 when even all 32 fall short). Low d80 ⇒ a few directions dominate; d80 ≈ d ⇒ a flat spectrum the subspace truncates.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`maxAff` = this cluster's subspace affinity to its **nearest** neighbour, `maxⱼ≠ᵢ ‖UᵢᵀUⱼ‖²_F / d` ∈ [0,1]. A per-cluster separation score: **high ⇒ some other cluster spans nearly the same directions**, so this row is a merge candidate. The affinity table below lists only the top pairs, so a near-duplicate cluster is invisible there unless its pair happens to rank; this column always shows it.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.25%–1.61% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-32) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 27 | 1,388,056 | 1.61% | 0.577 | 19 | 100 | 203 | 49.0% | 0.004 | 0.679 |
| 66 | 1,299,328 | 1.51% | 0.605 | 18 | 93 | 187 | 49.7% | 0.001 | 0.582 |
| 14 | 1,111,688 | 1.29% | 0.509 | 19 | 285 | 153 | 34.7% | 0.049 | 0.709 |
| 80 | 1,110,127 | 1.29% | 0.553 | 19 | 128 | 254 | 27.0% | 0.083 | 0.694 |
| 78 | 1,102,026 | 1.28% | 0.606 | 18 | 249 | 264 | 43.4% | 0.023 | 0.688 |
| 17 | 1,090,818 | 1.27% | 0.595 | 18 | 125 | 180 | 38.3% | 0.063 | 0.714 |
| 83 | 1,083,990 | 1.26% | 0.575 | 18 | 309 | 187 | 44.1% | 0.028 | 0.681 |
| 4 | 1,043,294 | 1.21% | 0.504 | 21 | 287 | 204 | 28.9% | 0.053 | 0.732 |
| 39 | 1,041,814 | 1.21% | 0.527 | 20 | 309 | 280 | 43.3% | 0.016 | 0.729 |
| 121 | 1,032,347 | 1.20% | 0.574 | 17 | 255 | 263 | 32.0% | 0.084 | 0.648 |
| 90 | 1,026,440 | 1.19% | 0.526 | 19 | 154 | 230 | 27.7% | 0.069 | 0.643 |
| 68 | 991,418 | 1.15% | 0.531 | 19 | 355 | 146 | 31.1% | 0.058 | 0.702 |
| 34 | 981,566 | 1.14% | 0.485 | 20 | 219 | 50 | 17.1% | 0.147 | 0.671 |
| 42 | 971,006 | 1.13% | 0.501 | 21 | 190 | 231 | 34.8% | 0.056 | 0.713 |
| 107 | 962,677 | 1.12% | 0.532 | 19 | 391 | 65 | 41.7% | 0.041 | 0.709 |
| 59 | 944,731 | 1.10% | 0.507 | 20 | 331 | 188 | 39.7% | 0.031 | 0.732 |
| 50 | 944,035 | 1.10% | 0.617 | 18 | 153 | 144 | 39.1% | 0.057 | 0.714 |
| 117 | 928,686 | 1.08% | 0.489 | 22 | 244 | 164 | 36.3% | 0.038 | 0.722 |
| 127 | 907,309 | 1.05% | 0.583 | 17 | 346 | 101 | 37.8% | 0.050 | 0.677 |
| 21 | 902,062 | 1.05% | 0.606 | 18 | 65 | 132 | 49.1% | 0.003 | 0.549 |
| 99 | 887,807 | 1.03% | 0.485 | 22 | 252 | 173 | 27.9% | 0.062 | 0.716 |
| 10 | 879,599 | 1.02% | 0.562 | 18 | 484 | 53 | 31.1% | 0.054 | 0.680 |
| 110 | 873,753 | 1.02% | 0.484 | 21 | 234 | 118 | 21.8% | 0.109 | 0.711 |
| 18 | 872,137 | 1.01% | 0.515 | 19 | 359 | 60 | 41.0% | 0.033 | 0.676 |
| 35 | 868,176 | 1.01% | 0.554 | 20 | 63 | 133 | 47.8% | 0.027 | 0.560 |
| 114 | 860,049 | 1.00% | 0.477 | 22 | 135 | 158 | 29.5% | 0.070 | 0.730 |
| 56 | 847,441 | 0.99% | 0.586 | 19 | 64 | 134 | 43.2% | 0.019 | 0.694 |
| 93 | 841,489 | 0.98% | 0.600 | 18 | 195 | 109 | 37.7% | 0.061 | 0.686 |
| 74 | 839,895 | 0.98% | 0.573 | 19 | 73 | 163 | 35.4% | 0.045 | 0.681 |
| 37 | 833,525 | 0.97% | 0.478 | 22 | 201 | 190 | 22.0% | 0.156 | 0.737 |
| 76 | 829,716 | 0.96% | 0.528 | 19 | 424 | 60 | 35.6% | 0.048 | 0.683 |
| 16 | 806,579 | 0.94% | 0.502 | 20 | 60 | 123 | 43.4% | 0.022 | 0.592 |
| 88 | 805,607 | 0.94% | 0.494 | 20 | 195 | 97 | 32.1% | 0.067 | 0.713 |
| 70 | 804,297 | 0.94% | 0.537 | 20 | 203 | 205 | 37.5% | 0.031 | 0.729 |
| 6 | 796,528 | 0.93% | 0.502 | 20 | 299 | 107 | 37.0% | 0.088 | 0.722 |
| 85 | 789,649 | 0.92% | 0.474 | 23 | 120 | 130 | 25.4% | 0.153 | 0.730 |
| 11 | 785,668 | 0.91% | 0.500 | 20 | 240 | 80 | 34.4% | 0.049 | 0.680 |
| 41 | 782,884 | 0.91% | 0.530 | 19 | 446 | 27 | 40.9% | 0.026 | 0.673 |
| 48 | 760,793 | 0.88% | 0.444 | 22 | 259 | 103 | 29.7% | 0.156 | 0.718 |
| 12 | 760,574 | 0.88% | 0.482 | 22 | 120 | 97 | 36.9% | 0.083 | 0.714 |
| 46 | 757,325 | 0.88% | 0.481 | 21 | 169 | 170 | 39.9% | 0.090 | 0.732 |
| 81 | 756,962 | 0.88% | 0.496 | 20 | 72 | 140 | 33.4% | 0.068 | 0.683 |
| 82 | 755,012 | 0.88% | 0.490 | 21 | 251 | 167 | 19.2% | 0.144 | 0.688 |
| 69 | 730,429 | 0.85% | 0.508 | 20 | 242 | 80 | 22.5% | 0.087 | 0.719 |
| 125 | 728,129 | 0.85% | 0.554 | 17 | 183 | 77 | 35.3% | 0.064 | 0.648 |
| 86 | 719,441 | 0.84% | 0.488 | 21 | 289 | 53 | 20.5% | 0.153 | 0.724 |
| 43 | 717,022 | 0.83% | 0.544 | 18 | 234 | 128 | 20.8% | 0.129 | 0.654 |
| 120 | 716,067 | 0.83% | 0.486 | 22 | 104 | 156 | 22.8% | 0.151 | 0.732 |
| 58 | 715,501 | 0.83% | 0.476 | 21 | 206 | 49 | 23.3% | 0.112 | 0.737 |
| 73 | 711,071 | 0.83% | 0.445 | 22 | 232 | 117 | 25.1% | 0.112 | 0.706 |
| 111 | 696,795 | 0.81% | 0.576 | 17 | 332 | 40 | 42.2% | 0.027 | 0.660 |
| 1 | 692,418 | 0.80% | 0.461 | 22 | 208 | 90 | 30.7% | 0.063 | 0.717 |
| 84 | 690,311 | 0.80% | 0.462 | 22 | 177 | 115 | 32.1% | 0.059 | 0.731 |
| 115 | 688,308 | 0.80% | 0.492 | 20 | 76 | 142 | 33.7% | 0.072 | 0.678 |
| 5 | 680,966 | 0.79% | 0.485 | 21 | 327 | 67 | 35.9% | 0.068 | 0.671 |
| 100 | 679,971 | 0.79% | 0.472 | 22 | 215 | 27 | 16.9% | 0.155 | 0.681 |
| 67 | 679,272 | 0.79% | 0.470 | 22 | 207 | 111 | 18.3% | 0.239 | 0.710 |
| 61 | 678,747 | 0.79% | 0.476 | 21 | 90 | 132 | 25.0% | 0.110 | 0.689 |
| 33 | 678,347 | 0.79% | 0.504 | 21 | 360 | 92 | 34.9% | 0.062 | 0.681 |
| 52 | 677,785 | 0.79% | 0.493 | 20 | 306 | 31 | 26.0% | 0.115 | 0.704 |
| 3 | 673,187 | 0.78% | 0.489 | 22 | 66 | 115 | 36.6% | 0.055 | 0.692 |
| 113 | 666,035 | 0.77% | 0.539 | 18 | 441 | 5 | 34.1% | 0.046 | 0.652 |
| 20 | 659,897 | 0.77% | 0.524 | 19 | 483 | 0 | 25.9% | 0.094 | 0.676 |
| 126 | 659,523 | 0.77% | 0.466 | 23 | 106 | 138 | 26.5% | 0.282 | 0.731 |
| 95 | 652,752 | 0.76% | 0.509 | 20 | 287 | 111 | 19.1% | 0.119 | 0.674 |
| 105 | 651,981 | 0.76% | 0.440 | 22 | 92 | 108 | 32.1% | 0.053 | 0.627 |
| 71 | 647,629 | 0.75% | 0.468 | 22 | 123 | 170 | 22.5% | 0.129 | 0.690 |
| 87 | 642,742 | 0.75% | 0.501 | 20 | 51 | 106 | 38.3% | 0.040 | 0.665 |
| 60 | 638,824 | 0.74% | 0.475 | 21 | 211 | 49 | 19.6% | 0.117 | 0.716 |
| 19 | 635,930 | 0.74% | 0.501 | 20 | 84 | 122 | 29.0% | 0.105 | 0.695 |
| 57 | 633,520 | 0.74% | 0.487 | 21 | 131 | 115 | 23.2% | 0.079 | 0.691 |
| 22 | 632,132 | 0.73% | 0.449 | 21 | 364 | 8 | 34.9% | 0.077 | 0.677 |
| 45 | 624,665 | 0.73% | 0.474 | 22 | 170 | 149 | 30.0% | 0.075 | 0.710 |
| 64 | 619,786 | 0.72% | 0.439 | 21 | 99 | 69 | 23.2% | 0.139 | 0.687 |
| 15 | 615,113 | 0.72% | 0.442 | 22 | 53 | 110 | 40.9% | 0.032 | 0.687 |
| 77 | 615,039 | 0.72% | 0.601 | 19 | 44 | 87 | 49.4% | 0.002 | 0.603 |
| 7 | 614,320 | 0.71% | 0.462 | 21 | 204 | 56 | 23.5% | 0.094 | 0.718 |
| 94 | 613,652 | 0.71% | 0.570 | 17 | 567 | 0 | 34.5% | 0.057 | 0.660 |
| 65 | 610,688 | 0.71% | 0.464 | 21 | 186 | 72 | 21.1% | 0.133 | 0.688 |
| 119 | 608,410 | 0.71% | 0.523 | 19 | 449 | 0 | 39.4% | 0.031 | 0.637 |
| 106 | 604,076 | 0.70% | 0.460 | 21 | 333 | 30 | 27.5% | 0.050 | 0.693 |
| 75 | 603,557 | 0.70% | 0.464 | 22 | 115 | 131 | 21.0% | 0.159 | 0.732 |
| 47 | 600,052 | 0.70% | 0.488 | 20 | 95 | 48 | 21.4% | 0.144 | 0.623 |
| 54 | 597,859 | 0.70% | 0.451 | 22 | 273 | 44 | 32.5% | 0.057 | 0.706 |
| 118 | 589,973 | 0.69% | 0.449 | 21 | 106 | 78 | 22.4% | 0.116 | 0.677 |
| 124 | 583,618 | 0.68% | 0.513 | 20 | 57 | 119 | 41.2% | 0.024 | 0.698 |
| 98 | 564,943 | 0.66% | 0.584 | 16 | 437 | 15 | 29.5% | 0.059 | 0.638 |
| 63 | 564,614 | 0.66% | 0.514 | 19 | 394 | 0 | 35.1% | 0.054 | 0.626 |
| 29 | 562,722 | 0.65% | 0.486 | 20 | 119 | 114 | 17.1% | 0.119 | 0.659 |
| 123 | 559,051 | 0.65% | 0.621 | 18 | 40 | 75 | 45.9% | 0.015 | 0.659 |
| 26 | 558,361 | 0.65% | 0.453 | 23 | 142 | 50 | 22.7% | 0.170 | 0.692 |
| 30 | 551,291 | 0.64% | 0.488 | 22 | 134 | 41 | 18.2% | 0.154 | 0.701 |
| 0 | 532,678 | 0.62% | 0.601 | 20 | 39 | 77 | 49.0% | 0.004 | 0.603 |
| 44 | 528,271 | 0.61% | 0.485 | 22 | 130 | 74 | 21.7% | 0.128 | 0.732 |
| 104 | 526,236 | 0.61% | 0.462 | 21 | 117 | 81 | 23.0% | 0.135 | 0.688 |
| 36 | 515,845 | 0.60% | 0.467 | 21 | 277 | 9 | 24.8% | 0.104 | 0.637 |
| 25 | 506,927 | 0.59% | 0.569 | 19 | 37 | 81 | 44.8% | 0.018 | 0.570 |
| 24 | 497,248 | 0.58% | 0.468 | 20 | 101 | 134 | 16.4% | 0.139 | 0.637 |
| 97 | 494,692 | 0.58% | 0.473 | 21 | 57 | 104 | 33.6% | 0.060 | 0.666 |
| 101 | 489,699 | 0.57% | 0.476 | 22 | 176 | 47 | 17.5% | 0.131 | 0.692 |
| 8 | 484,942 | 0.56% | 0.466 | 22 | 46 | 97 | 38.0% | 0.039 | 0.643 |
| 40 | 481,720 | 0.56% | 0.451 | 22 | 187 | 54 | 29.7% | 0.086 | 0.691 |
| 112 | 480,860 | 0.56% | 0.454 | 22 | 60 | 54 | 28.2% | 0.088 | 0.643 |
| 102 | 478,682 | 0.56% | 0.484 | 20 | 85 | 79 | 21.7% | 0.085 | 0.633 |
| 108 | 478,610 | 0.56% | 0.488 | 21 | 87 | 81 | 30.9% | 0.038 | 0.710 |
| 51 | 461,566 | 0.54% | 0.494 | 20 | 43 | 83 | 43.8% | 0.033 | 0.683 |
| 91 | 443,879 | 0.52% | 0.486 | 20 | 62 | 43 | 25.6% | 0.112 | 0.616 |
| 96 | 440,324 | 0.51% | 0.496 | 21 | 37 | 74 | 38.7% | 0.060 | 0.673 |
| 9 | 437,074 | 0.51% | 0.463 | 22 | 100 | 64 | 18.5% | 0.170 | 0.690 |
| 89 | 429,828 | 0.50% | 0.530 | 20 | 32 | 64 | 47.5% | 0.013 | 0.587 |
| 53 | 420,555 | 0.49% | 0.472 | 21 | 65 | 51 | 25.6% | 0.085 | 0.650 |
| 103 | 418,911 | 0.49% | 0.448 | 22 | 72 | 42 | 21.9% | 0.125 | 0.682 |
| 13 | 412,667 | 0.48% | 0.482 | 20 | 433 | 0 | 38.4% | 0.045 | 0.638 |
| 122 | 401,318 | 0.47% | 0.522 | 18 | 245 | 5 | 17.9% | 0.106 | 0.651 |
| 32 | 401,150 | 0.47% | 0.518 | 20 | 30 | 61 | 44.0% | 0.019 | 0.560 |
| 31 | 398,437 | 0.46% | 0.490 | 20 | 212 | 11 | 26.0% | 0.094 | 0.642 |
| 72 | 355,191 | 0.41% | 0.470 | 21 | 43 | 66 | 25.1% | 0.100 | 0.617 |
| 79 | 347,077 | 0.40% | 0.476 | 21 | 26 | 47 | 37.6% | 0.042 | 0.556 |
| 49 | 334,611 | 0.39% | 0.559 | 19 | 28 | 62 | 37.7% | 0.047 | 0.462 |
| 38 | 327,292 | 0.38% | 0.464 | 22 | 39 | 46 | 38.6% | 0.041 | 0.639 |
| 116 | 319,645 | 0.37% | 0.488 | 21 | 57 | 19 | 25.1% | 0.091 | 0.623 |
| 109 | 309,352 | 0.36% | 0.503 | 21 | 58 | 73 | 17.7% | 0.135 | 0.595 |
| 23 | 308,268 | 0.36% | 0.464 | 22 | 23 | 47 | 42.2% | 0.031 | 0.542 |
| 55 | 302,636 | 0.35% | 0.544 | 19 | 23 | 48 | 45.6% | 0.013 | 0.566 |
| 2 | 285,162 | 0.33% | 0.505 | 21 | 21 | 40 | 48.1% | 0.007 | 0.452 |
| 62 | 277,286 | 0.32% | 0.531 | 20 | 21 | 45 | 43.7% | 0.013 | 0.606 |
| 92 | 273,795 | 0.32% | 0.474 | 22 | 21 | 44 | 44.1% | 0.037 | 0.532 |
| 28 | 218,161 | 0.25% | 0.572 | 19 | 21 | 16 | 22.5% | 0.096 | 0.515 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **13, 20, 63, 94, 119**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v2_subspace_big/ --out runs/clustering/v2_subspace_big//temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 32 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.354, mean 0.371, max 0.737 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 37 ↔ 58 | 0.737 | 0.588 |
| 46 ↔ 75 | 0.732 | 0.613 |
| 4 ↔ 59 | 0.732 | 0.244 |
| 44 ↔ 120 | 0.732 | 0.588 |
| 84 ↔ 126 | 0.731 | 0.525 |
| 85 ↔ 114 | 0.730 | 0.796 |
| 39 ↔ 70 | 0.729 | 0.593 |
| 58 ↔ 86 | 0.724 | 0.630 |
| 6 ↔ 117 | 0.722 | 0.555 |
| 69 ↔ 117 | 0.719 | 0.505 |
| 7 ↔ 48 | 0.718 | 0.595 |
| 1 ↔ 46 | 0.717 | 0.354 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
