# Clustering report (subspace_kmeans) — `runs/clustering/v3_subspace_big_i100/`

*Generated 2026-07-31 17:59 by `analyze_clusters.py`. K=128 affine subspaces of dim 32 in 2048-dim token space, 86,016,000 tokens.*

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
| num_files | 7000 *(--files-from reused the sample; --num-files=1500 ignored)* |
| tokens_per_file | 12288 |
| clusters | 128 |
| dim | 32 |
| iters | 100 |
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
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v3_subspace_big_i100//sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v3_subspace_big_i100//sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

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
| 41 | 2676.25 | 0.90% | 217,922 | 1,388,071 |
| 42 | 2676.12 | 0.87% | 217,696 | 1,388,160 |
| 43 | 2675.97 | 0.85% | 217,533 | 1,388,256 |
| 44 | 2675.87 | 0.83% | 217,378 | 1,388,420 |
| 45 | 2675.74 | 0.81% | 217,286 | 1,388,574 |
| 46 | 2675.65 | 0.79% | 217,207 | 1,388,703 |
| 47 | 2675.52 | 0.78% | 217,148 | 1,388,759 |
| 48 | 2675.41 | 0.76% | 217,101 | 1,388,789 |
| 49 | 2675.30 | 0.75% | 217,187 | 1,388,767 |
| 50 | 2675.20 | 0.74% | 217,440 | 1,388,703 |
| 51 | 2675.10 | 0.72% | 218,074 | 1,388,669 |
| 52 | 2675.00 | 0.71% | 218,285 | 1,388,596 |
| 53 | 2674.91 | 0.69% | 218,046 | 1,388,480 |
| 54 | 2674.81 | 0.68% | 217,828 | 1,388,403 |
| 55 | 2674.73 | 0.66% | 217,690 | 1,388,314 |
| 56 | 2674.63 | 0.65% | 217,603 | 1,388,264 |
| 57 | 2674.55 | 0.64% | 217,568 | 1,388,224 |
| 58 | 2674.48 | 0.62% | 217,676 | 1,388,175 |
| 59 | 2674.41 | 0.62% | 217,966 | 1,388,181 |
| 60 | 2674.34 | 0.61% | 218,547 | 1,388,192 |
| 61 | 2674.25 | 0.60% | 219,487 | 1,388,220 |
| 62 | 2674.18 | 0.59% | 220,655 | 1,388,265 |
| 63 | 2674.11 | 0.58% | 221,405 | 1,388,292 |
| 64 | 2674.05 | 0.57% | 221,682 | 1,388,342 |
| 65 | 2673.98 | 0.56% | 221,842 | 1,388,419 |
| 66 | 2673.91 | 0.56% | 221,883 | 1,388,467 |
| 67 | 2673.86 | 0.55% | 221,871 | 1,388,515 |
| 68 | 2673.79 | 0.55% | 221,830 | 1,388,608 |
| 69 | 2673.74 | 0.54% | 221,820 | 1,388,696 |
| 70 | 2673.68 | 0.54% | 221,831 | 1,388,768 |
| 71 | 2673.61 | 0.53% | 221,903 | 1,388,802 |
| 72 | 2673.57 | 0.53% | 221,991 | 1,388,850 |
| 73 | 2673.52 | 0.53% | 222,065 | 1,388,889 |
| 74 | 2673.45 | 0.53% | 222,165 | 1,388,951 |
| 75 | 2673.41 | 0.52% | 222,249 | 1,389,015 |
| 76 | 2673.34 | 0.52% | 222,394 | 1,389,009 |
| 77 | 2673.29 | 0.52% | 222,592 | 1,389,012 |
| 78 | 2673.22 | 0.51% | 222,823 | 1,389,023 |
| 79 | 2673.17 | 0.51% | 223,071 | 1,389,019 |
| 80 | 2673.11 | 0.50% | 223,405 | 1,389,035 |
| 81 | 2673.05 | 0.49% | 223,743 | 1,389,072 |
| 82 | 2673.01 | 0.49% | 224,090 | 1,389,079 |
| 83 | 2672.95 | 0.48% | 224,490 | 1,389,100 |
| 84 | 2672.92 | 0.48% | 224,818 | 1,389,097 |
| 85 | 2672.85 | 0.48% | 225,083 | 1,389,071 |
| 86 | 2672.80 | 0.48% | 225,522 | 1,389,056 |
| 87 | 2672.74 | 0.48% | 226,222 | 1,389,021 |
| 88 | 2672.68 | 0.49% | 227,225 | 1,389,000 |
| 89 | 2672.60 | 0.49% | 228,213 | 1,388,970 |
| 90 | 2672.54 | 0.49% | 228,632 | 1,388,942 |
| 91 | 2672.47 | 0.49% | 228,674 | 1,388,902 |
| 92 | 2672.40 | 0.49% | 228,686 | 1,388,927 |
| 93 | 2672.32 | 0.48% | 228,749 | 1,388,933 |
| 94 | 2672.24 | 0.47% | 228,846 | 1,388,934 |
| 95 | 2672.19 | 0.45% | 229,016 | 1,388,917 |
| 96 | 2672.12 | 0.44% | 229,230 | 1,388,946 |
| 97 | 2672.09 | 0.43% | 229,527 | 1,388,945 |
| 98 | 2672.05 | 0.41% | 229,840 | 1,388,956 |
| 99 | 2672.02 | 0.39% | 230,208 | 1,388,976 |
| 100 | 2671.99 | 0.37% | 230,634 | 1,388,995 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5998**, split into:

- **10.2%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **45.3%** is `captured / total`. It is the chunk of `within` that the top-32 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5998, the same denominator as the other two, which is what lets all three add to 100%.
- **44.5%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-32): 0.513** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 32 subspace directions recover about 51%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 17 / median 20 / max 22** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=33 = a truncation warning. Your max is 22, below the cap → no cluster truncated, d=32 has headroom.

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

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.27%–1.61% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-32) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 27 | 1,388,995 | 1.61% | 0.577 | 19 | 100 | 203 | 49.0% | 0.004 | 0.681 |
| 66 | 1,296,134 | 1.51% | 0.605 | 18 | 93 | 186 | 49.7% | 0.001 | 0.581 |
| 4 | 1,116,025 | 1.30% | 0.506 | 21 | 282 | 253 | 30.8% | 0.048 | 0.723 |
| 80 | 1,079,174 | 1.25% | 0.557 | 19 | 117 | 237 | 28.3% | 0.075 | 0.694 |
| 14 | 1,073,228 | 1.25% | 0.504 | 20 | 276 | 139 | 33.5% | 0.050 | 0.681 |
| 39 | 1,072,871 | 1.25% | 0.532 | 19 | 311 | 316 | 41.5% | 0.025 | 0.718 |
| 17 | 1,056,263 | 1.23% | 0.595 | 18 | 124 | 181 | 39.0% | 0.050 | 0.733 |
| 83 | 1,022,055 | 1.19% | 0.573 | 18 | 285 | 161 | 44.0% | 0.028 | 0.678 |
| 117 | 1,000,571 | 1.16% | 0.486 | 22 | 180 | 206 | 35.2% | 0.042 | 0.719 |
| 78 | 990,355 | 1.15% | 0.622 | 17 | 175 | 270 | 44.3% | 0.026 | 0.710 |
| 90 | 977,719 | 1.14% | 0.529 | 19 | 133 | 216 | 27.3% | 0.075 | 0.642 |
| 107 | 963,528 | 1.12% | 0.531 | 19 | 364 | 89 | 42.0% | 0.034 | 0.719 |
| 121 | 959,524 | 1.12% | 0.579 | 17 | 248 | 260 | 28.7% | 0.096 | 0.652 |
| 59 | 948,690 | 1.10% | 0.505 | 20 | 338 | 122 | 39.8% | 0.030 | 0.717 |
| 70 | 940,039 | 1.09% | 0.533 | 20 | 275 | 237 | 37.8% | 0.043 | 0.723 |
| 127 | 915,816 | 1.06% | 0.587 | 17 | 349 | 92 | 35.4% | 0.061 | 0.673 |
| 42 | 908,425 | 1.06% | 0.503 | 20 | 216 | 194 | 28.2% | 0.082 | 0.691 |
| 110 | 904,386 | 1.05% | 0.477 | 22 | 234 | 90 | 22.8% | 0.114 | 0.720 |
| 68 | 902,605 | 1.05% | 0.536 | 19 | 328 | 124 | 30.1% | 0.064 | 0.713 |
| 34 | 899,258 | 1.05% | 0.492 | 20 | 200 | 56 | 16.8% | 0.151 | 0.666 |
| 18 | 888,152 | 1.03% | 0.516 | 19 | 301 | 81 | 41.2% | 0.032 | 0.670 |
| 50 | 884,120 | 1.03% | 0.611 | 18 | 129 | 113 | 39.8% | 0.055 | 0.733 |
| 99 | 882,012 | 1.03% | 0.491 | 21 | 256 | 177 | 22.9% | 0.075 | 0.719 |
| 76 | 876,223 | 1.02% | 0.527 | 19 | 429 | 42 | 37.3% | 0.043 | 0.674 |
| 21 | 875,116 | 1.02% | 0.610 | 18 | 63 | 127 | 49.2% | 0.004 | 0.545 |
| 74 | 873,550 | 1.02% | 0.573 | 19 | 77 | 173 | 34.0% | 0.049 | 0.681 |
| 10 | 867,783 | 1.01% | 0.568 | 18 | 460 | 40 | 31.3% | 0.049 | 0.667 |
| 93 | 858,669 | 1.00% | 0.626 | 18 | 75 | 170 | 42.5% | 0.051 | 0.710 |
| 37 | 853,277 | 0.99% | 0.481 | 22 | 189 | 178 | 22.6% | 0.145 | 0.729 |
| 35 | 844,416 | 0.98% | 0.556 | 20 | 61 | 131 | 47.5% | 0.027 | 0.547 |
| 114 | 830,198 | 0.97% | 0.479 | 22 | 139 | 162 | 26.0% | 0.057 | 0.720 |
| 6 | 825,791 | 0.96% | 0.502 | 20 | 317 | 115 | 33.0% | 0.088 | 0.709 |
| 86 | 817,470 | 0.95% | 0.499 | 21 | 271 | 40 | 23.4% | 0.113 | 0.695 |
| 11 | 815,892 | 0.95% | 0.503 | 20 | 245 | 84 | 37.8% | 0.036 | 0.690 |
| 20 | 815,540 | 0.95% | 0.531 | 19 | 442 | 8 | 24.2% | 0.102 | 0.681 |
| 85 | 809,508 | 0.94% | 0.471 | 22 | 120 | 147 | 24.8% | 0.139 | 0.699 |
| 46 | 786,430 | 0.91% | 0.481 | 22 | 174 | 189 | 37.6% | 0.070 | 0.721 |
| 52 | 776,059 | 0.90% | 0.503 | 20 | 297 | 85 | 27.0% | 0.097 | 0.695 |
| 69 | 775,111 | 0.90% | 0.506 | 20 | 256 | 103 | 22.2% | 0.102 | 0.700 |
| 67 | 774,017 | 0.90% | 0.472 | 22 | 219 | 129 | 18.3% | 0.226 | 0.695 |
| 41 | 772,777 | 0.90% | 0.530 | 19 | 501 | 5 | 41.7% | 0.027 | 0.661 |
| 81 | 769,418 | 0.89% | 0.493 | 20 | 77 | 140 | 31.1% | 0.077 | 0.683 |
| 82 | 767,241 | 0.89% | 0.487 | 21 | 254 | 126 | 19.9% | 0.136 | 0.692 |
| 43 | 761,791 | 0.89% | 0.548 | 18 | 258 | 147 | 21.5% | 0.127 | 0.652 |
| 94 | 751,578 | 0.87% | 0.580 | 17 | 314 | 10 | 39.1% | 0.025 | 0.664 |
| 56 | 747,067 | 0.87% | 0.593 | 18 | 56 | 118 | 45.5% | 0.013 | 0.694 |
| 120 | 746,766 | 0.87% | 0.485 | 22 | 112 | 163 | 22.3% | 0.146 | 0.689 |
| 16 | 737,366 | 0.86% | 0.504 | 20 | 55 | 112 | 44.0% | 0.020 | 0.612 |
| 33 | 726,982 | 0.85% | 0.501 | 21 | 370 | 77 | 34.1% | 0.062 | 0.678 |
| 1 | 726,754 | 0.84% | 0.464 | 22 | 215 | 64 | 31.5% | 0.061 | 0.721 |
| 111 | 725,034 | 0.84% | 0.583 | 17 | 329 | 30 | 41.9% | 0.032 | 0.646 |
| 45 | 723,436 | 0.84% | 0.472 | 22 | 178 | 166 | 32.1% | 0.071 | 0.706 |
| 48 | 719,823 | 0.84% | 0.448 | 22 | 241 | 79 | 30.9% | 0.168 | 0.702 |
| 12 | 714,050 | 0.83% | 0.485 | 22 | 93 | 99 | 35.9% | 0.068 | 0.720 |
| 71 | 712,345 | 0.83% | 0.465 | 22 | 129 | 156 | 32.8% | 0.098 | 0.678 |
| 126 | 711,652 | 0.83% | 0.467 | 22 | 109 | 135 | 29.9% | 0.245 | 0.730 |
| 100 | 709,168 | 0.82% | 0.471 | 22 | 195 | 46 | 17.7% | 0.170 | 0.700 |
| 84 | 708,865 | 0.82% | 0.460 | 22 | 191 | 123 | 32.3% | 0.046 | 0.730 |
| 113 | 708,616 | 0.82% | 0.538 | 18 | 377 | 27 | 39.0% | 0.028 | 0.640 |
| 5 | 705,774 | 0.82% | 0.491 | 21 | 356 | 21 | 33.9% | 0.068 | 0.677 |
| 115 | 700,604 | 0.81% | 0.493 | 20 | 75 | 143 | 35.1% | 0.066 | 0.659 |
| 73 | 693,868 | 0.81% | 0.447 | 22 | 253 | 118 | 24.3% | 0.107 | 0.703 |
| 58 | 687,390 | 0.80% | 0.466 | 22 | 182 | 83 | 28.2% | 0.085 | 0.729 |
| 60 | 678,692 | 0.79% | 0.463 | 21 | 182 | 106 | 25.3% | 0.094 | 0.678 |
| 61 | 662,411 | 0.77% | 0.474 | 21 | 93 | 128 | 22.9% | 0.118 | 0.676 |
| 3 | 650,241 | 0.76% | 0.488 | 22 | 64 | 102 | 36.5% | 0.053 | 0.689 |
| 95 | 644,800 | 0.75% | 0.504 | 20 | 286 | 72 | 20.9% | 0.113 | 0.667 |
| 125 | 636,192 | 0.74% | 0.559 | 18 | 115 | 88 | 30.2% | 0.083 | 0.648 |
| 57 | 630,025 | 0.73% | 0.488 | 21 | 137 | 103 | 23.3% | 0.079 | 0.693 |
| 119 | 629,023 | 0.73% | 0.526 | 18 | 467 | 3 | 39.9% | 0.029 | 0.636 |
| 65 | 616,747 | 0.72% | 0.465 | 21 | 180 | 75 | 20.3% | 0.130 | 0.692 |
| 29 | 612,471 | 0.71% | 0.497 | 20 | 128 | 126 | 17.6% | 0.128 | 0.654 |
| 105 | 612,333 | 0.71% | 0.438 | 22 | 86 | 103 | 31.5% | 0.052 | 0.631 |
| 22 | 611,642 | 0.71% | 0.454 | 21 | 368 | 4 | 35.4% | 0.070 | 0.669 |
| 77 | 610,598 | 0.71% | 0.602 | 19 | 44 | 87 | 49.4% | 0.002 | 0.601 |
| 64 | 609,173 | 0.71% | 0.440 | 21 | 98 | 69 | 22.5% | 0.142 | 0.683 |
| 63 | 606,909 | 0.71% | 0.518 | 18 | 398 | 0 | 34.6% | 0.057 | 0.613 |
| 9 | 606,117 | 0.70% | 0.461 | 22 | 111 | 111 | 20.5% | 0.139 | 0.695 |
| 124 | 605,052 | 0.70% | 0.514 | 20 | 62 | 115 | 40.7% | 0.030 | 0.687 |
| 54 | 599,138 | 0.70% | 0.452 | 21 | 251 | 26 | 33.0% | 0.107 | 0.703 |
| 19 | 595,591 | 0.69% | 0.501 | 20 | 71 | 118 | 28.8% | 0.099 | 0.709 |
| 15 | 584,131 | 0.68% | 0.445 | 22 | 49 | 102 | 41.1% | 0.032 | 0.683 |
| 106 | 582,081 | 0.68% | 0.463 | 21 | 298 | 32 | 28.1% | 0.073 | 0.684 |
| 30 | 578,422 | 0.67% | 0.488 | 22 | 136 | 24 | 18.4% | 0.192 | 0.699 |
| 98 | 575,581 | 0.67% | 0.579 | 17 | 438 | 6 | 32.2% | 0.058 | 0.626 |
| 47 | 573,918 | 0.67% | 0.495 | 20 | 81 | 66 | 23.9% | 0.127 | 0.596 |
| 101 | 557,835 | 0.65% | 0.469 | 22 | 190 | 84 | 20.2% | 0.132 | 0.695 |
| 44 | 557,043 | 0.65% | 0.485 | 22 | 103 | 134 | 22.9% | 0.106 | 0.694 |
| 7 | 553,266 | 0.64% | 0.471 | 21 | 220 | 21 | 20.7% | 0.104 | 0.679 |
| 88 | 545,556 | 0.63% | 0.515 | 20 | 44 | 94 | 38.9% | 0.040 | 0.685 |
| 87 | 533,496 | 0.62% | 0.504 | 20 | 40 | 86 | 46.7% | 0.009 | 0.685 |
| 104 | 533,271 | 0.62% | 0.468 | 21 | 114 | 81 | 22.3% | 0.141 | 0.695 |
| 0 | 529,945 | 0.62% | 0.601 | 20 | 38 | 76 | 49.0% | 0.004 | 0.601 |
| 25 | 527,495 | 0.61% | 0.569 | 19 | 38 | 84 | 45.0% | 0.018 | 0.571 |
| 40 | 518,152 | 0.60% | 0.449 | 22 | 219 | 40 | 30.1% | 0.063 | 0.660 |
| 36 | 515,319 | 0.60% | 0.471 | 20 | 288 | 16 | 25.7% | 0.101 | 0.620 |
| 26 | 514,639 | 0.60% | 0.456 | 22 | 99 | 60 | 22.2% | 0.155 | 0.695 |
| 97 | 511,843 | 0.60% | 0.473 | 21 | 61 | 112 | 32.0% | 0.068 | 0.683 |
| 8 | 502,243 | 0.58% | 0.467 | 22 | 49 | 102 | 37.3% | 0.043 | 0.650 |
| 102 | 501,329 | 0.58% | 0.488 | 20 | 74 | 88 | 24.1% | 0.067 | 0.635 |
| 53 | 500,122 | 0.58% | 0.470 | 21 | 93 | 36 | 39.4% | 0.041 | 0.642 |
| 31 | 497,772 | 0.58% | 0.488 | 20 | 226 | 3 | 18.7% | 0.117 | 0.669 |
| 108 | 488,894 | 0.57% | 0.490 | 21 | 83 | 92 | 29.1% | 0.042 | 0.706 |
| 24 | 482,063 | 0.56% | 0.468 | 20 | 99 | 82 | 16.4% | 0.141 | 0.645 |
| 112 | 475,354 | 0.55% | 0.454 | 22 | 58 | 54 | 28.1% | 0.086 | 0.641 |
| 89 | 455,536 | 0.53% | 0.526 | 20 | 33 | 67 | 46.9% | 0.013 | 0.586 |
| 96 | 452,402 | 0.53% | 0.496 | 21 | 37 | 72 | 39.2% | 0.065 | 0.643 |
| 91 | 438,305 | 0.51% | 0.484 | 20 | 64 | 44 | 25.0% | 0.115 | 0.611 |
| 13 | 413,635 | 0.48% | 0.477 | 20 | 266 | 1 | 36.9% | 0.040 | 0.645 |
| 118 | 412,878 | 0.48% | 0.439 | 22 | 36 | 76 | 37.3% | 0.086 | 0.629 |
| 122 | 411,800 | 0.48% | 0.516 | 18 | 184 | 7 | 17.5% | 0.113 | 0.633 |
| 51 | 408,517 | 0.47% | 0.492 | 20 | 37 | 70 | 39.9% | 0.036 | 0.683 |
| 123 | 390,566 | 0.45% | 0.657 | 19 | 28 | 57 | 48.7% | 0.004 | 0.583 |
| 32 | 388,371 | 0.45% | 0.521 | 20 | 29 | 60 | 45.9% | 0.016 | 0.555 |
| 75 | 376,551 | 0.44% | 0.462 | 21 | 54 | 81 | 23.0% | 0.188 | 0.694 |
| 103 | 369,167 | 0.43% | 0.454 | 22 | 52 | 86 | 25.0% | 0.113 | 0.651 |
| 72 | 356,141 | 0.41% | 0.470 | 21 | 44 | 65 | 24.5% | 0.103 | 0.627 |
| 79 | 335,264 | 0.39% | 0.477 | 21 | 25 | 47 | 38.3% | 0.041 | 0.548 |
| 49 | 322,607 | 0.38% | 0.571 | 19 | 26 | 58 | 39.0% | 0.044 | 0.459 |
| 38 | 317,441 | 0.37% | 0.462 | 22 | 37 | 51 | 35.9% | 0.051 | 0.655 |
| 2 | 312,315 | 0.36% | 0.504 | 21 | 23 | 45 | 49.2% | 0.003 | 0.439 |
| 109 | 311,144 | 0.36% | 0.503 | 21 | 58 | 48 | 17.7% | 0.133 | 0.600 |
| 55 | 310,139 | 0.36% | 0.543 | 19 | 23 | 49 | 45.3% | 0.014 | 0.552 |
| 116 | 306,518 | 0.36% | 0.496 | 21 | 55 | 23 | 20.6% | 0.124 | 0.596 |
| 23 | 305,421 | 0.36% | 0.464 | 22 | 23 | 47 | 42.6% | 0.031 | 0.547 |
| 62 | 286,967 | 0.33% | 0.529 | 20 | 22 | 47 | 43.1% | 0.016 | 0.603 |
| 92 | 273,396 | 0.32% | 0.474 | 22 | 21 | 43 | 43.7% | 0.038 | 0.538 |
| 28 | 230,634 | 0.27% | 0.565 | 19 | 18 | 19 | 24.8% | 0.079 | 0.504 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **63**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v3_subspace_big_i100/ --out runs/clustering/v3_subspace_big_i100//temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 32 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.352, mean 0.368, max 0.733 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 17 ↔ 50 | 0.733 | 0.306 |
| 84 ↔ 126 | 0.730 | 0.511 |
| 37 ↔ 58 | 0.729 | 0.471 |
| 4 ↔ 70 | 0.723 | 0.597 |
| 1 ↔ 46 | 0.721 | 0.368 |
| 1 ↔ 114 | 0.720 | 0.462 |
| 12 ↔ 110 | 0.720 | 0.579 |
| 70 ↔ 107 | 0.719 | 0.386 |
| 99 ↔ 117 | 0.719 | 0.287 |
| 39 ↔ 70 | 0.718 | 0.608 |
| 1 ↔ 110 | 0.717 | 0.455 |
| 4 ↔ 59 | 0.717 | 0.187 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
