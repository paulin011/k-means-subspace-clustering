# Clustering report (subspace_kmeans) — `runs/clustering/v11_k256_d64`

*Generated 2026-08-13 11:19 by `analyze_clusters.py`. K=256 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

## Overview

The model groups the 86,016,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 256 clusters, and fits a 64-dimensional flat (an *affine subspace*: a centroid plus a basis of directions) through each one. A token is assigned to whichever cluster leaves the smallest **orthogonal residual** — the part of the token that its cluster's subspace cannot reconstruct.

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
- **Uⱼ** (`model['U'][j]`, shape `[2048, 64]`): an orthonormal basis for cluster *j*'s subspace; its columns are PC directions in descending eigenvalue order.
- **Orthogonal residual** of a token *x* under cluster *j*: `‖x − μⱼ‖² − ‖Uⱼᵀ(x − μⱼ)‖²`. The first term is the squared distance to the centroid; the second is the part of that distance the subspace *captures*. What's left is the unexplained residual that the assignment minimises.
- **eigvals** (`model['eigvals'][j]`): the top-64 eigenvalues of cluster *j*'s within-cluster covariance — variance along each kept PC direction.
- **trace** (`model['trace'][j]`): mean squared distance of cluster *j*'s tokens to its centroid μⱼ — the cluster's total within-cluster variance.
- **counts** (`model['counts'][j]`): number of tokens in cluster *j*; **wⱼ = counts[j] / Σcounts** is its population share, used to weight every global average.

## Configuration

*How to read this: these are the run's input settings, taken from `model['config']` (plus `model['sampled_files']` for the true file count). `clusters` is K, `dim` is the subspace dimension d (`dim=0` ⇒ plain k-means). `iters` is the maximum number of training iterations (each one reassigns every token to its best cluster, then refits the centroids and subspaces); `tol` is the convergence threshold: once the fraction of tokens that change cluster in an iteration falls below it, the run stops early instead of using all `iters`. Together they bound how long the run takes. `seed` fixes both the token sample and the cluster initialisation so a run is reproducible.*

| parameter | value |
|---|---|
| src | latents_2 |
| num_files | 7000 *(--files-from reused the sample; --num-files=1500 ignored)* |
| tokens_per_file | 12288 |
| clusters | 256 |
| dim | 64 |
| iters | 25 |
| tol | 0.001 |
| linear | False |
| seed | 0 |
| chunk_size | 131072 |
| gpus | 2 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `82ca602ed7e7`
- **Files:** 7000 latent files, 12288 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v11_k256_d64/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v11_k256_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7352.91 | 100.00% | 69 | 22,618,934 |
| 2 | 2254.94 | 84.92% | 7,227 | 1,676,374 |
| 3 | 1948.82 | 43.77% | 38,580 | 1,420,539 |
| 4 | 1870.19 | 24.06% | 74,128 | 1,202,257 |
| 5 | 1832.87 | 16.60% | 82,224 | 1,154,694 |
| 6 | 1808.99 | 12.42% | 105,702 | 1,126,609 |
| 7 | 1792.50 | 9.70% | 121,053 | 1,115,802 |
| 8 | 1780.79 | 7.77% | 125,551 | 1,111,137 |
| 9 | 1771.74 | 6.46% | 130,207 | 1,110,004 |
| 10 | 1764.78 | 5.36% | 131,430 | 1,108,531 |
| 11 | 1759.35 | 4.56% | 130,001 | 1,105,968 |
| 12 | 1755.16 | 3.91% | 129,752 | 1,104,412 |
| 13 | 1751.93 | 3.42% | 128,714 | 1,104,304 |
| 14 | 1749.07 | 3.12% | 127,053 | 1,104,275 |
| 15 | 1746.47 | 2.84% | 125,706 | 1,104,229 |
| 16 | 1744.21 | 2.59% | 124,124 | 1,104,160 |
| 17 | 1742.24 | 2.37% | 121,837 | 1,104,054 |
| 18 | 1740.42 | 2.17% | 119,359 | 1,103,893 |
| 19 | 1738.78 | 1.95% | 116,439 | 1,103,603 |
| 20 | 1737.39 | 1.75% | 114,773 | 1,103,024 |
| 21 | 1736.21 | 1.58% | 114,580 | 1,101,508 |
| 22 | 1735.29 | 1.42% | 113,667 | 1,097,430 |
| 23 | 1734.47 | 1.30% | 112,954 | 1,092,442 |
| 24 | 1733.73 | 1.20% | 112,961 | 1,091,162 |
| 25 | 1733.11 | 1.11% | 113,089 | 1,091,142 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5999**, split into:

- **8.8%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **62.3%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5999, the same denominator as the other two, which is what lets all three add to 100%.
- **28.9%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.692** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 69%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 27 / median 37 / max 40** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=65 = a truncation warning. Your max is 40, below the cap → no cluster truncated, d=64 has headroom.

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`EVR(top-64)` = `Σ eigvals[j] / trace[j]` — fraction of this cluster's own variance captured by its subspace (1.0 = the subspace explains the cluster perfectly; near the global average ⇒ d truncates the spectrum).*
- *`d80` = smallest number of leading PC directions whose eigenvalues reach 80% of `Σ eigvals[j]` (capped at d+1=65 when even all 64 fall short). Low d80 ⇒ a few directions dominate; d80 ≈ d ⇒ a flat spectrum the subspace truncates.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 256 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`maxAff` = this cluster's subspace affinity to its **nearest** neighbour, `maxⱼ≠ᵢ ‖UᵢᵀUⱼ‖²_F / d` ∈ [0,1]. A per-cluster separation score: **high ⇒ some other cluster spans nearly the same directions**, so this row is a merge candidate. The affinity table below lists only the top pairs, so a near-duplicate cluster is invisible there unless its pair happens to rank; this column always shows it.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.13%–1.27% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 1,091,134 | 1.27% | 0.749 | 30 | 78 | 156 | 50.0% | 0.000 | 0.539 |
| 74 | 789,542 | 0.92% | 0.726 | 34 | 57 | 116 | 48.7% | 0.005 | 0.676 |
| 21 | 740,147 | 0.86% | 0.734 | 37 | 53 | 106 | 49.9% | 0.001 | 0.558 |
| 143 | 637,335 | 0.74% | 0.785 | 31 | 46 | 92 | 49.4% | 0.002 | 0.650 |
| 35 | 633,715 | 0.74% | 0.745 | 34 | 46 | 92 | 49.3% | 0.002 | 0.626 |
| 27 | 622,408 | 0.72% | 0.761 | 33 | 45 | 89 | 50.0% | 0.000 | 0.662 |
| 129 | 612,784 | 0.71% | 0.770 | 32 | 44 | 89 | 49.1% | 0.002 | 0.682 |
| 177 | 580,778 | 0.68% | 0.648 | 38 | 63 | 111 | 38.3% | 0.034 | 0.805 |
| 80 | 580,452 | 0.67% | 0.751 | 35 | 42 | 83 | 49.8% | 0.001 | 0.667 |
| 194 | 554,661 | 0.64% | 0.709 | 36 | 48 | 100 | 43.7% | 0.015 | 0.765 |
| 219 | 552,756 | 0.64% | 0.741 | 34 | 40 | 79 | 49.9% | 0.001 | 0.658 |
| 139 | 543,081 | 0.63% | 0.729 | 34 | 39 | 84 | 47.8% | 0.010 | 0.661 |
| 189 | 531,786 | 0.62% | 0.784 | 33 | 38 | 76 | 50.0% | 0.000 | 0.658 |
| 93 | 531,032 | 0.62% | 0.802 | 31 | 38 | 76 | 49.9% | 0.000 | 0.655 |
| 125 | 530,562 | 0.62% | 0.733 | 34 | 38 | 78 | 48.9% | 0.004 | 0.700 |
| 175 | 529,925 | 0.62% | 0.786 | 32 | 38 | 76 | 49.8% | 0.000 | 0.721 |
| 56 | 529,603 | 0.62% | 0.752 | 34 | 38 | 76 | 49.8% | 0.001 | 0.673 |
| 50 | 526,756 | 0.61% | 0.789 | 30 | 39 | 80 | 48.1% | 0.005 | 0.712 |
| 152 | 522,000 | 0.61% | 0.745 | 35 | 38 | 75 | 49.7% | 0.001 | 0.673 |
| 181 | 514,413 | 0.60% | 0.729 | 33 | 38 | 83 | 46.9% | 0.009 | 0.676 |
| 85 | 511,923 | 0.60% | 0.633 | 39 | 45 | 86 | 36.7% | 0.183 | 0.800 |
| 25 | 509,090 | 0.59% | 0.721 | 35 | 37 | 73 | 49.4% | 0.002 | 0.636 |
| 207 | 503,811 | 0.59% | 0.678 | 34 | 83 | 75 | 22.5% | 0.109 | 0.735 |
| 160 | 502,503 | 0.58% | 0.600 | 39 | 49 | 78 | 31.6% | 0.084 | 0.700 |
| 132 | 497,570 | 0.58% | 0.654 | 37 | 109 | 84 | 25.3% | 0.120 | 0.780 |
| 90 | 491,112 | 0.57% | 0.721 | 35 | 36 | 74 | 47.4% | 0.011 | 0.635 |
| 233 | 490,150 | 0.57% | 0.659 | 37 | 83 | 89 | 25.1% | 0.088 | 0.764 |
| 77 | 487,817 | 0.57% | 0.785 | 32 | 35 | 70 | 49.8% | 0.001 | 0.577 |
| 131 | 480,405 | 0.56% | 0.655 | 39 | 37 | 84 | 44.4% | 0.043 | 0.816 |
| 82 | 478,498 | 0.56% | 0.642 | 37 | 104 | 94 | 21.1% | 0.153 | 0.783 |
| 253 | 477,151 | 0.55% | 0.655 | 39 | 39 | 84 | 39.4% | 0.065 | 0.819 |
| 247 | 473,441 | 0.55% | 0.723 | 35 | 41 | 87 | 44.2% | 0.017 | 0.761 |
| 17 | 470,989 | 0.55% | 0.775 | 31 | 34 | 68 | 49.4% | 0.001 | 0.696 |
| 190 | 470,969 | 0.55% | 0.733 | 27 | 86 | 62 | 28.1% | 0.108 | 0.637 |
| 185 | 470,147 | 0.55% | 0.735 | 33 | 34 | 71 | 47.7% | 0.006 | 0.700 |
| 244 | 469,806 | 0.55% | 0.654 | 39 | 37 | 84 | 40.4% | 0.027 | 0.816 |
| 114 | 469,406 | 0.55% | 0.653 | 39 | 34 | 65 | 46.0% | 0.011 | 0.818 |
| 216 | 464,186 | 0.54% | 0.657 | 38 | 46 | 93 | 35.0% | 0.065 | 0.783 |
| 126 | 462,792 | 0.54% | 0.652 | 38 | 44 | 86 | 33.5% | 0.198 | 0.819 |
| 204 | 460,198 | 0.54% | 0.698 | 35 | 45 | 98 | 44.7% | 0.011 | 0.776 |
| 68 | 456,393 | 0.53% | 0.651 | 38 | 53 | 75 | 28.3% | 0.067 | 0.778 |
| 241 | 455,282 | 0.53% | 0.793 | 31 | 34 | 68 | 48.5% | 0.003 | 0.756 |
| 26 | 451,604 | 0.53% | 0.617 | 39 | 85 | 71 | 30.7% | 0.178 | 0.827 |
| 33 | 449,640 | 0.52% | 0.652 | 37 | 95 | 62 | 35.2% | 0.097 | 0.752 |
| 228 | 448,703 | 0.52% | 0.714 | 36 | 34 | 72 | 46.2% | 0.011 | 0.765 |
| 86 | 446,438 | 0.52% | 0.646 | 38 | 50 | 93 | 31.6% | 0.074 | 0.805 |
| 117 | 446,327 | 0.52% | 0.682 | 38 | 35 | 79 | 45.1% | 0.025 | 0.780 |
| 195 | 442,862 | 0.51% | 0.627 | 38 | 47 | 83 | 33.6% | 0.268 | 0.784 |
| 205 | 439,041 | 0.51% | 0.784 | 32 | 32 | 63 | 49.7% | 0.001 | 0.637 |
| 201 | 437,357 | 0.51% | 0.769 | 32 | 32 | 64 | 49.2% | 0.003 | 0.740 |
| 198 | 437,037 | 0.51% | 0.669 | 35 | 121 | 77 | 34.0% | 0.058 | 0.746 |
| 105 | 436,638 | 0.51% | 0.610 | 39 | 61 | 45 | 21.4% | 0.114 | 0.697 |
| 60 | 435,625 | 0.51% | 0.645 | 37 | 92 | 58 | 19.0% | 0.111 | 0.752 |
| 128 | 434,819 | 0.51% | 0.734 | 33 | 32 | 69 | 46.5% | 0.010 | 0.697 |
| 224 | 433,529 | 0.50% | 0.628 | 38 | 82 | 67 | 37.5% | 0.091 | 0.754 |
| 229 | 431,504 | 0.50% | 0.793 | 31 | 31 | 62 | 49.6% | 0.001 | 0.721 |
| 231 | 427,669 | 0.50% | 0.770 | 32 | 32 | 64 | 48.5% | 0.007 | 0.773 |
| 249 | 426,649 | 0.50% | 0.685 | 33 | 117 | 32 | 25.5% | 0.083 | 0.687 |
| 95 | 424,068 | 0.49% | 0.672 | 36 | 85 | 80 | 21.4% | 0.120 | 0.747 |
| 237 | 421,257 | 0.49% | 0.679 | 37 | 41 | 97 | 39.6% | 0.029 | 0.791 |
| 238 | 419,594 | 0.49% | 0.666 | 36 | 32 | 66 | 44.6% | 0.033 | 0.735 |
| 191 | 418,008 | 0.49% | 0.672 | 35 | 31 | 59 | 46.2% | 0.014 | 0.685 |
| 213 | 417,943 | 0.49% | 0.807 | 30 | 30 | 60 | 49.7% | 0.001 | 0.722 |
| 42 | 416,309 | 0.48% | 0.689 | 36 | 35 | 67 | 40.9% | 0.043 | 0.748 |
| 61 | 415,813 | 0.48% | 0.660 | 38 | 31 | 62 | 46.5% | 0.016 | 0.725 |
| 58 | 412,967 | 0.48% | 0.630 | 38 | 54 | 93 | 28.5% | 0.117 | 0.786 |
| 211 | 411,166 | 0.48% | 0.629 | 38 | 48 | 71 | 37.8% | 0.090 | 0.778 |
| 184 | 410,546 | 0.48% | 0.667 | 36 | 30 | 59 | 49.4% | 0.002 | 0.626 |
| 155 | 410,485 | 0.48% | 0.614 | 39 | 31 | 68 | 42.3% | 0.024 | 0.731 |
| 39 | 409,795 | 0.48% | 0.771 | 32 | 30 | 59 | 49.6% | 0.002 | 0.713 |
| 78 | 408,157 | 0.47% | 0.788 | 30 | 31 | 60 | 46.9% | 0.010 | 0.742 |
| 232 | 401,065 | 0.47% | 0.657 | 39 | 30 | 58 | 47.7% | 0.018 | 0.818 |
| 149 | 398,756 | 0.46% | 0.614 | 40 | 126 | 19 | 26.0% | 0.095 | 0.771 |
| 187 | 394,806 | 0.46% | 0.630 | 38 | 54 | 85 | 34.1% | 0.078 | 0.790 |
| 73 | 392,171 | 0.46% | 0.621 | 38 | 119 | 30 | 19.4% | 0.139 | 0.758 |
| 48 | 391,992 | 0.46% | 0.611 | 39 | 101 | 49 | 28.9% | 0.171 | 0.786 |
| 127 | 386,388 | 0.45% | 0.789 | 31 | 28 | 56 | 49.3% | 0.001 | 0.740 |
| 178 | 385,897 | 0.45% | 0.758 | 32 | 29 | 57 | 48.3% | 0.003 | 0.755 |
| 110 | 385,576 | 0.45% | 0.684 | 38 | 29 | 63 | 44.6% | 0.021 | 0.778 |
| 43 | 384,647 | 0.45% | 0.697 | 32 | 52 | 74 | 29.9% | 0.076 | 0.709 |
| 1 | 384,016 | 0.45% | 0.614 | 39 | 74 | 60 | 27.6% | 0.089 | 0.763 |
| 252 | 379,677 | 0.44% | 0.795 | 30 | 28 | 58 | 46.9% | 0.010 | 0.771 |
| 16 | 378,455 | 0.44% | 0.703 | 35 | 28 | 55 | 48.9% | 0.007 | 0.643 |
| 225 | 375,696 | 0.44% | 0.760 | 33 | 28 | 55 | 49.1% | 0.002 | 0.730 |
| 227 | 374,336 | 0.44% | 0.651 | 37 | 91 | 21 | 14.6% | 0.174 | 0.716 |
| 87 | 372,908 | 0.43% | 0.692 | 35 | 27 | 56 | 47.4% | 0.012 | 0.703 |
| 215 | 372,122 | 0.43% | 0.632 | 38 | 89 | 65 | 20.5% | 0.175 | 0.800 |
| 113 | 371,600 | 0.43% | 0.736 | 33 | 27 | 58 | 47.2% | 0.012 | 0.699 |
| 9 | 370,900 | 0.43% | 0.642 | 38 | 56 | 95 | 24.5% | 0.101 | 0.784 |
| 71 | 368,104 | 0.43% | 0.647 | 37 | 68 | 77 | 19.3% | 0.143 | 0.790 |
| 19 | 367,069 | 0.43% | 0.665 | 38 | 39 | 72 | 28.1% | 0.065 | 0.747 |
| 30 | 366,662 | 0.43% | 0.630 | 39 | 81 | 19 | 19.0% | 0.179 | 0.800 |
| 167 | 364,414 | 0.42% | 0.682 | 37 | 30 | 63 | 43.0% | 0.031 | 0.791 |
| 210 | 363,036 | 0.42% | 0.726 | 34 | 33 | 66 | 44.0% | 0.016 | 0.773 |
| 22 | 362,693 | 0.42% | 0.619 | 38 | 83 | 40 | 35.3% | 0.093 | 0.759 |
| 196 | 362,626 | 0.42% | 0.692 | 32 | 93 | 52 | 19.1% | 0.149 | 0.709 |
| 54 | 362,083 | 0.42% | 0.614 | 38 | 163 | 22 | 28.4% | 0.091 | 0.758 |
| 242 | 360,000 | 0.42% | 0.622 | 39 | 41 | 43 | 27.3% | 0.094 | 0.707 |
| 245 | 359,982 | 0.42% | 0.641 | 37 | 100 | 38 | 19.2% | 0.119 | 0.783 |
| 107 | 358,031 | 0.42% | 0.781 | 32 | 26 | 52 | 49.3% | 0.001 | 0.709 |
| 179 | 356,937 | 0.41% | 0.692 | 30 | 128 | 0 | 35.6% | 0.046 | 0.692 |
| 246 | 355,377 | 0.41% | 0.666 | 35 | 70 | 83 | 20.6% | 0.131 | 0.732 |
| 163 | 354,294 | 0.41% | 0.627 | 37 | 68 | 51 | 16.8% | 0.112 | 0.706 |
| 88 | 352,363 | 0.41% | 0.726 | 35 | 26 | 55 | 46.8% | 0.010 | 0.653 |
| 75 | 351,570 | 0.41% | 0.634 | 38 | 56 | 72 | 21.5% | 0.097 | 0.778 |
| 251 | 350,446 | 0.41% | 0.664 | 36 | 120 | 15 | 40.0% | 0.035 | 0.746 |
| 89 | 349,122 | 0.41% | 0.702 | 36 | 25 | 50 | 49.6% | 0.004 | 0.607 |
| 5 | 348,867 | 0.41% | 0.643 | 37 | 159 | 8 | 33.4% | 0.068 | 0.747 |
| 123 | 348,623 | 0.41% | 0.804 | 30 | 25 | 50 | 49.7% | 0.001 | 0.722 |
| 146 | 345,998 | 0.40% | 0.697 | 36 | 25 | 50 | 49.4% | 0.004 | 0.600 |
| 76 | 345,300 | 0.40% | 0.717 | 34 | 25 | 59 | 46.0% | 0.010 | 0.697 |
| 133 | 343,050 | 0.40% | 0.705 | 36 | 28 | 62 | 45.7% | 0.008 | 0.749 |
| 67 | 342,405 | 0.40% | 0.643 | 39 | 31 | 64 | 39.2% | 0.064 | 0.763 |
| 4 | 342,374 | 0.40% | 0.703 | 36 | 28 | 60 | 45.4% | 0.012 | 0.764 |
| 202 | 342,235 | 0.40% | 0.633 | 37 | 122 | 7 | 26.2% | 0.154 | 0.731 |
| 148 | 340,436 | 0.40% | 0.636 | 38 | 65 | 55 | 20.0% | 0.149 | 0.803 |
| 137 | 339,411 | 0.39% | 0.636 | 38 | 69 | 29 | 17.9% | 0.177 | 0.773 |
| 166 | 338,877 | 0.39% | 0.678 | 33 | 81 | 31 | 31.2% | 0.060 | 0.741 |
| 234 | 338,297 | 0.39% | 0.624 | 37 | 78 | 32 | 22.9% | 0.166 | 0.757 |
| 36 | 336,659 | 0.39% | 0.625 | 38 | 137 | 22 | 21.2% | 0.104 | 0.739 |
| 156 | 336,019 | 0.39% | 0.818 | 29 | 25 | 48 | 50.0% | 0.000 | 0.682 |
| 0 | 335,845 | 0.39% | 0.777 | 33 | 24 | 48 | 49.8% | 0.001 | 0.596 |
| 254 | 335,163 | 0.39% | 0.669 | 36 | 28 | 56 | 37.0% | 0.064 | 0.729 |
| 171 | 332,425 | 0.39% | 0.638 | 36 | 161 | 0 | 17.8% | 0.130 | 0.736 |
| 47 | 331,919 | 0.39% | 0.671 | 38 | 25 | 55 | 43.0% | 0.031 | 0.725 |
| 154 | 331,157 | 0.38% | 0.694 | 35 | 24 | 48 | 49.1% | 0.003 | 0.661 |
| 192 | 330,898 | 0.38% | 0.646 | 39 | 27 | 56 | 35.0% | 0.129 | 0.783 |
| 55 | 330,852 | 0.38% | 0.715 | 30 | 166 | 4 | 24.2% | 0.102 | 0.673 |
| 144 | 328,082 | 0.38% | 0.692 | 31 | 251 | 0 | 35.3% | 0.024 | 0.694 |
| 124 | 327,570 | 0.38% | 0.728 | 35 | 25 | 50 | 47.7% | 0.005 | 0.716 |
| 153 | 326,737 | 0.38% | 0.695 | 31 | 126 | 15 | 24.0% | 0.107 | 0.692 |
| 40 | 324,903 | 0.38% | 0.619 | 39 | 144 | 19 | 27.8% | 0.134 | 0.771 |
| 223 | 322,980 | 0.38% | 0.749 | 35 | 24 | 48 | 48.6% | 0.004 | 0.702 |
| 200 | 321,519 | 0.37% | 0.732 | 34 | 25 | 52 | 47.0% | 0.006 | 0.771 |
| 96 | 320,339 | 0.37% | 0.694 | 36 | 24 | 53 | 44.5% | 0.022 | 0.696 |
| 199 | 318,339 | 0.37% | 0.698 | 36 | 27 | 62 | 43.7% | 0.017 | 0.764 |
| 142 | 317,222 | 0.37% | 0.779 | 30 | 24 | 45 | 45.4% | 0.010 | 0.741 |
| 150 | 313,787 | 0.36% | 0.671 | 39 | 23 | 54 | 44.3% | 0.029 | 0.797 |
| 59 | 313,560 | 0.36% | 0.735 | 34 | 25 | 52 | 46.4% | 0.006 | 0.776 |
| 18 | 313,145 | 0.36% | 0.684 | 35 | 27 | 41 | 31.7% | 0.035 | 0.673 |
| 6 | 311,522 | 0.36% | 0.683 | 37 | 25 | 58 | 42.8% | 0.028 | 0.746 |
| 70 | 310,500 | 0.36% | 0.748 | 32 | 24 | 48 | 47.4% | 0.007 | 0.758 |
| 147 | 309,430 | 0.36% | 0.639 | 38 | 29 | 53 | 31.5% | 0.071 | 0.691 |
| 135 | 308,784 | 0.36% | 0.631 | 37 | 338 | 0 | 31.9% | 0.067 | 0.720 |
| 53 | 307,588 | 0.36% | 0.647 | 38 | 31 | 58 | 30.5% | 0.078 | 0.704 |
| 111 | 307,334 | 0.36% | 0.774 | 30 | 23 | 47 | 47.8% | 0.004 | 0.773 |
| 14 | 307,132 | 0.36% | 0.789 | 31 | 22 | 44 | 49.9% | 0.001 | 0.682 |
| 188 | 306,696 | 0.36% | 0.616 | 39 | 142 | 22 | 27.9% | 0.101 | 0.726 |
| 45 | 304,749 | 0.35% | 0.632 | 38 | 63 | 70 | 21.0% | 0.285 | 0.765 |
| 161 | 302,775 | 0.35% | 0.613 | 40 | 74 | 36 | 20.3% | 0.145 | 0.748 |
| 186 | 302,035 | 0.35% | 0.696 | 39 | 22 | 48 | 47.5% | 0.008 | 0.705 |
| 157 | 301,993 | 0.35% | 0.724 | 34 | 23 | 40 | 45.1% | 0.013 | 0.760 |
| 98 | 298,484 | 0.35% | 0.718 | 27 | 272 | 0 | 27.8% | 0.073 | 0.672 |
| 52 | 298,349 | 0.35% | 0.640 | 37 | 152 | 4 | 24.1% | 0.100 | 0.737 |
| 209 | 292,455 | 0.34% | 0.643 | 37 | 37 | 42 | 22.6% | 0.076 | 0.716 |
| 101 | 292,332 | 0.34% | 0.637 | 38 | 72 | 51 | 17.9% | 0.131 | 0.786 |
| 119 | 291,646 | 0.34% | 0.754 | 33 | 23 | 46 | 45.4% | 0.013 | 0.763 |
| 206 | 290,074 | 0.34% | 0.709 | 34 | 22 | 43 | 45.3% | 0.023 | 0.703 |
| 120 | 289,734 | 0.34% | 0.665 | 39 | 23 | 48 | 40.4% | 0.049 | 0.780 |
| 138 | 289,304 | 0.34% | 0.682 | 36 | 21 | 45 | 45.5% | 0.015 | 0.701 |
| 29 | 286,966 | 0.33% | 0.647 | 37 | 53 | 40 | 16.2% | 0.179 | 0.718 |
| 220 | 286,189 | 0.33% | 0.644 | 38 | 37 | 33 | 22.7% | 0.093 | 0.700 |
| 239 | 283,732 | 0.33% | 0.661 | 38 | 23 | 49 | 39.2% | 0.060 | 0.735 |
| 203 | 282,765 | 0.33% | 0.618 | 40 | 55 | 29 | 29.4% | 0.175 | 0.827 |
| 169 | 282,268 | 0.33% | 0.671 | 33 | 138 | 9 | 29.2% | 0.087 | 0.711 |
| 183 | 280,350 | 0.33% | 0.690 | 36 | 21 | 42 | 48.0% | 0.007 | 0.649 |
| 72 | 280,343 | 0.33% | 0.809 | 32 | 21 | 40 | 49.9% | 0.001 | 0.593 |
| 3 | 276,281 | 0.32% | 0.673 | 38 | 20 | 41 | 45.9% | 0.017 | 0.712 |
| 212 | 275,397 | 0.32% | 0.650 | 35 | 231 | 0 | 31.3% | 0.057 | 0.741 |
| 170 | 274,243 | 0.32% | 0.619 | 39 | 108 | 15 | 29.5% | 0.082 | 0.748 |
| 94 | 271,132 | 0.32% | 0.743 | 33 | 22 | 44 | 43.9% | 0.018 | 0.773 |
| 10 | 270,586 | 0.31% | 0.801 | 29 | 20 | 42 | 46.1% | 0.020 | 0.771 |
| 115 | 270,199 | 0.31% | 0.679 | 37 | 20 | 40 | 48.0% | 0.010 | 0.710 |
| 180 | 267,863 | 0.31% | 0.764 | 32 | 23 | 41 | 41.2% | 0.037 | 0.742 |
| 240 | 265,445 | 0.31% | 0.629 | 38 | 23 | 25 | 26.8% | 0.084 | 0.675 |
| 7 | 262,458 | 0.31% | 0.616 | 38 | 87 | 4 | 14.3% | 0.130 | 0.751 |
| 57 | 261,702 | 0.30% | 0.621 | 38 | 182 | 0 | 20.8% | 0.161 | 0.713 |
| 103 | 261,603 | 0.30% | 0.633 | 39 | 38 | 53 | 22.1% | 0.130 | 0.743 |
| 250 | 261,209 | 0.30% | 0.633 | 39 | 30 | 42 | 26.9% | 0.110 | 0.676 |
| 235 | 259,735 | 0.30% | 0.657 | 33 | 248 | 0 | 33.5% | 0.034 | 0.702 |
| 34 | 257,254 | 0.30% | 0.716 | 35 | 19 | 38 | 47.8% | 0.008 | 0.673 |
| 159 | 256,608 | 0.30% | 0.636 | 35 | 254 | 0 | 18.0% | 0.115 | 0.729 |
| 106 | 248,437 | 0.29% | 0.681 | 40 | 19 | 38 | 46.0% | 0.008 | 0.703 |
| 173 | 244,976 | 0.28% | 0.647 | 35 | 306 | 0 | 31.5% | 0.050 | 0.741 |
| 134 | 243,747 | 0.28% | 0.656 | 40 | 18 | 35 | 48.5% | 0.003 | 0.619 |
| 46 | 241,554 | 0.28% | 0.692 | 39 | 18 | 39 | 43.5% | 0.025 | 0.797 |
| 97 | 241,538 | 0.28% | 0.652 | 39 | 21 | 43 | 37.0% | 0.047 | 0.719 |
| 41 | 241,516 | 0.28% | 0.723 | 34 | 18 | 40 | 46.1% | 0.010 | 0.742 |
| 104 | 241,420 | 0.28% | 0.650 | 38 | 29 | 49 | 25.8% | 0.166 | 0.803 |
| 81 | 240,732 | 0.28% | 0.703 | 36 | 18 | 36 | 46.5% | 0.012 | 0.632 |
| 116 | 239,669 | 0.28% | 0.675 | 37 | 18 | 39 | 43.1% | 0.035 | 0.662 |
| 108 | 239,339 | 0.28% | 0.655 | 37 | 30 | 58 | 22.7% | 0.142 | 0.784 |
| 168 | 239,156 | 0.28% | 0.649 | 40 | 19 | 39 | 38.9% | 0.038 | 0.707 |
| 51 | 238,497 | 0.28% | 0.698 | 37 | 18 | 34 | 49.2% | 0.004 | 0.637 |
| 49 | 236,354 | 0.27% | 0.761 | 34 | 18 | 38 | 43.3% | 0.026 | 0.610 |
| 99 | 233,604 | 0.27% | 0.730 | 37 | 17 | 34 | 48.7% | 0.003 | 0.727 |
| 100 | 232,156 | 0.27% | 0.620 | 39 | 29 | 42 | 25.7% | 0.097 | 0.711 |
| 84 | 232,084 | 0.27% | 0.629 | 39 | 30 | 26 | 30.5% | 0.079 | 0.751 |
| 182 | 231,886 | 0.27% | 0.635 | 39 | 31 | 48 | 23.8% | 0.106 | 0.743 |
| 69 | 231,850 | 0.27% | 0.687 | 37 | 19 | 42 | 44.4% | 0.023 | 0.764 |
| 221 | 230,676 | 0.27% | 0.714 | 38 | 17 | 36 | 47.0% | 0.007 | 0.767 |
| 62 | 225,968 | 0.26% | 0.688 | 36 | 17 | 33 | 48.2% | 0.003 | 0.663 |
| 83 | 225,564 | 0.26% | 0.776 | 30 | 25 | 29 | 37.8% | 0.040 | 0.719 |
| 172 | 224,842 | 0.26% | 0.683 | 37 | 17 | 35 | 46.7% | 0.008 | 0.671 |
| 15 | 222,286 | 0.26% | 0.647 | 39 | 16 | 33 | 47.1% | 0.012 | 0.682 |
| 31 | 220,980 | 0.26% | 0.628 | 39 | 35 | 20 | 34.0% | 0.069 | 0.712 |
| 13 | 215,677 | 0.25% | 0.645 | 37 | 32 | 12 | 18.0% | 0.173 | 0.694 |
| 255 | 215,527 | 0.25% | 0.761 | 32 | 25 | 28 | 36.2% | 0.047 | 0.736 |
| 23 | 214,577 | 0.25% | 0.700 | 39 | 16 | 32 | 48.4% | 0.006 | 0.720 |
| 118 | 213,421 | 0.25% | 0.633 | 39 | 17 | 29 | 34.1% | 0.085 | 0.726 |
| 121 | 212,897 | 0.25% | 0.711 | 36 | 16 | 31 | 47.4% | 0.006 | 0.626 |
| 214 | 212,570 | 0.25% | 0.744 | 27 | 254 | 0 | 30.6% | 0.042 | 0.618 |
| 112 | 209,876 | 0.24% | 0.676 | 40 | 16 | 31 | 46.9% | 0.009 | 0.686 |
| 37 | 209,678 | 0.24% | 0.663 | 40 | 16 | 35 | 44.2% | 0.039 | 0.745 |
| 176 | 209,603 | 0.24% | 0.671 | 39 | 17 | 36 | 44.4% | 0.023 | 0.749 |
| 164 | 209,134 | 0.24% | 0.666 | 40 | 17 | 36 | 42.2% | 0.032 | 0.754 |
| 65 | 207,760 | 0.24% | 0.645 | 39 | 17 | 34 | 34.2% | 0.053 | 0.692 |
| 64 | 207,160 | 0.24% | 0.657 | 39 | 16 | 26 | 41.0% | 0.042 | 0.717 |
| 11 | 206,804 | 0.24% | 0.697 | 36 | 16 | 27 | 35.4% | 0.033 | 0.667 |
| 230 | 205,614 | 0.24% | 0.708 | 36 | 15 | 30 | 48.9% | 0.005 | 0.627 |
| 248 | 203,945 | 0.24% | 0.689 | 38 | 15 | 29 | 41.9% | 0.038 | 0.759 |
| 208 | 202,624 | 0.24% | 0.674 | 39 | 15 | 31 | 46.1% | 0.014 | 0.694 |
| 109 | 197,904 | 0.23% | 0.672 | 36 | 37 | 7 | 16.7% | 0.138 | 0.661 |
| 158 | 197,495 | 0.23% | 0.646 | 39 | 17 | 35 | 30.3% | 0.120 | 0.691 |
| 44 | 196,401 | 0.23% | 0.676 | 39 | 15 | 29 | 42.9% | 0.038 | 0.766 |
| 79 | 194,751 | 0.23% | 0.681 | 38 | 14 | 28 | 49.2% | 0.004 | 0.605 |
| 174 | 188,112 | 0.22% | 0.697 | 37 | 14 | 29 | 46.9% | 0.018 | 0.777 |
| 8 | 187,391 | 0.22% | 0.686 | 37 | 14 | 27 | 46.5% | 0.014 | 0.717 |
| 91 | 185,561 | 0.22% | 0.681 | 37 | 14 | 30 | 36.6% | 0.061 | 0.703 |
| 193 | 184,836 | 0.21% | 0.655 | 39 | 14 | 27 | 48.7% | 0.009 | 0.704 |
| 2 | 181,861 | 0.21% | 0.718 | 36 | 13 | 26 | 49.9% | 0.000 | 0.524 |
| 217 | 180,243 | 0.21% | 0.683 | 38 | 14 | 26 | 48.3% | 0.014 | 0.654 |
| 162 | 177,974 | 0.21% | 0.671 | 38 | 15 | 32 | 34.3% | 0.056 | 0.696 |
| 130 | 177,046 | 0.21% | 0.686 | 38 | 13 | 28 | 47.1% | 0.011 | 0.651 |
| 102 | 176,324 | 0.20% | 0.666 | 36 | 21 | 19 | 18.6% | 0.117 | 0.694 |
| 243 | 175,147 | 0.20% | 0.682 | 38 | 14 | 30 | 42.4% | 0.014 | 0.749 |
| 222 | 174,438 | 0.20% | 0.670 | 40 | 13 | 29 | 43.6% | 0.032 | 0.766 |
| 63 | 172,939 | 0.20% | 0.719 | 35 | 13 | 27 | 42.8% | 0.025 | 0.648 |
| 236 | 168,441 | 0.20% | 0.647 | 39 | 13 | 28 | 39.6% | 0.043 | 0.679 |
| 226 | 165,665 | 0.19% | 0.666 | 39 | 14 | 29 | 41.4% | 0.025 | 0.711 |
| 151 | 163,867 | 0.19% | 0.653 | 39 | 12 | 26 | 45.3% | 0.035 | 0.724 |
| 28 | 163,706 | 0.19% | 0.760 | 33 | 12 | 19 | 36.4% | 0.042 | 0.610 |
| 20 | 161,447 | 0.19% | 0.721 | 35 | 12 | 23 | 48.5% | 0.004 | 0.661 |
| 218 | 158,581 | 0.18% | 0.669 | 38 | 12 | 25 | 44.3% | 0.013 | 0.679 |
| 165 | 154,525 | 0.18% | 0.737 | 36 | 12 | 23 | 48.0% | 0.006 | 0.508 |
| 12 | 154,268 | 0.18% | 0.701 | 36 | 12 | 21 | 45.1% | 0.016 | 0.672 |
| 32 | 145,056 | 0.17% | 0.722 | 35 | 11 | 22 | 47.0% | 0.008 | 0.605 |
| 92 | 141,941 | 0.17% | 0.685 | 38 | 11 | 20 | 45.9% | 0.016 | 0.637 |
| 145 | 138,643 | 0.16% | 0.662 | 39 | 10 | 20 | 47.6% | 0.008 | 0.726 |
| 197 | 135,048 | 0.16% | 0.649 | 40 | 10 | 17 | 36.5% | 0.053 | 0.731 |
| 140 | 131,801 | 0.15% | 0.685 | 38 | 10 | 19 | 49.2% | 0.006 | 0.669 |
| 24 | 130,155 | 0.15% | 0.673 | 37 | 18 | 20 | 22.8% | 0.116 | 0.694 |
| 141 | 128,786 | 0.15% | 0.703 | 30 | 33 | 8 | 19.9% | 0.108 | 0.648 |
| 38 | 125,544 | 0.15% | 0.671 | 39 | 10 | 17 | 43.4% | 0.023 | 0.719 |
| 136 | 124,021 | 0.14% | 0.679 | 39 | 9 | 18 | 48.9% | 0.010 | 0.682 |
| 122 | 112,271 | 0.13% | 0.693 | 37 | 9 | 15 | 44.3% | 0.023 | 0.662 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **57, 98, 135, 144, 159, 171, 173, 179, 212, 214, 235**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v11_k256_d64 --out runs/clustering/v11_k256_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.411, mean 0.423, max 0.827 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 26 ↔ 203 | 0.827 | 0.816 |
| 126 ↔ 253 | 0.819 | 0.731 |
| 114 ↔ 232 | 0.818 | 0.806 |
| 131 ↔ 244 | 0.816 | 0.863 |
| 232 ↔ 253 | 0.808 | 0.778 |
| 86 ↔ 177 | 0.805 | 0.778 |
| 104 ↔ 148 | 0.803 | 0.670 |
| 85 ↔ 253 | 0.800 | 0.678 |
| 30 ↔ 215 | 0.800 | 0.698 |
| 46 ↔ 150 | 0.797 | 0.804 |
| 167 ↔ 237 | 0.791 | 0.752 |
| 71 ↔ 187 | 0.790 | 0.737 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
