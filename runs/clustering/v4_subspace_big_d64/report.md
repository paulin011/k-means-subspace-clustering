# Clustering report (subspace_kmeans) — `runs/clustering/v4_subspace_big_d64`

*Generated 2026-07-31 15:26 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

## Overview

The model groups the 86,016,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 128 clusters, and fits a 64-dimensional flat (an *affine subspace*: a centroid plus a basis of directions) through each one. A token is assigned to whichever cluster leaves the smallest **orthogonal residual** — the part of the token that its cluster's subspace cannot reconstruct.

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
| clusters | 128 |
| dim | 64 |
| iters | 25 |
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
  python3 src/subspace_kmeans.py --files-from runs/clustering/v4_subspace_big_d64/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v4_subspace_big_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7659.20 | 100.00% | 127 | 23,702,502 |
| 2 | 2341.22 | 77.58% | 1 | 3,982,684 |
| 3 | 2061.09 | 41.54% | 1 | 2,322,971 |
| 4 | 1986.33 | 22.32% | 2 | 1,931,379 |
| 5 | 1956.48 | 14.45% | 1 | 1,857,788 |
| 6 | 1940.54 | 10.50% | 1 | 1,826,473 |
| 7 | 1930.45 | 8.28% | 1 | 1,805,881 |
| 8 | 1923.26 | 6.86% | 1 | 1,786,130 |
| 9 | 1917.89 | 5.78% | 1 | 1,773,980 |
| 10 | 1913.55 | 5.02% | 2 | 1,756,582 |
| 11 | 1909.88 | 4.40% | 1 | 1,746,648 |
| 12 | 1906.88 | 3.87% | 1 | 1,742,497 |
| 13 | 1904.36 | 3.45% | 1 | 1,733,463 |
| 14 | 1902.14 | 3.14% | 1 | 1,725,030 |
| 15 | 1900.28 | 2.85% | 1 | 1,710,312 |
| 16 | 1898.63 | 2.62% | 1 | 1,686,483 |
| 17 | 1897.13 | 2.44% | 1 | 1,657,545 |
| 18 | 1895.78 | 2.25% | 1 | 1,632,698 |
| 19 | 1894.68 | 2.08% | 1 | 1,604,940 |
| 20 | 1893.69 | 1.96% | 1 | 1,567,931 |
| 21 | 1892.75 | 1.84% | 5 | 1,539,840 |
| 22 | 1891.89 | 1.73% | 1 | 1,519,600 |
| 23 | 1891.09 | 1.67% | 1 | 1,479,179 |
| 24 | 1890.26 | 1.58% | 1 | 1,478,181 |
| 25 | 1889.66 | 1.49% | 1 | 1,478,198 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5998**, split into:

- **8.4%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **60.1%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5998, the same denominator as the other two, which is what lets all three add to 100%.
- **31.5%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.664** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 66%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 28 / median 37 / max 65** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=65 = a truncation warning. Your max is 65, at the cap → a cluster is truncated; consider raising `--dim`.

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`EVR(top-64)` = `Σ eigvals[j] / trace[j]` — fraction of this cluster's own variance captured by its subspace (1.0 = the subspace explains the cluster perfectly; near the global average ⇒ d truncates the spectrum).*
- *`d80` = smallest number of leading PC directions whose eigenvalues reach 80% of `Σ eigvals[j]` (capped at d+1=65 when even all 64 fall short). Low d80 ⇒ a few directions dominate; d80 ≈ d ⇒ a flat spectrum the subspace truncates.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files` = share of the 7000 sampled time steps (latent files, 6-hourly) in which the cluster appears at least once. ≈100% ⇒ always present in time.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data; `cells@50%` = number of cells holding half the cluster's tokens (low = localized); `owned` = cells where this cluster is the most common label; `files` = share of the 7000 sampled time steps where the cluster appears; `tCV` = coefficient of variation of its share across time deciles (0 = constant in time).

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files | tCV |
|---|---|---|---|---|---|---|---|---|
| 27 | 1,478,198 | 1.7% | 0.713 | 32 | 106 | 212 | 100% | 0.00 |
| 66 | 1,460,322 | 1.7% | 0.728 | 31 | 105 | 209 | 100% | 0.00 |
| 50 | 1,355,978 | 1.6% | 0.750 | 29 | 99 | 208 | 100% | 0.01 |
| 78 | 1,201,732 | 1.4% | 0.728 | 29 | 169 | 153 | 100% | 0.06 |
| 74 | 1,171,060 | 1.4% | 0.703 | 34 | 84 | 170 | 100% | 0.00 |
| 114 | 1,139,747 | 1.3% | 0.632 | 39 | 90 | 175 | 100% | 0.06 |
| 42 | 1,108,410 | 1.3% | 0.655 | 36 | 119 | 182 | 100% | 0.03 |
| 6 | 1,094,357 | 1.3% | 0.666 | 34 | 188 | 217 | 100% | 0.06 |
| 17 | 1,079,263 | 1.3% | 0.745 | 30 | 78 | 154 | 100% | 0.00 |
| 39 | 1,044,110 | 1.2% | 0.680 | 33 | 174 | 177 | 100% | 0.04 |
| 119 | 1,031,287 | 1.2% | 0.696 | 32 | 141 | 190 | 100% | 0.03 |
| 117 | 1,025,402 | 1.2% | 0.645 | 38 | 105 | 212 | 100% | 0.06 |
| 4 | 1,023,376 | 1.2% | 0.650 | 37 | 189 | 180 | 100% | 0.05 |
| 85 | 1,001,659 | 1.2% | 0.634 | 39 | 79 | 157 | 100% | 0.07 |
| 125 | 1,000,632 | 1.2% | 0.698 | 33 | 75 | 167 | 100% | 0.02 |
| 35 | 981,117 | 1.1% | 0.714 | 34 | 71 | 145 | 100% | 0.00 |
| 93 | 941,493 | 1.1% | 0.765 | 30 | 68 | 136 | 100% | 0.00 |
| 86 | 941,407 | 1.1% | 0.635 | 38 | 105 | 165 | 100% | 0.05 |
| 83 | 932,810 | 1.1% | 0.739 | 31 | 71 | 151 | 100% | 0.01 |
| 70 | 919,353 | 1.1% | 0.686 | 34 | 106 | 212 | 100% | 0.03 |
| 56 | 907,862 | 1.1% | 0.718 | 34 | 65 | 132 | 100% | 0.00 |
| 99 | 899,834 | 1.0% | 0.627 | 38 | 105 | 180 | 100% | 0.05 |
| 0 | 899,110 | 1.0% | 0.723 | 33 | 64 | 129 | 100% | 0.00 |
| 61 | 898,252 | 1.0% | 0.644 | 37 | 68 | 147 | 100% | 0.03 |
| 88 | 883,911 | 1.0% | 0.663 | 34 | 146 | 139 | 100% | 0.10 |
| 126 | 864,804 | 1.0% | 0.631 | 38 | 109 | 189 | 97% | 0.16 |
| 43 | 847,542 | 1.0% | 0.690 | 30 | 191 | 122 | 97% | 0.12 |
| 21 | 845,765 | 1.0% | 0.715 | 37 | 61 | 121 | 100% | 0.00 |
| 82 | 842,453 | 1.0% | 0.632 | 37 | 177 | 142 | 82% | 0.14 |
| 67 | 835,690 | 1.0% | 0.625 | 38 | 169 | 171 | 100% | 0.20 |
| 58 | 834,029 | 1.0% | 0.642 | 37 | 132 | 116 | 100% | 0.08 |
| 121 | 825,209 | 1.0% | 0.687 | 32 | 90 | 142 | 100% | 0.07 |
| 10 | 821,282 | 1.0% | 0.692 | 35 | 68 | 148 | 100% | 0.02 |
| 76 | 816,959 | 0.9% | 0.668 | 33 | 108 | 121 | 100% | 0.05 |
| 105 | 815,422 | 0.9% | 0.590 | 39 | 76 | 125 | 100% | 0.05 |
| 34 | 811,525 | 0.9% | 0.634 | 36 | 89 | 68 | 100% | 0.10 |
| 75 | 805,992 | 0.9% | 0.611 | 39 | 102 | 180 | 100% | 0.09 |
| 60 | 805,468 | 0.9% | 0.628 | 37 | 169 | 116 | 100% | 0.10 |
| 89 | 803,782 | 0.9% | 0.651 | 37 | 58 | 116 | 100% | 0.00 |
| 33 | 800,505 | 0.9% | 0.647 | 37 | 176 | 130 | 100% | 0.08 |
| 111 | 788,869 | 0.9% | 0.718 | 29 | 252 | 21 | 100% | 0.02 |
| 107 | 786,495 | 0.9% | 0.706 | 33 | 66 | 139 | 100% | 0.01 |
| 1 | 780,614 | 0.9% | 0.612 | 38 | 82 | 126 | 100% | 0.12 |
| 41 | 778,361 | 0.9% | 0.664 | 33 | 349 | 6 | 100% | 0.06 |
| 95 | 772,026 | 0.9% | 0.659 | 36 | 146 | 128 | 100% | 0.10 |
| 113 | 762,784 | 0.9% | 0.700 | 28 | 145 | 96 | 100% | 0.05 |
| 25 | 758,333 | 0.9% | 0.702 | 35 | 55 | 110 | 100% | 0.00 |
| 127 | 741,281 | 0.9% | 0.774 | 33 | 53 | 108 | 100% | 0.00 |
| 115 | 728,943 | 0.8% | 0.644 | 36 | 57 | 125 | 100% | 0.03 |
| 90 | 726,782 | 0.8% | 0.703 | 35 | 52 | 117 | 100% | 0.01 |
| 59 | 723,254 | 0.8% | 0.695 | 35 | 60 | 125 | 100% | 0.01 |
| 18 | 721,229 | 0.8% | 0.665 | 32 | 222 | 42 | 100% | 0.03 |
| 48 | 713,321 | 0.8% | 0.601 | 39 | 158 | 83 | 100% | 0.14 |
| 110 | 710,479 | 0.8% | 0.662 | 38 | 57 | 133 | 100% | 0.02 |
| 5 | 701,999 | 0.8% | 0.640 | 37 | 263 | 19 | 100% | 0.05 |
| 80 | 700,085 | 0.8% | 0.732 | 35 | 51 | 101 | 100% | 0.00 |
| 73 | 698,614 | 0.8% | 0.610 | 38 | 195 | 71 | 89% | 0.13 |
| 52 | 697,128 | 0.8% | 0.657 | 34 | 163 | 89 | 90% | 0.13 |
| 98 | 692,746 | 0.8% | 0.674 | 31 | 309 | 0 | 100% | 0.04 |
| 16 | 684,538 | 0.8% | 0.666 | 35 | 50 | 100 | 100% | 0.01 |
| 26 | 682,347 | 0.8% | 0.605 | 40 | 145 | 84 | 100% | 0.08 |
| 71 | 672,429 | 0.8% | 0.626 | 38 | 111 | 130 | 100% | 0.12 |
| 22 | 672,176 | 0.8% | 0.606 | 39 | 105 | 98 | 100% | 0.07 |
| 123 | 669,338 | 0.8% | 0.772 | 31 | 48 | 96 | 100% | 0.00 |
| 9 | 668,359 | 0.8% | 0.633 | 38 | 111 | 145 | 100% | 0.11 |
| 19 | 661,401 | 0.8% | 0.647 | 37 | 62 | 112 | 100% | 0.05 |
| 46 | 660,932 | 0.8% | 0.651 | 38 | 50 | 91 | 100% | 0.08 |
| 30 | 660,465 | 0.8% | 0.635 | 38 | 140 | 27 | 95% | 0.13 |
| 53 | 659,102 | 0.8% | 0.624 | 38 | 60 | 90 | 100% | 0.05 |
| 11 | 650,914 | 0.8% | 0.641 | 35 | 89 | 82 | 98% | 0.08 |
| 84 | 626,275 | 0.7% | 0.617 | 38 | 97 | 92 | 100% | 0.08 |
| 68 | 610,561 | 0.7% | 0.665 | 37 | 48 | 93 | 100% | 0.05 |
| 54 | 610,464 | 0.7% | 0.604 | 39 | 199 | 47 | 100% | 0.06 |
| 106 | 593,973 | 0.7% | 0.617 | 39 | 163 | 38 | 100% | 0.08 |
| 100 | 582,939 | 0.7% | 0.609 | 38 | 131 | 73 | 88% | 0.18 |
| 15 | 582,024 | 0.7% | 0.601 | 39 | 46 | 92 | 100% | 0.04 |
| 24 | 579,339 | 0.7% | 0.613 | 37 | 101 | 77 | 72% | 0.14 |
| 64 | 574,400 | 0.7% | 0.592 | 38 | 81 | 51 | 87% | 0.14 |
| 29 | 573,945 | 0.7% | 0.623 | 38 | 90 | 87 | 73% | 0.13 |
| 104 | 572,306 | 0.7% | 0.616 | 38 | 107 | 89 | 94% | 0.14 |
| 120 | 566,048 | 0.7% | 0.649 | 39 | 45 | 92 | 100% | 0.06 |
| 12 | 565,276 | 0.7% | 0.638 | 38 | 69 | 78 | 100% | 0.07 |
| 45 | 558,198 | 0.6% | 0.624 | 38 | 103 | 96 | 100% | 0.07 |
| 7 | 556,068 | 0.6% | 0.610 | 38 | 156 | 18 | 99% | 0.12 |
| 94 | 543,046 | 0.6% | 0.717 | 29 | 155 | 27 | 100% | 0.05 |
| 101 | 541,428 | 0.6% | 0.619 | 38 | 147 | 84 | 100% | 0.08 |
| 31 | 524,425 | 0.6% | 0.615 | 38 | 149 | 33 | 100% | 0.08 |
| 23 | 517,622 | 0.6% | 0.611 | 40 | 53 | 56 | 100% | 0.08 |
| 13 | 516,644 | 0.6% | 0.621 | 36 | 537 | 0 | 100% | 0.07 |
| 91 | 512,341 | 0.6% | 0.642 | 37 | 39 | 75 | 100% | 0.03 |
| 40 | 500,456 | 0.6% | 0.607 | 39 | 149 | 45 | 100% | 0.07 |
| 57 | 496,922 | 0.6% | 0.624 | 38 | 180 | 25 | 99% | 0.11 |
| 81 | 496,427 | 0.6% | 0.660 | 36 | 36 | 76 | 100% | 0.01 |
| 3 | 495,026 | 0.6% | 0.649 | 38 | 37 | 74 | 100% | 0.02 |
| 36 | 494,835 | 0.6% | 0.632 | 36 | 273 | 0 | 100% | 0.06 |
| 97 | 482,749 | 0.6% | 0.626 | 39 | 44 | 84 | 100% | 0.04 |
| 8 | 481,913 | 0.6% | 0.636 | 38 | 40 | 83 | 100% | 0.05 |
| 87 | 473,689 | 0.6% | 0.665 | 36 | 34 | 69 | 100% | 0.01 |
| 47 | 469,527 | 0.5% | 0.648 | 38 | 34 | 70 | 100% | 0.03 |
| 108 | 465,933 | 0.5% | 0.631 | 38 | 60 | 88 | 100% | 0.09 |
| 124 | 461,084 | 0.5% | 0.713 | 36 | 34 | 70 | 100% | 0.00 |
| 44 | 459,651 | 0.5% | 0.644 | 38 | 42 | 70 | 100% | 0.10 |
| 118 | 448,523 | 0.5% | 0.604 | 40 | 34 | 70 | 100% | 0.05 |
| 51 | 448,206 | 0.5% | 0.655 | 37 | 33 | 69 | 100% | 0.02 |
| 102 | 406,939 | 0.5% | 0.631 | 37 | 57 | 53 | 98% | 0.09 |
| 77 | 405,378 | 0.5% | 0.781 | 32 | 30 | 59 | 100% | 0.00 |
| 72 | 398,986 | 0.5% | 0.622 | 40 | 29 | 61 | 100% | 0.02 |
| 20 | 395,901 | 0.5% | 0.674 | 34 | 29 | 45 | 100% | 0.03 |
| 69 | 393,048 | 0.5% | 0.663 | 37 | 32 | 77 | 100% | 0.02 |
| 103 | 384,949 | 0.4% | 0.617 | 39 | 43 | 66 | 100% | 0.09 |
| 55 | 377,770 | 0.4% | 0.697 | 34 | 26 | 44 | 100% | 0.02 |
| 96 | 373,790 | 0.4% | 0.667 | 37 | 28 | 55 | 100% | 0.03 |
| 122 | 369,689 | 0.4% | 0.650 | 33 | 265 | 3 | 100% | 0.07 |
| 14 | 369,164 | 0.4% | 0.755 | 34 | 27 | 55 | 100% | 0.01 |
| 116 | 351,981 | 0.4% | 0.644 | 37 | 45 | 42 | 100% | 0.09 |
| 112 | 350,053 | 0.4% | 0.645 | 40 | 27 | 57 | 100% | 0.02 |
| 37 | 349,382 | 0.4% | 0.633 | 40 | 29 | 57 | 100% | 0.06 |
| 62 | 345,346 | 0.4% | 0.668 | 36 | 25 | 51 | 100% | 0.01 |
| 32 | 341,312 | 0.4% | 0.671 | 36 | 25 | 51 | 100% | 0.01 |
| 109 | 326,805 | 0.4% | 0.653 | 38 | 24 | 40 | 100% | 0.07 |
| 92 | 324,124 | 0.4% | 0.632 | 40 | 24 | 45 | 100% | 0.02 |
| 49 | 322,522 | 0.4% | 0.716 | 34 | 24 | 52 | 100% | 0.02 |
| 63 | 308,635 | 0.4% | 0.679 | 36 | 23 | 46 | 100% | 0.01 |
| 79 | 290,794 | 0.3% | 0.651 | 38 | 21 | 42 | 100% | 0.00 |
| 2 | 266,121 | 0.3% | 0.672 | 38 | 19 | 37 | 100% | 0.00 |
| 38 | 230,740 | 0.3% | 0.640 | 40 | 18 | 32 | 100% | 0.02 |
| 28 | 227,854 | 0.3% | 0.729 | 34 | 17 | 27 | 100% | 0.04 |
| 65 | 1 | 0.0% | 0.000 | 65 | 1 | 0 | 0% | 3.00 |

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/temporal_spatial.py --dir runs/clustering/v4_subspace_big_d64 --out runs/clustering/v4_subspace_big_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.416, mean 0.425, max 0.806 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 30 ↔ 126 | 0.806 | 0.447 |
| 85 ↔ 126 | 0.802 | 0.724 |
| 114 ↔ 117 | 0.799 | 0.718 |
| 120 ↔ 126 | 0.799 | 0.431 |
| 110 ↔ 117 | 0.791 | 0.699 |
| 10 ↔ 59 | 0.787 | 0.745 |
| 5 ↔ 33 | 0.785 | 0.312 |
| 68 ↔ 117 | 0.781 | 0.709 |
| 44 ↔ 126 | 0.781 | 0.430 |
| 26 ↔ 48 | 0.780 | 0.775 |
| 85 ↔ 114 | 0.775 | 0.717 |
| 58 ↔ 86 | 0.773 | 0.538 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
