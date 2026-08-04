# Clustering report (subspace_kmeans) — `runs/clustering/v4_subspace_big_d64`

*Generated 2026-08-04 13:33 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

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
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v4_subspace_big_d64/sample.json --seed 0 --tokens-per-file 12288 \
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
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`maxAff` = this cluster's subspace affinity to its **nearest** neighbour, `maxⱼ≠ᵢ ‖UᵢᵀUⱼ‖²_F / d` ∈ [0,1]. A per-cluster separation score: **high ⇒ some other cluster spans nearly the same directions**, so this row is a merge candidate. The affinity table below lists only the top pairs, so a near-duplicate cluster is invisible there unless its pair happens to rank; this column always shows it.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.00%–1.72% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 27 | 1,478,198 | 1.72% | 0.713 | 32 | 106 | 212 | 49.8% | 0.001 | 0.650 |
| 66 | 1,460,322 | 1.70% | 0.728 | 31 | 105 | 209 | 49.9% | 0.000 | 0.584 |
| 50 | 1,355,978 | 1.58% | 0.750 | 29 | 99 | 208 | 47.9% | 0.008 | 0.718 |
| 78 | 1,201,732 | 1.40% | 0.728 | 29 | 169 | 153 | 40.3% | 0.055 | 0.709 |
| 74 | 1,171,060 | 1.36% | 0.703 | 34 | 84 | 170 | 48.7% | 0.004 | 0.675 |
| 114 | 1,139,747 | 1.33% | 0.632 | 39 | 90 | 175 | 43.7% | 0.059 | 0.799 |
| 42 | 1,108,410 | 1.29% | 0.655 | 36 | 119 | 182 | 40.1% | 0.035 | 0.762 |
| 6 | 1,094,357 | 1.27% | 0.666 | 34 | 188 | 217 | 36.8% | 0.056 | 0.758 |
| 17 | 1,079,263 | 1.25% | 0.745 | 30 | 78 | 154 | 49.4% | 0.002 | 0.718 |
| 39 | 1,044,110 | 1.21% | 0.680 | 33 | 174 | 177 | 42.3% | 0.040 | 0.758 |
| 119 | 1,031,287 | 1.20% | 0.696 | 32 | 141 | 190 | 41.6% | 0.033 | 0.758 |
| 117 | 1,025,402 | 1.19% | 0.645 | 38 | 105 | 212 | 42.2% | 0.061 | 0.799 |
| 4 | 1,023,376 | 1.19% | 0.650 | 37 | 189 | 180 | 33.6% | 0.051 | 0.756 |
| 85 | 1,001,659 | 1.16% | 0.634 | 39 | 79 | 157 | 42.4% | 0.067 | 0.802 |
| 125 | 1,000,632 | 1.16% | 0.698 | 33 | 75 | 167 | 45.8% | 0.017 | 0.695 |
| 35 | 981,117 | 1.14% | 0.714 | 34 | 71 | 145 | 48.9% | 0.005 | 0.643 |
| 93 | 941,493 | 1.09% | 0.765 | 30 | 68 | 136 | 49.3% | 0.002 | 0.667 |
| 86 | 941,407 | 1.09% | 0.635 | 38 | 105 | 165 | 37.1% | 0.053 | 0.773 |
| 83 | 932,810 | 1.08% | 0.739 | 31 | 71 | 151 | 46.6% | 0.011 | 0.675 |
| 70 | 919,353 | 1.07% | 0.686 | 34 | 106 | 212 | 40.1% | 0.029 | 0.758 |
| 56 | 907,862 | 1.06% | 0.718 | 34 | 65 | 132 | 49.1% | 0.003 | 0.685 |
| 99 | 899,834 | 1.05% | 0.627 | 38 | 105 | 180 | 36.2% | 0.049 | 0.773 |
| 0 | 899,110 | 1.05% | 0.723 | 33 | 64 | 129 | 49.4% | 0.002 | 0.639 |
| 61 | 898,252 | 1.04% | 0.644 | 37 | 68 | 147 | 44.4% | 0.025 | 0.698 |
| 88 | 883,911 | 1.03% | 0.663 | 34 | 146 | 139 | 24.2% | 0.103 | 0.760 |
| 126 | 864,804 | 1.01% | 0.631 | 38 | 109 | 189 | 25.8% | 0.160 | 0.806 |
| 43 | 847,542 | 0.99% | 0.690 | 30 | 191 | 122 | 24.5% | 0.115 | 0.650 |
| 21 | 845,765 | 0.98% | 0.715 | 37 | 61 | 121 | 49.8% | 0.002 | 0.557 |
| 82 | 842,453 | 0.98% | 0.632 | 37 | 177 | 142 | 19.5% | 0.138 | 0.750 |
| 67 | 835,690 | 0.97% | 0.625 | 38 | 169 | 171 | 19.6% | 0.199 | 0.757 |
| 58 | 834,029 | 0.97% | 0.642 | 37 | 132 | 116 | 28.6% | 0.076 | 0.773 |
| 121 | 825,209 | 0.96% | 0.687 | 32 | 90 | 142 | 30.4% | 0.072 | 0.695 |
| 10 | 821,282 | 0.95% | 0.692 | 35 | 68 | 148 | 45.3% | 0.023 | 0.787 |
| 76 | 816,959 | 0.95% | 0.668 | 33 | 108 | 121 | 36.3% | 0.049 | 0.697 |
| 105 | 815,422 | 0.95% | 0.590 | 39 | 76 | 125 | 32.4% | 0.051 | 0.702 |
| 34 | 811,525 | 0.94% | 0.634 | 36 | 89 | 68 | 23.2% | 0.102 | 0.692 |
| 75 | 805,992 | 0.94% | 0.611 | 39 | 102 | 180 | 29.7% | 0.092 | 0.773 |
| 60 | 805,468 | 0.94% | 0.628 | 37 | 169 | 116 | 19.9% | 0.103 | 0.762 |
| 89 | 803,782 | 0.93% | 0.651 | 37 | 58 | 116 | 49.0% | 0.003 | 0.644 |
| 33 | 800,505 | 0.93% | 0.647 | 37 | 176 | 130 | 37.9% | 0.078 | 0.785 |
| 111 | 788,869 | 0.92% | 0.718 | 29 | 252 | 21 | 41.9% | 0.020 | 0.708 |
| 107 | 786,495 | 0.91% | 0.706 | 33 | 66 | 139 | 45.9% | 0.012 | 0.727 |
| 1 | 780,614 | 0.91% | 0.612 | 38 | 82 | 126 | 38.4% | 0.122 | 0.757 |
| 41 | 778,361 | 0.90% | 0.664 | 33 | 349 | 6 | 30.4% | 0.059 | 0.720 |
| 95 | 772,026 | 0.90% | 0.659 | 36 | 146 | 128 | 23.9% | 0.104 | 0.759 |
| 113 | 762,784 | 0.89% | 0.700 | 28 | 145 | 96 | 36.9% | 0.054 | 0.655 |
| 25 | 758,333 | 0.88% | 0.702 | 35 | 55 | 110 | 49.1% | 0.004 | 0.619 |
| 127 | 741,281 | 0.86% | 0.774 | 33 | 53 | 108 | 49.5% | 0.001 | 0.586 |
| 115 | 728,943 | 0.85% | 0.644 | 36 | 57 | 125 | 44.0% | 0.027 | 0.729 |
| 90 | 726,782 | 0.84% | 0.703 | 35 | 52 | 117 | 46.3% | 0.013 | 0.634 |
| 59 | 723,254 | 0.84% | 0.695 | 35 | 60 | 125 | 44.4% | 0.013 | 0.787 |
| 18 | 721,229 | 0.84% | 0.665 | 32 | 222 | 42 | 41.4% | 0.027 | 0.711 |
| 48 | 713,321 | 0.83% | 0.601 | 39 | 158 | 83 | 31.8% | 0.144 | 0.780 |
| 110 | 710,479 | 0.83% | 0.662 | 38 | 57 | 133 | 44.1% | 0.024 | 0.791 |
| 5 | 701,999 | 0.82% | 0.640 | 37 | 263 | 19 | 38.5% | 0.051 | 0.785 |
| 80 | 700,085 | 0.81% | 0.732 | 35 | 51 | 101 | 49.5% | 0.002 | 0.665 |
| 73 | 698,614 | 0.81% | 0.610 | 38 | 195 | 71 | 20.0% | 0.128 | 0.765 |
| 52 | 697,128 | 0.81% | 0.657 | 34 | 163 | 89 | 23.2% | 0.126 | 0.708 |
| 98 | 692,746 | 0.81% | 0.674 | 31 | 309 | 0 | 37.6% | 0.042 | 0.711 |
| 16 | 684,538 | 0.80% | 0.666 | 35 | 50 | 100 | 47.9% | 0.008 | 0.644 |
| 26 | 682,347 | 0.79% | 0.605 | 40 | 145 | 84 | 29.5% | 0.080 | 0.780 |
| 71 | 672,429 | 0.78% | 0.626 | 38 | 111 | 130 | 24.5% | 0.124 | 0.755 |
| 22 | 672,176 | 0.78% | 0.606 | 39 | 105 | 98 | 37.4% | 0.067 | 0.751 |
| 123 | 669,338 | 0.78% | 0.772 | 31 | 48 | 96 | 49.7% | 0.001 | 0.697 |
| 9 | 668,359 | 0.78% | 0.633 | 38 | 111 | 145 | 27.9% | 0.112 | 0.759 |
| 19 | 661,401 | 0.77% | 0.647 | 37 | 62 | 112 | 39.0% | 0.053 | 0.752 |
| 46 | 660,932 | 0.77% | 0.651 | 38 | 50 | 91 | 37.9% | 0.077 | 0.764 |
| 30 | 660,465 | 0.77% | 0.635 | 38 | 140 | 27 | 19.9% | 0.126 | 0.806 |
| 53 | 659,102 | 0.77% | 0.624 | 38 | 60 | 90 | 32.7% | 0.050 | 0.745 |
| 11 | 650,914 | 0.76% | 0.641 | 35 | 89 | 82 | 26.7% | 0.076 | 0.686 |
| 84 | 626,275 | 0.73% | 0.617 | 38 | 97 | 92 | 38.6% | 0.082 | 0.762 |
| 68 | 610,561 | 0.71% | 0.665 | 37 | 48 | 93 | 33.6% | 0.052 | 0.781 |
| 54 | 610,464 | 0.71% | 0.604 | 39 | 199 | 47 | 31.7% | 0.057 | 0.765 |
| 106 | 593,973 | 0.69% | 0.617 | 39 | 163 | 38 | 27.1% | 0.076 | 0.767 |
| 100 | 582,939 | 0.68% | 0.609 | 38 | 131 | 73 | 20.3% | 0.184 | 0.743 |
| 15 | 582,024 | 0.68% | 0.601 | 39 | 46 | 92 | 41.0% | 0.035 | 0.693 |
| 24 | 579,339 | 0.67% | 0.613 | 37 | 101 | 77 | 17.4% | 0.139 | 0.694 |
| 64 | 574,400 | 0.67% | 0.592 | 38 | 81 | 51 | 21.7% | 0.141 | 0.702 |
| 29 | 573,945 | 0.67% | 0.623 | 38 | 90 | 87 | 20.0% | 0.128 | 0.710 |
| 104 | 572,306 | 0.67% | 0.616 | 38 | 107 | 89 | 21.5% | 0.136 | 0.730 |
| 120 | 566,048 | 0.66% | 0.649 | 39 | 45 | 92 | 39.9% | 0.057 | 0.799 |
| 12 | 565,276 | 0.66% | 0.638 | 38 | 69 | 78 | 29.1% | 0.072 | 0.764 |
| 45 | 558,198 | 0.65% | 0.624 | 38 | 103 | 96 | 34.0% | 0.074 | 0.757 |
| 7 | 556,068 | 0.65% | 0.610 | 38 | 156 | 18 | 16.5% | 0.122 | 0.755 |
| 94 | 543,046 | 0.63% | 0.717 | 29 | 155 | 27 | 32.8% | 0.053 | 0.676 |
| 101 | 541,428 | 0.63% | 0.619 | 38 | 147 | 84 | 25.3% | 0.085 | 0.767 |
| 31 | 524,425 | 0.61% | 0.615 | 38 | 149 | 33 | 32.3% | 0.085 | 0.738 |
| 23 | 517,622 | 0.60% | 0.611 | 40 | 53 | 56 | 31.2% | 0.080 | 0.708 |
| 13 | 516,644 | 0.60% | 0.621 | 36 | 537 | 0 | 39.2% | 0.072 | 0.717 |
| 91 | 512,341 | 0.60% | 0.642 | 37 | 39 | 75 | 42.8% | 0.029 | 0.691 |
| 40 | 500,456 | 0.58% | 0.607 | 39 | 149 | 45 | 33.2% | 0.066 | 0.751 |
| 57 | 496,922 | 0.58% | 0.624 | 38 | 180 | 25 | 24.9% | 0.108 | 0.738 |
| 81 | 496,427 | 0.58% | 0.660 | 36 | 36 | 76 | 47.1% | 0.009 | 0.675 |
| 3 | 495,026 | 0.58% | 0.649 | 38 | 37 | 74 | 44.3% | 0.022 | 0.742 |
| 36 | 494,835 | 0.58% | 0.632 | 36 | 273 | 0 | 31.3% | 0.063 | 0.726 |
| 97 | 482,749 | 0.56% | 0.626 | 39 | 44 | 84 | 39.5% | 0.036 | 0.725 |
| 8 | 481,913 | 0.56% | 0.636 | 38 | 40 | 83 | 39.3% | 0.049 | 0.725 |
| 87 | 473,689 | 0.55% | 0.665 | 36 | 34 | 69 | 45.3% | 0.013 | 0.690 |
| 47 | 469,527 | 0.55% | 0.648 | 38 | 34 | 70 | 42.6% | 0.033 | 0.745 |
| 108 | 465,933 | 0.54% | 0.631 | 38 | 60 | 88 | 26.0% | 0.095 | 0.757 |
| 124 | 461,084 | 0.54% | 0.713 | 36 | 34 | 70 | 48.4% | 0.005 | 0.664 |
| 44 | 459,651 | 0.53% | 0.644 | 38 | 42 | 70 | 35.6% | 0.100 | 0.781 |
| 118 | 448,523 | 0.52% | 0.604 | 40 | 34 | 70 | 45.1% | 0.055 | 0.705 |
| 51 | 448,206 | 0.52% | 0.655 | 37 | 33 | 69 | 45.8% | 0.020 | 0.675 |
| 102 | 406,939 | 0.47% | 0.631 | 37 | 57 | 53 | 20.8% | 0.088 | 0.744 |
| 77 | 405,378 | 0.47% | 0.781 | 32 | 30 | 59 | 49.1% | 0.002 | 0.564 |
| 72 | 398,986 | 0.46% | 0.622 | 40 | 29 | 61 | 44.0% | 0.022 | 0.665 |
| 20 | 395,901 | 0.46% | 0.674 | 34 | 29 | 45 | 40.5% | 0.029 | 0.679 |
| 69 | 393,048 | 0.46% | 0.663 | 37 | 32 | 77 | 43.2% | 0.022 | 0.737 |
| 103 | 384,949 | 0.45% | 0.617 | 39 | 43 | 66 | 28.5% | 0.092 | 0.694 |
| 55 | 377,770 | 0.44% | 0.697 | 34 | 26 | 44 | 42.4% | 0.021 | 0.612 |
| 96 | 373,790 | 0.43% | 0.667 | 37 | 28 | 55 | 42.1% | 0.033 | 0.676 |
| 122 | 369,689 | 0.43% | 0.650 | 33 | 265 | 3 | 23.8% | 0.067 | 0.669 |
| 14 | 369,164 | 0.43% | 0.755 | 34 | 27 | 55 | 48.3% | 0.007 | 0.665 |
| 116 | 351,981 | 0.41% | 0.644 | 37 | 45 | 42 | 26.5% | 0.089 | 0.662 |
| 112 | 350,053 | 0.41% | 0.645 | 40 | 27 | 57 | 42.6% | 0.020 | 0.702 |
| 37 | 349,382 | 0.41% | 0.633 | 40 | 29 | 57 | 38.0% | 0.063 | 0.712 |
| 62 | 345,346 | 0.40% | 0.668 | 36 | 25 | 51 | 46.5% | 0.009 | 0.666 |
| 32 | 341,312 | 0.40% | 0.671 | 36 | 25 | 51 | 46.0% | 0.011 | 0.609 |
| 109 | 326,805 | 0.38% | 0.653 | 38 | 24 | 40 | 26.8% | 0.067 | 0.644 |
| 92 | 324,124 | 0.38% | 0.632 | 40 | 24 | 45 | 44.2% | 0.023 | 0.627 |
| 49 | 322,522 | 0.37% | 0.716 | 34 | 24 | 52 | 44.2% | 0.023 | 0.595 |
| 63 | 308,635 | 0.36% | 0.679 | 36 | 23 | 46 | 45.1% | 0.015 | 0.644 |
| 79 | 290,794 | 0.34% | 0.651 | 38 | 21 | 42 | 48.9% | 0.005 | 0.605 |
| 2 | 266,121 | 0.31% | 0.672 | 38 | 19 | 37 | 48.5% | 0.003 | 0.526 |
| 38 | 230,740 | 0.27% | 0.640 | 40 | 18 | 32 | 44.5% | 0.024 | 0.688 |
| 28 | 227,854 | 0.26% | 0.729 | 34 | 17 | 27 | 39.0% | 0.036 | 0.595 |
| 65 | 1 | 0.00% | 0.000 | 65 | 1 | 0 | 0.0% | 3.000 | 0.000 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **13, 36, 65, 98** under a quarter of the mean cluster size: **65**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v4_subspace_big_d64 --out runs/clustering/v4_subspace_big_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

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
