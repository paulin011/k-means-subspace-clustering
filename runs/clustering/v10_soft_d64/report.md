# Clustering report (subspace_em) — `runs/clustering/v10_soft_d64`

*Generated 2026-08-04 13:31 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

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
  python3 src/clustering/subspace_em.py --files-from runs/clustering/v10_soft_d64/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v10_soft_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7659.20 | 100.00% | 127 | 23,702,502 |
| 2 | 2354.90 | 77.38% | 1,035 | 5,556,158 |
| 3 | 2077.21 | 42.73% | 8,246 | 2,352,340 |
| 4 | 1996.98 | 23.59% | 13,665 | 1,849,249 |
| 5 | 1964.63 | 15.09% | 13,860 | 1,716,431 |
| 6 | 1947.90 | 11.16% | 14,751 | 1,653,571 |
| 7 | 1937.13 | 8.93% | 19,291 | 1,605,100 |
| 8 | 1929.19 | 7.49% | 20,857 | 1,562,249 |
| 9 | 1922.83 | 6.41% | 20,862 | 1,534,766 |
| 10 | 1917.86 | 5.50% | 20,863 | 1,519,161 |
| 11 | 1914.11 | 4.88% | 20,863 | 1,506,119 |
| 12 | 1910.87 | 4.43% | 20,863 | 1,490,823 |
| 13 | 1907.97 | 4.03% | 20,863 | 1,480,928 |
| 14 | 1905.48 | 3.63% | 20,862 | 1,467,659 |
| 15 | 1903.27 | 3.27% | 20,862 | 1,458,825 |
| 16 | 1901.46 | 2.94% | 20,862 | 1,450,851 |
| 17 | 1900.00 | 2.69% | 20,862 | 1,444,818 |
| 18 | 1898.79 | 2.50% | 20,862 | 1,439,717 |
| 19 | 1897.67 | 2.35% | 20,862 | 1,436,672 |
| 20 | 1896.62 | 2.20% | 20,862 | 1,433,069 |
| 21 | 1895.68 | 2.05% | 20,862 | 1,430,020 |
| 22 | 1894.92 | 1.91% | 20,863 | 1,426,841 |
| 23 | 1894.23 | 1.81% | 20,863 | 1,425,250 |
| 24 | 1893.55 | 1.72% | 20,863 | 1,421,592 |
| 25 | 1892.96 | 1.61% | 20,863 | 1,419,097 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **6000**, split into:

- **8.6%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **59.8%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 6000, the same denominator as the other two, which is what lets all three add to 100%.
- **31.6%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.664** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 66%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 27 / median 36 / max 40** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=65 = a truncation warning. Your max is 40, below the cap → no cluster truncated, d=64 has headroom.

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

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.02%–1.65% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 1,418,880 | 1.65% | 0.733 | 31 | 102 | 203 | 49.8% | 0.000 | 0.560 |
| 70 | 1,233,329 | 1.43% | 0.683 | 33 | 185 | 269 | 42.0% | 0.036 | 0.749 |
| 27 | 1,210,821 | 1.41% | 0.727 | 32 | 87 | 176 | 49.2% | 0.003 | 0.632 |
| 69 | 1,059,821 | 1.23% | 0.656 | 36 | 120 | 241 | 38.3% | 0.040 | 0.785 |
| 114 | 1,023,940 | 1.19% | 0.635 | 39 | 84 | 155 | 42.4% | 0.076 | 0.791 |
| 4 | 1,023,217 | 1.19% | 0.656 | 35 | 173 | 249 | 33.9% | 0.047 | 0.755 |
| 17 | 1,018,157 | 1.18% | 0.744 | 30 | 74 | 148 | 48.8% | 0.004 | 0.716 |
| 93 | 1,016,627 | 1.18% | 0.761 | 29 | 80 | 178 | 46.2% | 0.013 | 0.707 |
| 76 | 982,538 | 1.14% | 0.667 | 33 | 149 | 155 | 37.4% | 0.038 | 0.722 |
| 60 | 953,775 | 1.11% | 0.632 | 37 | 203 | 122 | 23.5% | 0.076 | 0.778 |
| 99 | 953,636 | 1.11% | 0.624 | 38 | 119 | 174 | 41.7% | 0.062 | 0.773 |
| 34 | 952,461 | 1.11% | 0.631 | 35 | 159 | 65 | 18.9% | 0.126 | 0.702 |
| 21 | 940,383 | 1.09% | 0.708 | 37 | 68 | 140 | 48.8% | 0.005 | 0.551 |
| 50 | 920,624 | 1.07% | 0.767 | 30 | 67 | 138 | 48.5% | 0.004 | 0.671 |
| 85 | 916,840 | 1.07% | 0.634 | 39 | 74 | 149 | 40.4% | 0.079 | 0.783 |
| 121 | 899,358 | 1.05% | 0.699 | 31 | 142 | 182 | 30.2% | 0.087 | 0.722 |
| 0 | 892,436 | 1.04% | 0.725 | 33 | 64 | 130 | 49.3% | 0.002 | 0.603 |
| 5 | 871,180 | 1.01% | 0.638 | 37 | 253 | 70 | 39.8% | 0.048 | 0.782 |
| 6 | 871,105 | 1.01% | 0.678 | 33 | 201 | 113 | 34.7% | 0.053 | 0.738 |
| 48 | 869,873 | 1.01% | 0.603 | 38 | 166 | 126 | 31.9% | 0.128 | 0.749 |
| 68 | 864,728 | 1.01% | 0.638 | 38 | 138 | 142 | 26.4% | 0.073 | 0.793 |
| 1 | 854,126 | 0.99% | 0.612 | 38 | 94 | 135 | 39.1% | 0.108 | 0.751 |
| 33 | 838,247 | 0.97% | 0.648 | 36 | 239 | 107 | 35.4% | 0.073 | 0.782 |
| 107 | 838,016 | 0.97% | 0.662 | 33 | 312 | 21 | 43.0% | 0.016 | 0.696 |
| 84 | 831,163 | 0.97% | 0.602 | 38 | 245 | 34 | 33.2% | 0.099 | 0.761 |
| 82 | 828,244 | 0.96% | 0.635 | 37 | 173 | 151 | 19.6% | 0.141 | 0.745 |
| 22 | 825,516 | 0.96% | 0.602 | 38 | 140 | 118 | 40.0% | 0.067 | 0.742 |
| 43 | 821,545 | 0.96% | 0.695 | 29 | 190 | 93 | 25.4% | 0.112 | 0.642 |
| 26 | 814,125 | 0.95% | 0.603 | 39 | 167 | 120 | 31.7% | 0.070 | 0.770 |
| 54 | 813,253 | 0.95% | 0.598 | 39 | 244 | 61 | 32.0% | 0.045 | 0.751 |
| 94 | 811,335 | 0.94% | 0.703 | 30 | 181 | 24 | 40.4% | 0.027 | 0.685 |
| 18 | 811,088 | 0.94% | 0.673 | 32 | 171 | 139 | 42.1% | 0.035 | 0.708 |
| 88 | 807,360 | 0.94% | 0.656 | 34 | 126 | 90 | 29.4% | 0.085 | 0.761 |
| 59 | 804,516 | 0.94% | 0.695 | 35 | 81 | 172 | 44.5% | 0.019 | 0.775 |
| 95 | 804,388 | 0.94% | 0.650 | 36 | 197 | 113 | 25.7% | 0.093 | 0.798 |
| 58 | 802,851 | 0.93% | 0.652 | 36 | 143 | 99 | 25.3% | 0.107 | 0.783 |
| 105 | 802,184 | 0.93% | 0.591 | 39 | 82 | 114 | 31.4% | 0.061 | 0.679 |
| 61 | 798,763 | 0.93% | 0.648 | 36 | 61 | 126 | 45.1% | 0.024 | 0.693 |
| 86 | 797,515 | 0.93% | 0.643 | 37 | 86 | 144 | 38.6% | 0.043 | 0.783 |
| 42 | 796,533 | 0.93% | 0.667 | 36 | 85 | 151 | 40.7% | 0.042 | 0.798 |
| 113 | 794,246 | 0.92% | 0.684 | 29 | 216 | 41 | 38.3% | 0.038 | 0.681 |
| 111 | 780,943 | 0.91% | 0.744 | 29 | 69 | 134 | 43.5% | 0.011 | 0.679 |
| 119 | 775,782 | 0.90% | 0.754 | 31 | 59 | 121 | 47.8% | 0.007 | 0.656 |
| 73 | 774,394 | 0.90% | 0.612 | 38 | 210 | 101 | 19.0% | 0.126 | 0.751 |
| 83 | 766,092 | 0.89% | 0.748 | 30 | 78 | 153 | 41.4% | 0.040 | 0.707 |
| 74 | 762,474 | 0.89% | 0.734 | 35 | 55 | 117 | 48.0% | 0.008 | 0.610 |
| 125 | 754,644 | 0.88% | 0.716 | 34 | 61 | 139 | 40.7% | 0.032 | 0.681 |
| 31 | 753,947 | 0.88% | 0.615 | 37 | 184 | 60 | 25.2% | 0.074 | 0.748 |
| 10 | 752,172 | 0.87% | 0.691 | 36 | 65 | 143 | 44.5% | 0.030 | 0.775 |
| 41 | 735,728 | 0.86% | 0.663 | 33 | 451 | 7 | 35.5% | 0.044 | 0.696 |
| 56 | 731,656 | 0.85% | 0.730 | 34 | 54 | 117 | 46.0% | 0.013 | 0.681 |
| 67 | 730,502 | 0.85% | 0.638 | 38 | 82 | 145 | 39.2% | 0.045 | 0.778 |
| 100 | 723,834 | 0.84% | 0.603 | 38 | 158 | 86 | 22.0% | 0.176 | 0.752 |
| 75 | 719,023 | 0.84% | 0.612 | 39 | 96 | 143 | 24.4% | 0.126 | 0.773 |
| 39 | 716,241 | 0.83% | 0.715 | 34 | 61 | 147 | 45.6% | 0.018 | 0.703 |
| 127 | 715,309 | 0.83% | 0.782 | 34 | 52 | 104 | 49.3% | 0.002 | 0.580 |
| 40 | 710,517 | 0.83% | 0.603 | 39 | 156 | 104 | 30.8% | 0.074 | 0.770 |
| 52 | 710,460 | 0.83% | 0.668 | 33 | 181 | 88 | 22.8% | 0.131 | 0.705 |
| 115 | 708,052 | 0.82% | 0.646 | 36 | 61 | 132 | 39.6% | 0.050 | 0.712 |
| 53 | 700,807 | 0.81% | 0.625 | 38 | 64 | 96 | 32.5% | 0.054 | 0.739 |
| 89 | 691,919 | 0.80% | 0.656 | 36 | 50 | 105 | 47.0% | 0.013 | 0.625 |
| 13 | 684,593 | 0.80% | 0.612 | 37 | 592 | 0 | 38.7% | 0.068 | 0.726 |
| 36 | 682,392 | 0.79% | 0.640 | 34 | 373 | 5 | 37.7% | 0.038 | 0.722 |
| 71 | 679,089 | 0.79% | 0.629 | 38 | 97 | 126 | 27.6% | 0.119 | 0.755 |
| 117 | 678,255 | 0.79% | 0.660 | 38 | 60 | 124 | 40.2% | 0.067 | 0.793 |
| 45 | 674,877 | 0.78% | 0.619 | 38 | 137 | 99 | 35.5% | 0.069 | 0.770 |
| 7 | 674,523 | 0.78% | 0.587 | 39 | 82 | 104 | 26.6% | 0.108 | 0.703 |
| 123 | 668,723 | 0.78% | 0.772 | 31 | 48 | 98 | 48.8% | 0.005 | 0.716 |
| 101 | 666,715 | 0.78% | 0.626 | 37 | 140 | 109 | 18.2% | 0.214 | 0.777 |
| 78 | 663,933 | 0.77% | 0.780 | 31 | 49 | 100 | 48.3% | 0.005 | 0.657 |
| 98 | 662,371 | 0.77% | 0.707 | 27 | 416 | 4 | 31.3% | 0.049 | 0.669 |
| 11 | 661,731 | 0.77% | 0.642 | 35 | 97 | 102 | 25.8% | 0.076 | 0.679 |
| 126 | 661,664 | 0.77% | 0.620 | 38 | 74 | 134 | 30.5% | 0.266 | 0.817 |
| 55 | 660,696 | 0.77% | 0.718 | 29 | 165 | 41 | 32.6% | 0.073 | 0.672 |
| 104 | 660,591 | 0.77% | 0.618 | 38 | 119 | 87 | 21.8% | 0.136 | 0.738 |
| 80 | 660,365 | 0.77% | 0.745 | 35 | 48 | 100 | 47.7% | 0.006 | 0.659 |
| 19 | 647,137 | 0.75% | 0.648 | 37 | 61 | 109 | 36.9% | 0.066 | 0.752 |
| 16 | 638,865 | 0.74% | 0.670 | 35 | 47 | 103 | 43.3% | 0.021 | 0.630 |
| 29 | 635,562 | 0.74% | 0.615 | 38 | 108 | 99 | 19.8% | 0.132 | 0.711 |
| 90 | 635,095 | 0.74% | 0.722 | 34 | 47 | 103 | 45.5% | 0.020 | 0.615 |
| 63 | 631,250 | 0.73% | 0.624 | 35 | 114 | 60 | 38.3% | 0.031 | 0.679 |
| 30 | 623,091 | 0.72% | 0.636 | 38 | 136 | 30 | 19.7% | 0.130 | 0.802 |
| 24 | 622,258 | 0.72% | 0.612 | 37 | 106 | 83 | 17.8% | 0.139 | 0.686 |
| 23 | 617,904 | 0.72% | 0.605 | 39 | 60 | 71 | 32.1% | 0.075 | 0.705 |
| 46 | 613,937 | 0.71% | 0.652 | 38 | 47 | 78 | 39.9% | 0.070 | 0.743 |
| 44 | 607,730 | 0.71% | 0.643 | 38 | 77 | 135 | 27.3% | 0.092 | 0.817 |
| 35 | 591,551 | 0.69% | 0.745 | 34 | 44 | 92 | 47.3% | 0.012 | 0.626 |
| 14 | 580,934 | 0.68% | 0.751 | 34 | 42 | 88 | 47.7% | 0.008 | 0.659 |
| 9 | 580,015 | 0.67% | 0.635 | 38 | 87 | 125 | 25.7% | 0.098 | 0.770 |
| 120 | 574,382 | 0.67% | 0.652 | 38 | 49 | 102 | 35.9% | 0.080 | 0.804 |
| 87 | 572,090 | 0.67% | 0.664 | 35 | 42 | 90 | 45.7% | 0.016 | 0.679 |
| 97 | 568,194 | 0.66% | 0.627 | 38 | 55 | 105 | 36.4% | 0.057 | 0.729 |
| 57 | 561,663 | 0.65% | 0.615 | 38 | 310 | 1 | 25.5% | 0.113 | 0.748 |
| 64 | 561,433 | 0.65% | 0.605 | 38 | 86 | 45 | 22.7% | 0.130 | 0.720 |
| 15 | 560,117 | 0.65% | 0.603 | 40 | 44 | 92 | 40.4% | 0.018 | 0.703 |
| 118 | 559,080 | 0.65% | 0.597 | 40 | 43 | 86 | 44.8% | 0.058 | 0.709 |
| 106 | 557,753 | 0.65% | 0.613 | 39 | 115 | 45 | 31.5% | 0.080 | 0.755 |
| 3 | 551,712 | 0.64% | 0.652 | 38 | 42 | 80 | 43.5% | 0.035 | 0.747 |
| 25 | 540,541 | 0.63% | 0.720 | 34 | 39 | 82 | 47.8% | 0.008 | 0.592 |
| 102 | 529,619 | 0.62% | 0.629 | 36 | 71 | 66 | 20.8% | 0.098 | 0.704 |
| 103 | 519,116 | 0.60% | 0.620 | 39 | 73 | 104 | 22.8% | 0.127 | 0.750 |
| 65 | 517,042 | 0.60% | 0.605 | 40 | 42 | 69 | 31.3% | 0.068 | 0.672 |
| 116 | 500,465 | 0.58% | 0.626 | 37 | 52 | 70 | 30.9% | 0.067 | 0.663 |
| 81 | 495,013 | 0.58% | 0.659 | 36 | 37 | 76 | 46.7% | 0.009 | 0.665 |
| 47 | 493,088 | 0.57% | 0.656 | 38 | 37 | 81 | 45.0% | 0.029 | 0.739 |
| 108 | 486,260 | 0.57% | 0.625 | 38 | 60 | 93 | 27.9% | 0.083 | 0.748 |
| 122 | 483,458 | 0.56% | 0.641 | 33 | 301 | 4 | 27.6% | 0.064 | 0.673 |
| 12 | 478,396 | 0.56% | 0.629 | 39 | 45 | 74 | 31.3% | 0.075 | 0.742 |
| 51 | 474,972 | 0.55% | 0.656 | 37 | 36 | 74 | 44.6% | 0.024 | 0.664 |
| 20 | 467,281 | 0.54% | 0.670 | 35 | 34 | 69 | 47.0% | 0.012 | 0.679 |
| 91 | 451,093 | 0.52% | 0.651 | 37 | 35 | 60 | 39.5% | 0.056 | 0.680 |
| 124 | 448,346 | 0.52% | 0.674 | 36 | 35 | 74 | 45.4% | 0.017 | 0.670 |
| 110 | 366,604 | 0.43% | 0.715 | 38 | 28 | 61 | 45.9% | 0.016 | 0.756 |
| 72 | 361,361 | 0.42% | 0.630 | 40 | 31 | 58 | 33.6% | 0.064 | 0.665 |
| 37 | 349,872 | 0.41% | 0.637 | 39 | 29 | 60 | 38.6% | 0.067 | 0.744 |
| 96 | 343,708 | 0.40% | 0.682 | 37 | 26 | 58 | 41.8% | 0.033 | 0.676 |
| 62 | 332,888 | 0.39% | 0.673 | 36 | 25 | 51 | 45.2% | 0.019 | 0.632 |
| 2 | 311,634 | 0.36% | 0.673 | 37 | 23 | 45 | 49.4% | 0.002 | 0.514 |
| 109 | 300,835 | 0.35% | 0.654 | 37 | 48 | 33 | 18.0% | 0.132 | 0.665 |
| 77 | 299,052 | 0.35% | 0.793 | 32 | 22 | 43 | 49.5% | 0.002 | 0.599 |
| 112 | 290,903 | 0.34% | 0.658 | 40 | 22 | 47 | 41.6% | 0.024 | 0.724 |
| 38 | 271,594 | 0.32% | 0.641 | 40 | 21 | 39 | 43.7% | 0.025 | 0.700 |
| 79 | 256,832 | 0.30% | 0.664 | 38 | 19 | 34 | 43.5% | 0.026 | 0.614 |
| 49 | 197,551 | 0.23% | 0.775 | 33 | 16 | 37 | 38.1% | 0.053 | 0.597 |
| 28 | 195,004 | 0.23% | 0.754 | 33 | 14 | 19 | 32.1% | 0.055 | 0.597 |
| 32 | 145,351 | 0.17% | 0.729 | 35 | 11 | 21 | 49.4% | 0.002 | 0.614 |
| 8 | 51,745 | 0.06% | 0.718 | 34 | 4 | 8 | 46.1% | 0.020 | 0.742 |
| 92 | 20,866 | 0.02% | 0.750 | 31 | 2 | 3 | 49.7% | 0.004 | 0.695 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **13** under a quarter of the mean cluster size: **8, 32, 92**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v10_soft_d64 --out runs/clustering/v10_soft_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.408, mean 0.425, max 0.817 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 44 ↔ 126 | 0.817 | 0.640 |
| 44 ↔ 120 | 0.804 | 0.549 |
| 30 ↔ 126 | 0.802 | 0.476 |
| 42 ↔ 95 | 0.798 | 0.455 |
| 68 ↔ 117 | 0.793 | 0.730 |
| 114 ↔ 117 | 0.791 | 0.713 |
| 30 ↔ 44 | 0.786 | 0.303 |
| 69 ↔ 117 | 0.785 | 0.775 |
| 58 ↔ 86 | 0.783 | 0.533 |
| 85 ↔ 114 | 0.783 | 0.739 |
| 5 ↔ 33 | 0.782 | 0.418 |
| 68 ↔ 114 | 0.782 | 0.607 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
