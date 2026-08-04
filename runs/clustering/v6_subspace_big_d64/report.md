# Clustering report (subspace_kmeans) — `runs/clustering/v6_subspace_big_d64`

*Generated 2026-08-04 13:41 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

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
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v6_subspace_big_d64/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v6_subspace_big_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7659.20 | 100.00% | 127 | 23,702,502 |
| 2 | 2341.22 | 78.79% | 10,159 | 3,926,862 |
| 3 | 2060.47 | 41.53% | 61,340 | 2,322,919 |
| 4 | 1985.88 | 22.34% | 126,507 | 1,931,178 |
| 5 | 1956.09 | 14.48% | 167,980 | 1,857,619 |
| 6 | 1940.02 | 10.55% | 183,034 | 1,826,365 |
| 7 | 1929.77 | 8.33% | 195,648 | 1,805,857 |
| 8 | 1922.47 | 6.90% | 197,386 | 1,786,274 |
| 9 | 1917.02 | 5.81% | 208,298 | 1,774,344 |
| 10 | 1912.66 | 5.03% | 226,272 | 1,757,146 |
| 11 | 1908.95 | 4.41% | 228,819 | 1,746,603 |
| 12 | 1905.90 | 3.91% | 227,578 | 1,741,665 |
| 13 | 1903.30 | 3.51% | 225,997 | 1,734,123 |
| 14 | 1901.06 | 3.19% | 226,654 | 1,724,183 |
| 15 | 1899.06 | 2.94% | 234,905 | 1,707,514 |
| 16 | 1897.28 | 2.69% | 238,568 | 1,683,719 |
| 17 | 1895.79 | 2.45% | 236,630 | 1,655,101 |
| 18 | 1894.49 | 2.24% | 235,052 | 1,630,741 |
| 19 | 1893.40 | 2.08% | 233,230 | 1,602,085 |
| 20 | 1892.44 | 1.96% | 231,244 | 1,564,526 |
| 21 | 1891.56 | 1.84% | 229,662 | 1,537,811 |
| 22 | 1890.72 | 1.75% | 228,719 | 1,506,551 |
| 23 | 1889.88 | 1.68% | 229,007 | 1,478,192 |
| 24 | 1889.14 | 1.60% | 228,983 | 1,478,190 |
| 25 | 1888.51 | 1.55% | 227,886 | 1,478,211 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5999**, split into:

- **8.4%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **60.1%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5999, the same denominator as the other two, which is what lets all three add to 100%.
- **31.5%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.664** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 66%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 28 / median 37 / max 40** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=65 = a truncation warning. Your max is 40, below the cap → no cluster truncated, d=64 has headroom.

## Held-out generalization

*How to read this: the variance split above is measured on the very tokens the model was fit on, so a flexible model can look good just by memorising them. `holdout_eval.py` freezes the trained `means`/`U` and replays the assignment rule on fresh tokens from latent files the run never touched. If the **held-out residual** matches the **in-sample residual**, the subspaces capture reusable structure; a much larger held-out residual (gap > 3%) means the bases were fitting noise.*

The variance split above is measured on the tokens the model was *fit* on. `holdout_eval.py` froze the trained means and bases and replayed the assignment rule on **2,457,600 tokens from 200 latent files the run never saw**. A held-out residual close to the in-sample residual means the subspaces capture reusable structure rather than memorising the training tokens.

| residual (unexplained variance) | fraction |
|---|---|
| in-sample | 31.5% |
| held-out | 31.5% |
| generalization gap | +0.0% |

Objective/token (the minimised orthogonal residual): in-sample **1888.51** vs held-out **1887.16**. Verdict: the subspaces **generalise**; the in-sample residual is trustworthy.

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
- *`margin` = mean relative assignment margin `(R₂−R₁)/R₁` over the tokens this cluster owns — how much worse the **second-best** subspace is. `near%` = share of those tokens with margin < 10% (**near-ties**: assigned to this cluster, but barely). **This is the subspace-native separation metric**; the usual silhouette coefficient does not apply here because it assumes Euclidean distance to a centroid and spherical clusters. It **correlates strongly with `maxAff`** (v6: Pearson +0.77) but is not redundant with it: `maxAff` sees only the angle between two subspaces, blind to where their means sit and where the data is dense, so it explains only ~59% of `near%`'s variance and names a different closest rival for half the clusters. Measured over the full dataset by `file_signature.py` (`runs/signatures/v6_d64_margin/cluster_margin.csv`).*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.26%–1.72% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff | margin | near% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 27 | 1,478,280 | 1.72% | 0.713 | 32 | 106 | 212 | 49.8% | 0.001 | 0.650 | 0.519 | 1.3% |
| 66 | 1,460,374 | 1.70% | 0.728 | 31 | 105 | 209 | 49.9% | 0.000 | 0.583 | 0.835 | 0.9% |
| 50 | 1,357,175 | 1.58% | 0.750 | 29 | 99 | 208 | 47.9% | 0.008 | 0.718 | 0.310 | 8.8% |
| 78 | 1,189,324 | 1.38% | 0.728 | 29 | 169 | 153 | 40.3% | 0.055 | 0.708 | 0.235 | 28.6% |
| 74 | 1,166,246 | 1.36% | 0.703 | 34 | 84 | 170 | 48.7% | 0.004 | 0.675 | 0.353 | 7.1% |
| 114 | 1,126,020 | 1.31% | 0.634 | 39 | 88 | 171 | 44.5% | 0.052 | 0.787 | 0.136 | 41.3% |
| 42 | 1,101,820 | 1.28% | 0.655 | 36 | 119 | 182 | 40.1% | 0.035 | 0.762 | 0.121 | 49.2% |
| 17 | 1,078,211 | 1.25% | 0.745 | 30 | 78 | 154 | 49.4% | 0.002 | 0.718 | 0.328 | 4.4% |
| 6 | 1,062,982 | 1.24% | 0.667 | 34 | 187 | 215 | 36.2% | 0.058 | 0.757 | 0.142 | 44.4% |
| 117 | 1,041,406 | 1.21% | 0.645 | 38 | 109 | 211 | 42.8% | 0.055 | 0.789 | 0.110 | 53.6% |
| 119 | 1,037,322 | 1.21% | 0.696 | 32 | 143 | 192 | 41.5% | 0.033 | 0.758 | 0.171 | 36.2% |
| 39 | 1,010,078 | 1.17% | 0.680 | 33 | 176 | 176 | 41.9% | 0.041 | 0.757 | 0.135 | 47.8% |
| 125 | 987,812 | 1.15% | 0.698 | 33 | 75 | 167 | 45.8% | 0.017 | 0.695 | 0.229 | 22.1% |
| 35 | 980,507 | 1.14% | 0.714 | 34 | 71 | 145 | 48.9% | 0.005 | 0.643 | 0.451 | 5.2% |
| 4 | 969,736 | 1.13% | 0.651 | 37 | 172 | 171 | 32.8% | 0.053 | 0.753 | 0.110 | 55.1% |
| 85 | 965,750 | 1.12% | 0.634 | 39 | 76 | 150 | 42.3% | 0.071 | 0.801 | 0.139 | 38.4% |
| 93 | 941,388 | 1.09% | 0.765 | 30 | 68 | 136 | 49.3% | 0.002 | 0.667 | 0.517 | 3.5% |
| 86 | 933,673 | 1.09% | 0.635 | 38 | 105 | 166 | 37.1% | 0.053 | 0.773 | 0.117 | 52.5% |
| 83 | 919,895 | 1.07% | 0.739 | 31 | 70 | 149 | 46.5% | 0.012 | 0.674 | 0.283 | 21.3% |
| 70 | 913,540 | 1.06% | 0.686 | 34 | 105 | 215 | 40.0% | 0.029 | 0.758 | 0.149 | 39.4% |
| 56 | 906,415 | 1.05% | 0.718 | 34 | 65 | 132 | 49.1% | 0.003 | 0.685 | 0.413 | 3.9% |
| 61 | 897,718 | 1.04% | 0.644 | 37 | 68 | 147 | 44.4% | 0.025 | 0.698 | 0.220 | 22.6% |
| 88 | 896,537 | 1.04% | 0.663 | 34 | 146 | 138 | 24.2% | 0.103 | 0.759 | 0.136 | 45.1% |
| 0 | 893,306 | 1.04% | 0.724 | 33 | 64 | 129 | 49.4% | 0.002 | 0.639 | 0.472 | 3.6% |
| 10 | 877,976 | 1.02% | 0.690 | 35 | 75 | 159 | 44.6% | 0.023 | 0.777 | 0.198 | 25.6% |
| 43 | 857,982 | 1.00% | 0.690 | 30 | 191 | 120 | 24.5% | 0.115 | 0.650 | 0.225 | 32.9% |
| 126 | 854,044 | 0.99% | 0.631 | 38 | 108 | 187 | 25.7% | 0.160 | 0.805 | 0.130 | 43.5% |
| 58 | 846,578 | 0.98% | 0.642 | 37 | 133 | 115 | 28.6% | 0.076 | 0.773 | 0.101 | 60.3% |
| 21 | 845,800 | 0.98% | 0.715 | 37 | 61 | 121 | 49.8% | 0.002 | 0.557 | 0.633 | 0.9% |
| 67 | 844,028 | 0.98% | 0.625 | 38 | 169 | 171 | 19.6% | 0.198 | 0.757 | 0.118 | 51.6% |
| 82 | 841,801 | 0.98% | 0.632 | 37 | 176 | 143 | 19.5% | 0.138 | 0.751 | 0.127 | 47.5% |
| 65 | 821,228 | 0.95% | 0.629 | 38 | 109 | 179 | 29.1% | 0.045 | 0.768 | 0.108 | 55.2% |
| 121 | 815,755 | 0.95% | 0.687 | 32 | 90 | 141 | 30.4% | 0.072 | 0.695 | 0.211 | 34.9% |
| 105 | 815,500 | 0.95% | 0.590 | 39 | 76 | 126 | 32.5% | 0.051 | 0.702 | 0.218 | 20.8% |
| 76 | 815,450 | 0.95% | 0.668 | 33 | 108 | 120 | 36.3% | 0.049 | 0.697 | 0.136 | 48.6% |
| 89 | 802,706 | 0.93% | 0.651 | 37 | 58 | 116 | 49.0% | 0.003 | 0.644 | 0.356 | 7.0% |
| 34 | 799,867 | 0.93% | 0.634 | 36 | 88 | 70 | 23.2% | 0.102 | 0.692 | 0.168 | 37.0% |
| 33 | 794,909 | 0.92% | 0.647 | 37 | 178 | 126 | 37.7% | 0.075 | 0.782 | 0.096 | 62.6% |
| 41 | 784,567 | 0.91% | 0.664 | 33 | 361 | 7 | 30.3% | 0.059 | 0.721 | 0.145 | 47.5% |
| 95 | 782,218 | 0.91% | 0.659 | 36 | 146 | 130 | 23.9% | 0.104 | 0.759 | 0.128 | 50.1% |
| 111 | 778,978 | 0.91% | 0.718 | 29 | 249 | 19 | 42.0% | 0.021 | 0.707 | 0.191 | 39.2% |
| 113 | 773,796 | 0.90% | 0.700 | 28 | 145 | 96 | 36.9% | 0.054 | 0.655 | 0.192 | 39.5% |
| 127 | 764,811 | 0.89% | 0.770 | 33 | 55 | 112 | 49.4% | 0.001 | 0.579 | 0.842 | 2.9% |
| 107 | 764,686 | 0.89% | 0.709 | 33 | 62 | 135 | 45.7% | 0.013 | 0.729 | 0.266 | 26.9% |
| 25 | 762,090 | 0.89% | 0.702 | 35 | 55 | 110 | 49.1% | 0.004 | 0.620 | 0.567 | 4.7% |
| 60 | 761,143 | 0.88% | 0.630 | 37 | 152 | 111 | 19.9% | 0.110 | 0.754 | 0.118 | 54.2% |
| 1 | 750,086 | 0.87% | 0.613 | 38 | 77 | 127 | 38.3% | 0.134 | 0.757 | 0.111 | 53.1% |
| 59 | 748,545 | 0.87% | 0.686 | 35 | 68 | 125 | 43.1% | 0.017 | 0.777 | 0.171 | 37.4% |
| 18 | 723,938 | 0.84% | 0.665 | 32 | 222 | 43 | 41.4% | 0.027 | 0.711 | 0.162 | 43.4% |
| 90 | 717,769 | 0.83% | 0.703 | 35 | 52 | 116 | 46.3% | 0.013 | 0.633 | 0.362 | 14.1% |
| 5 | 703,659 | 0.82% | 0.640 | 37 | 258 | 24 | 38.5% | 0.052 | 0.782 | 0.099 | 61.1% |
| 52 | 702,915 | 0.82% | 0.657 | 34 | 163 | 89 | 23.2% | 0.126 | 0.708 | 0.139 | 47.5% |
| 115 | 700,811 | 0.81% | 0.646 | 36 | 55 | 118 | 44.8% | 0.024 | 0.725 | 0.210 | 22.7% |
| 80 | 699,772 | 0.81% | 0.732 | 35 | 51 | 101 | 49.5% | 0.002 | 0.665 | 0.516 | 2.4% |
| 73 | 694,952 | 0.81% | 0.610 | 38 | 194 | 71 | 19.9% | 0.128 | 0.766 | 0.087 | 65.0% |
| 48 | 693,308 | 0.81% | 0.602 | 39 | 143 | 101 | 31.9% | 0.155 | 0.784 | 0.089 | 63.2% |
| 98 | 686,220 | 0.80% | 0.675 | 31 | 308 | 0 | 37.0% | 0.043 | 0.710 | 0.157 | 42.3% |
| 16 | 686,167 | 0.80% | 0.666 | 35 | 50 | 100 | 47.9% | 0.008 | 0.644 | 0.323 | 11.3% |
| 26 | 672,725 | 0.78% | 0.605 | 40 | 142 | 83 | 29.3% | 0.079 | 0.784 | 0.092 | 64.1% |
| 9 | 669,981 | 0.78% | 0.633 | 38 | 111 | 145 | 28.0% | 0.111 | 0.759 | 0.116 | 52.9% |
| 123 | 669,385 | 0.78% | 0.772 | 31 | 48 | 96 | 49.7% | 0.001 | 0.697 | 0.521 | 1.3% |
| 12 | 669,357 | 0.78% | 0.630 | 39 | 107 | 102 | 27.9% | 0.069 | 0.768 | 0.095 | 62.9% |
| 71 | 669,149 | 0.78% | 0.626 | 38 | 110 | 127 | 24.8% | 0.123 | 0.757 | 0.146 | 46.3% |
| 53 | 664,838 | 0.77% | 0.625 | 38 | 60 | 90 | 32.7% | 0.050 | 0.745 | 0.208 | 22.4% |
| 110 | 663,856 | 0.77% | 0.664 | 38 | 52 | 114 | 44.3% | 0.026 | 0.789 | 0.188 | 32.1% |
| 22 | 658,454 | 0.77% | 0.606 | 39 | 100 | 102 | 36.9% | 0.066 | 0.757 | 0.108 | 55.1% |
| 30 | 655,618 | 0.76% | 0.635 | 38 | 139 | 27 | 19.8% | 0.126 | 0.805 | 0.127 | 50.2% |
| 19 | 655,315 | 0.76% | 0.647 | 37 | 62 | 111 | 39.0% | 0.053 | 0.752 | 0.198 | 34.8% |
| 11 | 641,966 | 0.75% | 0.641 | 35 | 89 | 82 | 26.7% | 0.076 | 0.686 | 0.166 | 40.4% |
| 46 | 639,388 | 0.74% | 0.652 | 38 | 48 | 89 | 38.8% | 0.074 | 0.752 | 0.180 | 35.0% |
| 99 | 635,940 | 0.74% | 0.620 | 38 | 177 | 45 | 25.6% | 0.136 | 0.754 | 0.088 | 66.1% |
| 75 | 634,780 | 0.74% | 0.614 | 39 | 80 | 117 | 24.5% | 0.142 | 0.750 | 0.147 | 42.7% |
| 68 | 633,578 | 0.74% | 0.665 | 37 | 53 | 101 | 32.3% | 0.060 | 0.778 | 0.184 | 42.9% |
| 84 | 603,777 | 0.70% | 0.619 | 38 | 91 | 91 | 38.6% | 0.085 | 0.763 | 0.096 | 62.1% |
| 54 | 603,070 | 0.70% | 0.604 | 39 | 198 | 47 | 31.7% | 0.057 | 0.766 | 0.091 | 63.8% |
| 106 | 600,761 | 0.70% | 0.617 | 39 | 162 | 40 | 27.1% | 0.077 | 0.767 | 0.083 | 70.1% |
| 100 | 581,731 | 0.68% | 0.609 | 38 | 130 | 73 | 20.4% | 0.184 | 0.743 | 0.113 | 54.0% |
| 24 | 576,516 | 0.67% | 0.613 | 37 | 101 | 76 | 17.4% | 0.140 | 0.694 | 0.237 | 27.9% |
| 15 | 567,759 | 0.66% | 0.603 | 39 | 44 | 87 | 41.2% | 0.034 | 0.693 | 0.214 | 26.4% |
| 104 | 566,788 | 0.66% | 0.616 | 38 | 106 | 87 | 21.5% | 0.137 | 0.729 | 0.153 | 42.3% |
| 29 | 565,539 | 0.66% | 0.623 | 38 | 90 | 87 | 20.0% | 0.127 | 0.709 | 0.150 | 44.3% |
| 64 | 564,196 | 0.66% | 0.593 | 38 | 82 | 45 | 21.5% | 0.145 | 0.695 | 0.180 | 32.5% |
| 45 | 560,673 | 0.65% | 0.624 | 38 | 103 | 96 | 34.1% | 0.075 | 0.757 | 0.093 | 66.4% |
| 94 | 557,217 | 0.65% | 0.716 | 29 | 185 | 17 | 32.3% | 0.056 | 0.684 | 0.167 | 45.6% |
| 120 | 553,959 | 0.64% | 0.649 | 39 | 44 | 90 | 39.7% | 0.058 | 0.799 | 0.200 | 31.6% |
| 101 | 549,923 | 0.64% | 0.619 | 38 | 146 | 83 | 25.2% | 0.085 | 0.767 | 0.087 | 65.3% |
| 3 | 544,351 | 0.63% | 0.648 | 38 | 41 | 83 | 43.8% | 0.026 | 0.747 | 0.208 | 29.0% |
| 31 | 527,439 | 0.61% | 0.614 | 38 | 142 | 58 | 32.4% | 0.069 | 0.745 | 0.105 | 58.8% |
| 7 | 525,595 | 0.61% | 0.609 | 38 | 155 | 19 | 16.9% | 0.116 | 0.757 | 0.115 | 54.3% |
| 23 | 525,136 | 0.61% | 0.610 | 40 | 53 | 68 | 31.6% | 0.077 | 0.707 | 0.189 | 32.0% |
| 91 | 501,121 | 0.58% | 0.643 | 37 | 38 | 75 | 42.7% | 0.028 | 0.687 | 0.323 | 14.4% |
| 57 | 500,189 | 0.58% | 0.624 | 38 | 180 | 24 | 24.9% | 0.107 | 0.738 | 0.105 | 60.2% |
| 40 | 494,922 | 0.58% | 0.607 | 39 | 149 | 45 | 33.1% | 0.067 | 0.751 | 0.109 | 59.6% |
| 81 | 493,516 | 0.57% | 0.660 | 36 | 36 | 76 | 47.1% | 0.009 | 0.675 | 0.262 | 17.9% |
| 36 | 488,755 | 0.57% | 0.633 | 36 | 274 | 1 | 31.3% | 0.063 | 0.727 | 0.158 | 43.4% |
| 13 | 483,921 | 0.56% | 0.622 | 36 | 543 | 0 | 39.4% | 0.068 | 0.718 | 0.159 | 48.1% |
| 8 | 483,208 | 0.56% | 0.636 | 38 | 40 | 85 | 38.3% | 0.054 | 0.729 | 0.216 | 22.0% |
| 87 | 470,334 | 0.55% | 0.665 | 36 | 34 | 70 | 45.2% | 0.013 | 0.691 | 0.302 | 14.9% |
| 97 | 466,368 | 0.54% | 0.630 | 39 | 42 | 80 | 38.5% | 0.042 | 0.729 | 0.199 | 28.6% |
| 44 | 466,301 | 0.54% | 0.643 | 38 | 42 | 70 | 35.2% | 0.101 | 0.781 | 0.158 | 40.9% |
| 124 | 465,228 | 0.54% | 0.708 | 36 | 34 | 71 | 48.0% | 0.006 | 0.666 | 0.462 | 10.0% |
| 108 | 462,929 | 0.54% | 0.631 | 38 | 59 | 85 | 26.0% | 0.095 | 0.756 | 0.167 | 40.6% |
| 47 | 462,475 | 0.54% | 0.648 | 38 | 35 | 70 | 42.4% | 0.033 | 0.745 | 0.240 | 20.6% |
| 118 | 445,230 | 0.52% | 0.605 | 40 | 34 | 68 | 45.0% | 0.057 | 0.702 | 0.226 | 24.7% |
| 51 | 443,709 | 0.52% | 0.655 | 37 | 33 | 69 | 45.8% | 0.020 | 0.675 | 0.287 | 16.9% |
| 77 | 409,919 | 0.48% | 0.782 | 32 | 30 | 59 | 49.1% | 0.003 | 0.566 | 1.337 | 1.8% |
| 102 | 406,825 | 0.47% | 0.631 | 37 | 57 | 53 | 20.8% | 0.087 | 0.744 | 0.189 | 34.9% |
| 72 | 400,629 | 0.47% | 0.621 | 40 | 29 | 61 | 44.6% | 0.020 | 0.665 | 0.323 | 14.8% |
| 20 | 393,641 | 0.46% | 0.674 | 34 | 29 | 46 | 40.5% | 0.029 | 0.679 | 0.226 | 31.4% |
| 103 | 383,353 | 0.45% | 0.617 | 39 | 43 | 66 | 28.4% | 0.093 | 0.694 | 0.216 | 28.7% |
| 69 | 378,784 | 0.44% | 0.663 | 37 | 31 | 73 | 42.7% | 0.024 | 0.736 | 0.196 | 34.5% |
| 122 | 373,027 | 0.43% | 0.650 | 33 | 263 | 3 | 23.8% | 0.067 | 0.668 | 0.175 | 45.7% |
| 14 | 369,303 | 0.43% | 0.756 | 34 | 27 | 55 | 48.3% | 0.006 | 0.665 | 0.772 | 3.5% |
| 96 | 367,525 | 0.43% | 0.668 | 37 | 27 | 58 | 42.4% | 0.032 | 0.679 | 0.372 | 21.0% |
| 116 | 353,149 | 0.41% | 0.644 | 37 | 45 | 42 | 26.6% | 0.089 | 0.661 | 0.227 | 28.8% |
| 112 | 351,651 | 0.41% | 0.644 | 40 | 27 | 56 | 42.3% | 0.021 | 0.706 | 0.281 | 16.5% |
| 37 | 343,244 | 0.40% | 0.634 | 40 | 28 | 56 | 38.1% | 0.063 | 0.712 | 0.210 | 41.1% |
| 55 | 342,432 | 0.40% | 0.696 | 34 | 25 | 41 | 43.9% | 0.017 | 0.605 | 0.456 | 17.2% |
| 32 | 340,022 | 0.40% | 0.671 | 36 | 25 | 51 | 46.0% | 0.011 | 0.609 | 0.425 | 6.0% |
| 62 | 333,940 | 0.39% | 0.669 | 36 | 25 | 50 | 46.8% | 0.009 | 0.662 | 0.402 | 15.2% |
| 109 | 326,804 | 0.38% | 0.653 | 38 | 24 | 41 | 26.6% | 0.068 | 0.645 | 0.408 | 12.6% |
| 49 | 323,256 | 0.38% | 0.716 | 34 | 24 | 52 | 44.2% | 0.023 | 0.595 | 0.742 | 3.8% |
| 92 | 321,518 | 0.37% | 0.632 | 40 | 24 | 46 | 44.1% | 0.024 | 0.627 | 0.353 | 12.4% |
| 63 | 310,554 | 0.36% | 0.679 | 36 | 23 | 46 | 45.1% | 0.015 | 0.644 | 0.332 | 15.7% |
| 79 | 291,018 | 0.34% | 0.651 | 38 | 21 | 42 | 48.9% | 0.005 | 0.605 | 0.454 | 4.1% |
| 2 | 265,811 | 0.31% | 0.672 | 38 | 19 | 37 | 48.5% | 0.003 | 0.526 | 0.653 | 3.3% |
| 38 | 235,900 | 0.27% | 0.641 | 40 | 18 | 33 | 44.4% | 0.025 | 0.686 | 0.281 | 19.1% |
| 28 | 227,159 | 0.26% | 0.729 | 34 | 17 | 27 | 39.0% | 0.036 | 0.595 | 0.711 | 4.3% |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **13, 98** `near% > 50` (most of the cluster's own tokens are near-ties with a rival subspace ⇒ it is a slice of a continuum, not a separated mode) — 29 of 128, worst first: **106 (70%), 45 (66%), 99 (66%), 101 (65%), 73 (65%), 26 (64%), 54 (64%), 48 (63%), 12 (63%), 33 (63%)**, … 19 more.

Separation summary: mean `near%` **33.0%** across clusters (range 0.9%–70.1%); least-separated cluster **106** (70.1% near-ties, closest rival **45** taking 19.2% of its runner-up mass). A high global near-tie rate is not a defect of the fit — it is evidence the token manifold is a continuum, which is exactly what the soft/MPPCA assignment (`--soft`) is for.

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v6_subspace_big_d64 --out runs/clustering/v6_subspace_big_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure; see `runs/clustering/v6_subspace_big_d64/temporal_report.md`.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.419, mean 0.432, max 0.805 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 30 ↔ 126 | 0.805 | 0.445 |
| 85 ↔ 126 | 0.801 | 0.722 |
| 120 ↔ 126 | 0.799 | 0.436 |
| 110 ↔ 117 | 0.789 | 0.686 |
| 114 ↔ 117 | 0.787 | 0.691 |
| 26 ↔ 48 | 0.784 | 0.779 |
| 5 ↔ 33 | 0.782 | 0.337 |
| 44 ↔ 126 | 0.781 | 0.419 |
| 68 ↔ 117 | 0.778 | 0.654 |
| 85 ↔ 114 | 0.778 | 0.725 |
| 10 ↔ 59 | 0.777 | 0.617 |
| 58 ↔ 86 | 0.773 | 0.538 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
