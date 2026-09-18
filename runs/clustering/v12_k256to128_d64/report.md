# Clustering report (subspace_kmeans_merged_residual) — `runs/clustering/v12_k256to128_d64`

*Generated 2026-08-12 17:27 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

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
| chunk_size | 131072 |
| gpus | 2 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `82ca602ed7e7`
- **Files:** 7000 latent files, 12288 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans_merged_residual.py --files-from runs/clustering/v12_k256to128_d64/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v12_k256to128_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 1889.48 | 100.00% | 229,849 | 1,651,626 |
| 2 | 1882.66 | 4.16% | 224,423 | 1,589,676 |
| 3 | 1879.23 | 2.93% | 219,032 | 1,544,413 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5999**, split into:

- **7.2%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **61.5%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5999, the same denominator as the other two, which is what lets all three add to 100%.
- **31.3%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.672** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 67%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

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

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.25%–1.75% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 68 | 1,508,278 | 1.75% | 0.626 | 39 | 143 | 244 | 37.1% | 0.056 | 0.782 |
| 119 | 1,345,089 | 1.56% | 0.659 | 35 | 141 | 272 | 42.1% | 0.030 | 0.781 |
| 100 | 1,207,699 | 1.40% | 0.682 | 35 | 96 | 199 | 46.2% | 0.008 | 0.781 |
| 89 | 1,200,146 | 1.40% | 0.633 | 38 | 128 | 195 | 38.4% | 0.029 | 0.755 |
| 112 | 1,157,232 | 1.35% | 0.635 | 37 | 141 | 175 | 34.2% | 0.049 | 0.784 |
| 41 | 1,123,243 | 1.31% | 0.628 | 37 | 211 | 131 | 21.0% | 0.124 | 0.707 |
| 33 | 1,101,602 | 1.28% | 0.749 | 30 | 79 | 157 | 49.9% | 0.000 | 0.569 |
| 62 | 1,100,865 | 1.28% | 0.651 | 38 | 93 | 181 | 42.4% | 0.021 | 0.775 |
| 103 | 1,048,446 | 1.22% | 0.665 | 33 | 166 | 184 | 37.3% | 0.052 | 0.763 |
| 43 | 1,034,023 | 1.20% | 0.632 | 39 | 85 | 163 | 39.2% | 0.115 | 0.823 |
| 109 | 1,032,219 | 1.20% | 0.611 | 38 | 132 | 206 | 30.5% | 0.077 | 0.757 |
| 124 | 1,012,407 | 1.18% | 0.652 | 35 | 166 | 172 | 24.3% | 0.116 | 0.753 |
| 21 | 1,008,418 | 1.17% | 0.620 | 38 | 151 | 202 | 26.6% | 0.146 | 0.766 |
| 74 | 1,008,205 | 1.17% | 0.734 | 31 | 73 | 148 | 48.8% | 0.008 | 0.747 |
| 101 | 1,002,168 | 1.17% | 0.628 | 38 | 96 | 157 | 35.7% | 0.201 | 0.823 |
| 118 | 986,257 | 1.15% | 0.648 | 36 | 97 | 161 | 34.0% | 0.060 | 0.764 |
| 125 | 982,519 | 1.14% | 0.682 | 34 | 88 | 185 | 45.7% | 0.010 | 0.777 |
| 23 | 977,667 | 1.14% | 0.598 | 39 | 183 | 116 | 32.5% | 0.120 | 0.783 |
| 17 | 966,126 | 1.12% | 0.612 | 39 | 201 | 117 | 28.7% | 0.072 | 0.775 |
| 60 | 963,204 | 1.12% | 0.638 | 39 | 72 | 140 | 45.4% | 0.028 | 0.819 |
| 18 | 942,827 | 1.10% | 0.642 | 37 | 157 | 116 | 40.6% | 0.069 | 0.784 |
| 110 | 939,085 | 1.09% | 0.640 | 35 | 344 | 3 | 39.1% | 0.040 | 0.718 |
| 49 | 935,624 | 1.09% | 0.668 | 35 | 101 | 162 | 31.4% | 0.067 | 0.764 |
| 27 | 931,469 | 1.08% | 0.705 | 28 | 330 | 1 | 28.3% | 0.061 | 0.634 |
| 14 | 921,847 | 1.07% | 0.608 | 39 | 112 | 148 | 35.6% | 0.136 | 0.783 |
| 1 | 913,535 | 1.06% | 0.609 | 38 | 106 | 140 | 40.2% | 0.095 | 0.757 |
| 79 | 909,041 | 1.06% | 0.606 | 39 | 228 | 92 | 38.5% | 0.023 | 0.731 |
| 114 | 907,586 | 1.06% | 0.616 | 38 | 175 | 99 | 32.9% | 0.057 | 0.737 |
| 20 | 898,260 | 1.04% | 0.673 | 31 | 191 | 105 | 23.2% | 0.117 | 0.646 |
| 122 | 883,961 | 1.03% | 0.760 | 30 | 64 | 128 | 49.2% | 0.004 | 0.734 |
| 10 | 883,037 | 1.03% | 0.743 | 32 | 64 | 128 | 49.6% | 0.001 | 0.675 |
| 78 | 879,837 | 1.02% | 0.633 | 38 | 87 | 135 | 30.1% | 0.109 | 0.800 |
| 95 | 857,980 | 1.00% | 0.627 | 37 | 112 | 149 | 32.5% | 0.099 | 0.737 |
| 116 | 842,051 | 0.98% | 0.641 | 37 | 76 | 136 | 41.0% | 0.033 | 0.741 |
| 117 | 828,617 | 0.96% | 0.743 | 31 | 60 | 120 | 49.4% | 0.004 | 0.696 |
| 90 | 827,044 | 0.96% | 0.657 | 31 | 307 | 15 | 39.4% | 0.032 | 0.682 |
| 58 | 812,970 | 0.95% | 0.719 | 32 | 61 | 130 | 47.4% | 0.006 | 0.747 |
| 105 | 812,899 | 0.95% | 0.739 | 32 | 59 | 120 | 49.3% | 0.002 | 0.685 |
| 36 | 810,915 | 0.94% | 0.726 | 33 | 58 | 118 | 49.1% | 0.003 | 0.674 |
| 108 | 797,883 | 0.93% | 0.701 | 33 | 69 | 131 | 43.7% | 0.020 | 0.777 |
| 35 | 793,581 | 0.92% | 0.610 | 38 | 187 | 65 | 23.2% | 0.101 | 0.750 |
| 54 | 781,630 | 0.91% | 0.613 | 39 | 135 | 105 | 20.5% | 0.144 | 0.784 |
| 120 | 780,330 | 0.91% | 0.643 | 36 | 59 | 117 | 45.2% | 0.030 | 0.727 |
| 111 | 772,141 | 0.90% | 0.769 | 30 | 56 | 110 | 49.7% | 0.001 | 0.670 |
| 22 | 767,529 | 0.89% | 0.649 | 39 | 60 | 110 | 43.2% | 0.026 | 0.782 |
| 121 | 757,463 | 0.88% | 0.624 | 39 | 60 | 122 | 44.2% | 0.028 | 0.734 |
| 5 | 752,770 | 0.88% | 0.632 | 37 | 104 | 156 | 27.1% | 0.089 | 0.784 |
| 42 | 752,655 | 0.88% | 0.730 | 31 | 55 | 105 | 48.7% | 0.003 | 0.738 |
| 31 | 741,849 | 0.86% | 0.641 | 38 | 56 | 116 | 45.4% | 0.021 | 0.720 |
| 12 | 741,734 | 0.86% | 0.734 | 36 | 53 | 106 | 49.9% | 0.001 | 0.564 |
| 6 | 724,352 | 0.84% | 0.762 | 30 | 52 | 99 | 47.5% | 0.014 | 0.734 |
| 106 | 692,309 | 0.80% | 0.616 | 36 | 325 | 0 | 22.0% | 0.115 | 0.723 |
| 84 | 686,466 | 0.80% | 0.593 | 38 | 61 | 98 | 33.0% | 0.080 | 0.712 |
| 4 | 678,198 | 0.79% | 0.674 | 36 | 58 | 124 | 45.1% | 0.016 | 0.759 |
| 34 | 674,434 | 0.78% | 0.620 | 39 | 60 | 113 | 41.2% | 0.047 | 0.763 |
| 97 | 669,569 | 0.78% | 0.723 | 27 | 92 | 82 | 31.1% | 0.088 | 0.636 |
| 19 | 655,939 | 0.76% | 0.743 | 34 | 47 | 94 | 49.7% | 0.001 | 0.634 |
| 11 | 649,067 | 0.75% | 0.641 | 37 | 51 | 108 | 41.0% | 0.028 | 0.729 |
| 75 | 647,041 | 0.75% | 0.784 | 30 | 47 | 92 | 49.7% | 0.002 | 0.651 |
| 44 | 646,540 | 0.75% | 0.671 | 35 | 48 | 100 | 47.2% | 0.013 | 0.713 |
| 83 | 639,067 | 0.74% | 0.764 | 34 | 46 | 92 | 49.6% | 0.001 | 0.620 |
| 56 | 630,765 | 0.73% | 0.730 | 35 | 46 | 92 | 48.6% | 0.005 | 0.615 |
| 67 | 627,146 | 0.73% | 0.769 | 31 | 45 | 89 | 49.6% | 0.001 | 0.675 |
| 15 | 622,760 | 0.72% | 0.761 | 33 | 45 | 89 | 50.0% | 0.000 | 0.641 |
| 0 | 622,035 | 0.72% | 0.743 | 33 | 45 | 88 | 49.6% | 0.001 | 0.612 |
| 81 | 615,646 | 0.72% | 0.653 | 36 | 45 | 91 | 47.6% | 0.007 | 0.670 |
| 70 | 602,951 | 0.70% | 0.616 | 37 | 280 | 5 | 37.7% | 0.056 | 0.723 |
| 85 | 602,458 | 0.70% | 0.605 | 39 | 134 | 82 | 24.5% | 0.106 | 0.747 |
| 92 | 597,750 | 0.69% | 0.649 | 36 | 43 | 85 | 48.3% | 0.004 | 0.635 |
| 39 | 594,187 | 0.69% | 0.750 | 35 | 43 | 85 | 49.9% | 0.000 | 0.660 |
| 98 | 591,904 | 0.69% | 0.660 | 35 | 43 | 81 | 45.0% | 0.018 | 0.704 |
| 72 | 569,325 | 0.66% | 0.729 | 34 | 41 | 84 | 48.5% | 0.006 | 0.670 |
| 86 | 563,599 | 0.66% | 0.618 | 37 | 101 | 89 | 18.3% | 0.121 | 0.712 |
| 25 | 560,853 | 0.65% | 0.788 | 30 | 41 | 80 | 49.8% | 0.001 | 0.699 |
| 91 | 560,180 | 0.65% | 0.726 | 33 | 41 | 82 | 48.0% | 0.006 | 0.674 |
| 113 | 559,641 | 0.65% | 0.740 | 34 | 40 | 80 | 50.0% | 0.000 | 0.641 |
| 123 | 552,433 | 0.64% | 0.611 | 39 | 53 | 74 | 29.6% | 0.080 | 0.717 |
| 65 | 550,493 | 0.64% | 0.732 | 34 | 40 | 79 | 49.5% | 0.002 | 0.697 |
| 26 | 544,549 | 0.63% | 0.625 | 39 | 64 | 77 | 24.9% | 0.092 | 0.720 |
| 48 | 534,214 | 0.62% | 0.801 | 31 | 39 | 76 | 49.8% | 0.001 | 0.651 |
| 88 | 533,018 | 0.62% | 0.785 | 32 | 39 | 76 | 49.9% | 0.002 | 0.696 |
| 96 | 531,943 | 0.62% | 0.784 | 33 | 38 | 76 | 50.0% | 0.000 | 0.640 |
| 77 | 531,811 | 0.62% | 0.611 | 39 | 39 | 75 | 46.3% | 0.012 | 0.664 |
| 28 | 531,233 | 0.62% | 0.752 | 34 | 38 | 76 | 49.9% | 0.001 | 0.673 |
| 50 | 528,964 | 0.61% | 0.658 | 37 | 42 | 88 | 42.4% | 0.027 | 0.716 |
| 47 | 527,941 | 0.61% | 0.720 | 35 | 38 | 80 | 48.0% | 0.008 | 0.643 |
| 16 | 524,693 | 0.61% | 0.627 | 38 | 67 | 81 | 22.8% | 0.114 | 0.743 |
| 80 | 523,864 | 0.61% | 0.745 | 35 | 38 | 75 | 49.9% | 0.001 | 0.673 |
| 63 | 523,261 | 0.61% | 0.599 | 40 | 40 | 82 | 42.5% | 0.065 | 0.729 |
| 13 | 522,516 | 0.61% | 0.720 | 35 | 38 | 75 | 49.4% | 0.002 | 0.636 |
| 107 | 521,654 | 0.61% | 0.679 | 35 | 38 | 80 | 46.2% | 0.018 | 0.713 |
| 55 | 515,960 | 0.60% | 0.607 | 39 | 66 | 58 | 22.3% | 0.109 | 0.694 |
| 30 | 515,833 | 0.60% | 0.695 | 35 | 41 | 88 | 47.0% | 0.007 | 0.755 |
| 61 | 501,305 | 0.58% | 0.640 | 37 | 38 | 80 | 43.4% | 0.028 | 0.693 |
| 37 | 500,573 | 0.58% | 0.784 | 32 | 36 | 72 | 49.5% | 0.001 | 0.594 |
| 71 | 484,151 | 0.56% | 0.654 | 37 | 35 | 74 | 45.3% | 0.016 | 0.693 |
| 93 | 480,320 | 0.56% | 0.735 | 33 | 35 | 70 | 48.8% | 0.004 | 0.697 |
| 66 | 476,394 | 0.55% | 0.730 | 33 | 35 | 71 | 47.8% | 0.008 | 0.683 |
| 94 | 469,484 | 0.55% | 0.678 | 39 | 34 | 71 | 48.1% | 0.008 | 0.678 |
| 29 | 464,171 | 0.54% | 0.634 | 38 | 50 | 37 | 31.6% | 0.096 | 0.720 |
| 126 | 452,266 | 0.53% | 0.654 | 38 | 34 | 70 | 40.3% | 0.060 | 0.737 |
| 51 | 451,454 | 0.52% | 0.630 | 39 | 37 | 75 | 41.1% | 0.029 | 0.734 |
| 82 | 449,962 | 0.52% | 0.612 | 39 | 33 | 71 | 45.0% | 0.018 | 0.712 |
| 127 | 448,270 | 0.52% | 0.624 | 39 | 39 | 63 | 33.8% | 0.064 | 0.686 |
| 45 | 427,507 | 0.50% | 0.718 | 35 | 31 | 63 | 46.3% | 0.011 | 0.650 |
| 53 | 419,747 | 0.49% | 0.660 | 36 | 31 | 59 | 43.2% | 0.016 | 0.659 |
| 24 | 411,597 | 0.48% | 0.720 | 34 | 30 | 59 | 49.6% | 0.001 | 0.498 |
| 32 | 400,218 | 0.47% | 0.658 | 37 | 29 | 60 | 46.9% | 0.007 | 0.686 |
| 115 | 397,997 | 0.46% | 0.757 | 33 | 29 | 56 | 49.2% | 0.002 | 0.704 |
| 40 | 396,254 | 0.46% | 0.664 | 36 | 29 | 58 | 47.6% | 0.010 | 0.646 |
| 59 | 395,549 | 0.46% | 0.736 | 33 | 29 | 59 | 48.0% | 0.008 | 0.702 |
| 3 | 391,829 | 0.46% | 0.658 | 38 | 29 | 55 | 45.0% | 0.021 | 0.713 |
| 9 | 390,674 | 0.45% | 0.701 | 35 | 28 | 56 | 48.7% | 0.007 | 0.669 |
| 7 | 389,993 | 0.45% | 0.626 | 37 | 59 | 36 | 19.0% | 0.153 | 0.642 |
| 102 | 362,267 | 0.42% | 0.616 | 40 | 28 | 53 | 37.5% | 0.044 | 0.711 |
| 57 | 361,034 | 0.42% | 0.649 | 37 | 29 | 39 | 25.5% | 0.078 | 0.676 |
| 64 | 360,411 | 0.42% | 0.724 | 35 | 27 | 53 | 48.5% | 0.003 | 0.724 |
| 8 | 352,124 | 0.41% | 0.621 | 40 | 26 | 55 | 46.2% | 0.024 | 0.694 |
| 46 | 351,675 | 0.41% | 0.701 | 36 | 26 | 50 | 49.5% | 0.003 | 0.635 |
| 52 | 350,134 | 0.41% | 0.613 | 40 | 31 | 54 | 32.0% | 0.068 | 0.725 |
| 76 | 348,771 | 0.41% | 0.697 | 36 | 25 | 50 | 49.7% | 0.003 | 0.620 |
| 104 | 348,593 | 0.41% | 0.731 | 34 | 26 | 52 | 48.3% | 0.004 | 0.760 |
| 87 | 331,119 | 0.38% | 0.641 | 40 | 26 | 59 | 42.1% | 0.032 | 0.721 |
| 38 | 322,347 | 0.37% | 0.651 | 38 | 24 | 48 | 47.4% | 0.007 | 0.604 |
| 2 | 316,049 | 0.37% | 0.672 | 37 | 23 | 45 | 49.4% | 0.002 | 0.525 |
| 99 | 294,344 | 0.34% | 0.616 | 40 | 22 | 43 | 46.0% | 0.028 | 0.722 |
| 69 | 260,549 | 0.30% | 0.651 | 40 | 19 | 37 | 48.8% | 0.004 | 0.622 |
| 73 | 214,343 | 0.25% | 0.672 | 34 | 17 | 22 | 30.0% | 0.058 | 0.640 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **106**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v12_k256to128_d64 --out runs/clustering/v12_k256to128_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.392, mean 0.414, max 0.823 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 43 ↔ 101 | 0.823 | 0.750 |
| 43 ↔ 60 | 0.819 | 0.757 |
| 78 ↔ 101 | 0.800 | 0.410 |
| 18 ↔ 112 | 0.784 | 0.287 |
| 5 ↔ 54 | 0.784 | 0.314 |
| 14 ↔ 23 | 0.783 | 0.727 |
| 22 ↔ 68 | 0.782 | 0.667 |
| 100 ↔ 119 | 0.781 | 0.755 |
| 108 ↔ 125 | 0.777 | 0.791 |
| 17 ↔ 101 | 0.775 | 0.340 |
| 22 ↔ 62 | 0.775 | 0.711 |
| 62 ↔ 119 | 0.773 | 0.663 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
