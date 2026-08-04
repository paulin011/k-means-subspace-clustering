# Clustering report (subspace_kmeans) — `runs/clustering/v5_subspace_big_d128`

*Generated 2026-08-04 13:35 by `analyze_clusters.py`. K=128 affine subspaces of dim 128 in 2048-dim token space, 86,016,000 tokens.*

## Overview

The model groups the 86,016,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 128 clusters, and fits a 128-dimensional flat (an *affine subspace*: a centroid plus a basis of directions) through each one. A token is assigned to whichever cluster leaves the smallest **orthogonal residual** — the part of the token that its cluster's subspace cannot reconstruct.

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
- **Uⱼ** (`model['U'][j]`, shape `[2048, 128]`): an orthonormal basis for cluster *j*'s subspace; its columns are PC directions in descending eigenvalue order.
- **Orthogonal residual** of a token *x* under cluster *j*: `‖x − μⱼ‖² − ‖Uⱼᵀ(x − μⱼ)‖²`. The first term is the squared distance to the centroid; the second is the part of that distance the subspace *captures*. What's left is the unexplained residual that the assignment minimises.
- **eigvals** (`model['eigvals'][j]`): the top-128 eigenvalues of cluster *j*'s within-cluster covariance — variance along each kept PC direction.
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
| dim | 128 |
| iters | 25 |
| tol | 0.001 |
| linear | False |
| seed | 0 |
| chunk_size | 65536 |
| gpus | 2 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `82ca602ed7e7`
- **Files:** 7000 latent files, 12288 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v5_subspace_big_d128/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v5_subspace_big_d128/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7659.20 | 100.00% | 127 | 23,702,502 |
| 2 | 1537.71 | 82.05% | 1 | 3,046,976 |
| 3 | 1254.74 | 39.89% | 1 | 1,956,018 |
| 4 | 1183.18 | 21.74% | 1 | 1,847,308 |
| 5 | 1148.56 | 14.95% | 1 | 1,802,821 |
| 6 | 1126.78 | 10.79% | 1 | 1,782,011 |
| 7 | 1112.20 | 8.17% | 1 | 1,761,839 |
| 8 | 1101.79 | 6.25% | 1 | 1,736,340 |
| 9 | 1094.42 | 4.85% | 1 | 1,709,658 |
| 10 | 1088.87 | 3.89% | 1 | 1,695,932 |
| 11 | 1084.82 | 3.20% | 1 | 1,686,683 |
| 12 | 1081.58 | 2.69% | 1 | 1,671,045 |
| 13 | 1079.01 | 2.27% | 1 | 1,660,471 |
| 14 | 1076.99 | 1.95% | 1 | 1,650,157 |
| 15 | 1075.33 | 1.72% | 1 | 1,636,778 |
| 16 | 1073.82 | 1.52% | 1 | 1,628,287 |
| 17 | 1072.43 | 1.33% | 1 | 1,614,236 |
| 18 | 1071.37 | 1.13% | 1 | 1,594,367 |
| 19 | 1070.59 | 0.97% | 1 | 1,585,536 |
| 20 | 1070.00 | 0.88% | 1 | 1,581,365 |
| 21 | 1069.43 | 0.80% | 1 | 1,580,816 |
| 22 | 1068.93 | 0.72% | 1 | 1,579,721 |
| 23 | 1068.55 | 0.65% | 1 | 1,575,612 |
| 24 | 1068.22 | 0.60% | 1 | 1,574,534 |
| 25 | 1067.90 | 0.56% | 1 | 1,574,526 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5998**, split into:

- **6.2%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **76.0%** is `captured / total`. It is the chunk of `within` that the top-128 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5998, the same denominator as the other two, which is what lets all three add to 100%.
- **17.8%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-128): 0.816** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 128 subspace directions recover about 82%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 47 / median 63 / max 129** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=129 = a truncation warning. Your max is 129, at the cap → a cluster is truncated; consider raising `--dim`.

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`EVR(top-128)` = `Σ eigvals[j] / trace[j]` — fraction of this cluster's own variance captured by its subspace (1.0 = the subspace explains the cluster perfectly; near the global average ⇒ d truncates the spectrum).*
- *`d80` = smallest number of leading PC directions whose eigenvalues reach 80% of `Σ eigvals[j]` (capped at d+1=129 when even all 128 fall short). Low d80 ⇒ a few directions dominate; d80 ≈ d ⇒ a flat spectrum the subspace truncates.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`maxAff` = this cluster's subspace affinity to its **nearest** neighbour, `maxⱼ≠ᵢ ‖UᵢᵀUⱼ‖²_F / d` ∈ [0,1]. A per-cluster separation score: **high ⇒ some other cluster spans nearly the same directions**, so this row is a merge candidate. The affinity table below lists only the top pairs, so a near-duplicate cluster is invisible there unless its pair happens to rank; this column always shows it.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.00%–1.83% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-128) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 1,574,526 | 1.83% | 0.845 | 53 | 113 | 225 | 50.0% | 0.000 | 0.597 |
| 114 | 1,365,618 | 1.59% | 0.789 | 66 | 99 | 217 | 43.9% | 0.021 | 0.808 |
| 27 | 1,343,642 | 1.56% | 0.843 | 55 | 96 | 192 | 50.0% | 0.000 | 0.665 |
| 59 | 1,271,399 | 1.48% | 0.831 | 60 | 97 | 207 | 47.0% | 0.006 | 0.749 |
| 127 | 1,175,809 | 1.37% | 0.862 | 52 | 84 | 168 | 50.0% | 0.000 | 0.693 |
| 74 | 1,103,359 | 1.28% | 0.838 | 56 | 79 | 158 | 49.9% | 0.001 | 0.689 |
| 39 | 1,083,907 | 1.26% | 0.846 | 57 | 78 | 158 | 49.4% | 0.001 | 0.742 |
| 70 | 1,014,693 | 1.18% | 0.843 | 55 | 73 | 148 | 49.3% | 0.001 | 0.792 |
| 56 | 1,007,427 | 1.17% | 0.845 | 55 | 72 | 144 | 50.0% | 0.000 | 0.721 |
| 35 | 1,006,905 | 1.17% | 0.850 | 57 | 72 | 144 | 49.9% | 0.000 | 0.667 |
| 117 | 1,000,516 | 1.16% | 0.813 | 63 | 74 | 154 | 47.4% | 0.011 | 0.780 |
| 50 | 994,073 | 1.16% | 0.877 | 47 | 72 | 142 | 50.0% | 0.000 | 0.711 |
| 31 | 988,045 | 1.15% | 0.770 | 66 | 112 | 102 | 28.4% | 0.062 | 0.829 |
| 82 | 986,238 | 1.15% | 0.788 | 64 | 89 | 177 | 35.7% | 0.060 | 0.817 |
| 78 | 980,030 | 1.14% | 0.869 | 49 | 71 | 140 | 50.0% | 0.000 | 0.736 |
| 42 | 961,181 | 1.12% | 0.800 | 60 | 76 | 146 | 40.4% | 0.046 | 0.792 |
| 21 | 952,025 | 1.11% | 0.866 | 62 | 69 | 136 | 50.0% | 0.000 | 0.608 |
| 86 | 934,307 | 1.09% | 0.799 | 64 | 69 | 157 | 44.3% | 0.011 | 0.824 |
| 83 | 922,805 | 1.07% | 0.864 | 52 | 66 | 132 | 49.9% | 0.000 | 0.733 |
| 0 | 916,815 | 1.07% | 0.858 | 51 | 66 | 131 | 49.9% | 0.000 | 0.666 |
| 14 | 909,963 | 1.06% | 0.861 | 53 | 65 | 130 | 50.0% | 0.000 | 0.665 |
| 93 | 896,151 | 1.04% | 0.883 | 49 | 65 | 128 | 50.0% | 0.000 | 0.731 |
| 107 | 894,539 | 1.04% | 0.875 | 56 | 64 | 128 | 49.9% | 0.000 | 0.633 |
| 125 | 894,157 | 1.04% | 0.836 | 56 | 64 | 128 | 49.9% | 0.001 | 0.721 |
| 68 | 889,356 | 1.03% | 0.881 | 48 | 64 | 127 | 50.0% | 0.000 | 0.731 |
| 61 | 888,123 | 1.03% | 0.795 | 63 | 64 | 128 | 48.9% | 0.005 | 0.738 |
| 73 | 885,567 | 1.03% | 0.770 | 68 | 156 | 58 | 23.8% | 0.087 | 0.797 |
| 18 | 880,798 | 1.02% | 0.806 | 56 | 96 | 159 | 38.1% | 0.045 | 0.755 |
| 110 | 857,207 | 1.00% | 0.815 | 64 | 62 | 133 | 46.9% | 0.012 | 0.773 |
| 48 | 846,496 | 0.98% | 0.775 | 67 | 79 | 127 | 36.2% | 0.122 | 0.829 |
| 1 | 846,093 | 0.98% | 0.765 | 67 | 70 | 122 | 34.9% | 0.105 | 0.798 |
| 33 | 835,300 | 0.97% | 0.810 | 62 | 65 | 140 | 46.5% | 0.014 | 0.796 |
| 25 | 832,368 | 0.97% | 0.834 | 58 | 60 | 119 | 49.9% | 0.000 | 0.683 |
| 6 | 827,268 | 0.96% | 0.818 | 63 | 63 | 132 | 47.9% | 0.007 | 0.796 |
| 4 | 827,126 | 0.96% | 0.825 | 61 | 65 | 136 | 46.7% | 0.008 | 0.788 |
| 85 | 826,362 | 0.96% | 0.790 | 66 | 60 | 109 | 43.3% | 0.029 | 0.802 |
| 95 | 821,434 | 0.95% | 0.817 | 59 | 67 | 152 | 38.5% | 0.048 | 0.755 |
| 52 | 806,289 | 0.94% | 0.793 | 58 | 125 | 84 | 23.3% | 0.129 | 0.751 |
| 46 | 802,565 | 0.93% | 0.802 | 65 | 58 | 124 | 44.3% | 0.024 | 0.808 |
| 90 | 795,516 | 0.92% | 0.827 | 59 | 57 | 116 | 49.0% | 0.004 | 0.696 |
| 94 | 794,584 | 0.92% | 0.847 | 57 | 57 | 115 | 49.4% | 0.001 | 0.766 |
| 43 | 785,081 | 0.91% | 0.817 | 50 | 151 | 34 | 23.8% | 0.123 | 0.737 |
| 60 | 780,128 | 0.91% | 0.793 | 65 | 60 | 126 | 45.1% | 0.048 | 0.826 |
| 89 | 777,649 | 0.90% | 0.805 | 62 | 56 | 111 | 49.8% | 0.001 | 0.660 |
| 80 | 777,408 | 0.90% | 0.850 | 56 | 56 | 110 | 49.7% | 0.001 | 0.721 |
| 16 | 765,719 | 0.89% | 0.816 | 61 | 55 | 111 | 48.7% | 0.005 | 0.718 |
| 124 | 762,844 | 0.89% | 0.829 | 62 | 56 | 117 | 47.7% | 0.005 | 0.761 |
| 115 | 754,420 | 0.88% | 0.794 | 64 | 56 | 115 | 46.6% | 0.020 | 0.780 |
| 5 | 742,078 | 0.86% | 0.800 | 64 | 55 | 115 | 40.6% | 0.032 | 0.826 |
| 71 | 739,486 | 0.86% | 0.783 | 64 | 76 | 105 | 36.7% | 0.077 | 0.796 |
| 126 | 735,090 | 0.85% | 0.786 | 63 | 60 | 123 | 37.8% | 0.167 | 0.796 |
| 26 | 729,791 | 0.85% | 0.767 | 68 | 112 | 90 | 31.3% | 0.105 | 0.791 |
| 123 | 728,131 | 0.85% | 0.881 | 48 | 53 | 104 | 50.0% | 0.000 | 0.752 |
| 76 | 722,591 | 0.84% | 0.826 | 58 | 52 | 106 | 48.3% | 0.006 | 0.770 |
| 40 | 719,166 | 0.84% | 0.776 | 67 | 84 | 136 | 28.8% | 0.088 | 0.815 |
| 15 | 709,928 | 0.83% | 0.768 | 69 | 51 | 106 | 48.9% | 0.004 | 0.765 |
| 58 | 706,870 | 0.82% | 0.789 | 65 | 55 | 119 | 43.3% | 0.021 | 0.824 |
| 54 | 697,371 | 0.81% | 0.769 | 66 | 185 | 34 | 38.2% | 0.062 | 0.797 |
| 19 | 697,353 | 0.81% | 0.806 | 64 | 52 | 113 | 42.8% | 0.022 | 0.792 |
| 96 | 693,214 | 0.81% | 0.821 | 63 | 51 | 107 | 47.0% | 0.010 | 0.742 |
| 69 | 692,651 | 0.81% | 0.809 | 64 | 53 | 112 | 46.8% | 0.012 | 0.787 |
| 119 | 679,130 | 0.79% | 0.865 | 52 | 49 | 97 | 49.9% | 0.000 | 0.799 |
| 121 | 662,092 | 0.77% | 0.835 | 57 | 48 | 98 | 48.1% | 0.006 | 0.728 |
| 84 | 649,226 | 0.75% | 0.772 | 66 | 62 | 97 | 40.6% | 0.138 | 0.796 |
| 22 | 646,002 | 0.75% | 0.775 | 67 | 73 | 83 | 37.3% | 0.118 | 0.802 |
| 17 | 637,385 | 0.74% | 0.880 | 47 | 46 | 91 | 50.0% | 0.000 | 0.752 |
| 105 | 636,797 | 0.74% | 0.762 | 69 | 48 | 90 | 34.1% | 0.046 | 0.754 |
| 81 | 631,723 | 0.73% | 0.811 | 63 | 46 | 92 | 48.6% | 0.005 | 0.707 |
| 36 | 630,825 | 0.73% | 0.775 | 62 | 189 | 5 | 31.7% | 0.052 | 0.745 |
| 98 | 613,360 | 0.71% | 0.811 | 51 | 268 | 0 | 31.4% | 0.048 | 0.716 |
| 97 | 608,638 | 0.71% | 0.787 | 67 | 53 | 105 | 36.2% | 0.057 | 0.716 |
| 51 | 605,314 | 0.70% | 0.796 | 65 | 44 | 93 | 46.4% | 0.015 | 0.711 |
| 53 | 600,865 | 0.70% | 0.778 | 65 | 50 | 86 | 31.4% | 0.056 | 0.762 |
| 120 | 594,010 | 0.69% | 0.798 | 63 | 44 | 91 | 44.8% | 0.030 | 0.773 |
| 104 | 590,766 | 0.69% | 0.781 | 65 | 72 | 78 | 24.3% | 0.160 | 0.769 |
| 23 | 587,598 | 0.68% | 0.771 | 68 | 43 | 85 | 48.5% | 0.005 | 0.765 |
| 122 | 581,011 | 0.68% | 0.784 | 60 | 126 | 8 | 20.2% | 0.103 | 0.784 |
| 55 | 569,578 | 0.66% | 0.846 | 57 | 42 | 84 | 48.8% | 0.002 | 0.796 |
| 62 | 567,793 | 0.66% | 0.809 | 64 | 42 | 84 | 47.6% | 0.008 | 0.780 |
| 118 | 564,627 | 0.66% | 0.772 | 69 | 41 | 81 | 48.9% | 0.008 | 0.759 |
| 3 | 563,759 | 0.66% | 0.805 | 65 | 41 | 80 | 44.0% | 0.033 | 0.771 |
| 10 | 563,345 | 0.65% | 0.853 | 52 | 41 | 80 | 49.6% | 0.001 | 0.729 |
| 45 | 551,565 | 0.64% | 0.784 | 67 | 64 | 104 | 33.4% | 0.055 | 0.824 |
| 101 | 551,429 | 0.64% | 0.781 | 65 | 50 | 98 | 33.4% | 0.079 | 0.824 |
| 75 | 549,794 | 0.64% | 0.798 | 68 | 40 | 90 | 43.0% | 0.019 | 0.798 |
| 111 | 547,956 | 0.64% | 0.874 | 50 | 40 | 78 | 49.8% | 0.001 | 0.799 |
| 57 | 542,378 | 0.63% | 0.802 | 66 | 42 | 96 | 39.0% | 0.036 | 0.798 |
| 77 | 531,817 | 0.62% | 0.887 | 48 | 38 | 76 | 50.0% | 0.000 | 0.619 |
| 7 | 525,432 | 0.61% | 0.762 | 70 | 48 | 73 | 28.8% | 0.084 | 0.767 |
| 24 | 519,492 | 0.60% | 0.777 | 65 | 85 | 54 | 19.0% | 0.129 | 0.768 |
| 30 | 518,387 | 0.60% | 0.800 | 65 | 39 | 75 | 42.9% | 0.055 | 0.802 |
| 113 | 513,800 | 0.60% | 0.844 | 53 | 38 | 73 | 48.3% | 0.006 | 0.770 |
| 11 | 513,319 | 0.60% | 0.808 | 61 | 38 | 68 | 36.3% | 0.027 | 0.715 |
| 47 | 510,144 | 0.59% | 0.800 | 64 | 37 | 79 | 42.3% | 0.024 | 0.762 |
| 37 | 508,849 | 0.59% | 0.795 | 68 | 38 | 78 | 47.2% | 0.010 | 0.804 |
| 41 | 487,240 | 0.57% | 0.840 | 59 | 36 | 73 | 48.4% | 0.003 | 0.796 |
| 79 | 486,394 | 0.57% | 0.800 | 67 | 35 | 79 | 43.3% | 0.026 | 0.734 |
| 9 | 484,866 | 0.56% | 0.791 | 67 | 38 | 83 | 38.7% | 0.039 | 0.817 |
| 112 | 478,073 | 0.56% | 0.805 | 69 | 35 | 75 | 45.8% | 0.017 | 0.700 |
| 99 | 477,003 | 0.55% | 0.811 | 65 | 35 | 73 | 47.6% | 0.008 | 0.716 |
| 67 | 474,430 | 0.55% | 0.802 | 67 | 35 | 79 | 42.9% | 0.017 | 0.794 |
| 44 | 461,596 | 0.54% | 0.803 | 65 | 34 | 70 | 45.3% | 0.018 | 0.757 |
| 108 | 457,743 | 0.53% | 0.800 | 64 | 40 | 83 | 31.0% | 0.076 | 0.758 |
| 87 | 452,132 | 0.53% | 0.820 | 62 | 33 | 65 | 48.2% | 0.006 | 0.713 |
| 88 | 447,984 | 0.52% | 0.821 | 61 | 33 | 67 | 47.1% | 0.013 | 0.720 |
| 34 | 442,614 | 0.51% | 0.830 | 59 | 32 | 64 | 48.9% | 0.003 | 0.718 |
| 20 | 429,480 | 0.50% | 0.823 | 62 | 31 | 63 | 46.4% | 0.012 | 0.720 |
| 106 | 426,988 | 0.50% | 0.800 | 66 | 32 | 70 | 43.8% | 0.014 | 0.794 |
| 100 | 419,376 | 0.49% | 0.782 | 68 | 31 | 65 | 45.8% | 0.030 | 0.765 |
| 72 | 415,513 | 0.48% | 0.787 | 68 | 30 | 57 | 48.5% | 0.005 | 0.711 |
| 38 | 415,143 | 0.48% | 0.804 | 67 | 31 | 65 | 46.4% | 0.029 | 0.752 |
| 29 | 413,423 | 0.48% | 0.792 | 66 | 46 | 84 | 28.6% | 0.089 | 0.723 |
| 103 | 411,923 | 0.48% | 0.777 | 67 | 48 | 52 | 23.1% | 0.119 | 0.751 |
| 64 | 405,477 | 0.47% | 0.770 | 67 | 60 | 26 | 18.8% | 0.162 | 0.756 |
| 91 | 388,275 | 0.45% | 0.811 | 64 | 29 | 59 | 45.7% | 0.021 | 0.743 |
| 102 | 376,716 | 0.44% | 0.785 | 64 | 39 | 54 | 24.5% | 0.065 | 0.719 |
| 13 | 365,026 | 0.42% | 0.779 | 67 | 34 | 37 | 25.0% | 0.090 | 0.735 |
| 63 | 334,290 | 0.39% | 0.820 | 61 | 24 | 48 | 48.8% | 0.005 | 0.708 |
| 32 | 325,048 | 0.38% | 0.824 | 63 | 24 | 46 | 49.5% | 0.002 | 0.681 |
| 109 | 311,353 | 0.36% | 0.810 | 64 | 23 | 37 | 37.9% | 0.032 | 0.678 |
| 2 | 307,961 | 0.36% | 0.820 | 64 | 22 | 44 | 50.0% | 0.000 | 0.604 |
| 49 | 280,020 | 0.33% | 0.872 | 56 | 21 | 40 | 50.0% | 0.000 | 0.676 |
| 116 | 274,973 | 0.32% | 0.804 | 62 | 48 | 6 | 17.2% | 0.153 | 0.696 |
| 12 | 266,955 | 0.31% | 0.814 | 65 | 20 | 40 | 37.6% | 0.046 | 0.722 |
| 28 | 238,103 | 0.28% | 0.872 | 56 | 18 | 34 | 50.0% | 0.000 | 0.676 |
| 92 | 234,123 | 0.27% | 0.812 | 66 | 17 | 33 | 48.7% | 0.003 | 0.704 |
| 8 | 1 | 0.00% | 0.000 | 129 | 1 | 0 | 0.0% | 3.000 | 0.000 |
| 65 | 1 | 0.00% | 0.000 | 129 | 1 | 0 | 0.0% | 3.000 | 0.000 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **8, 65, 98** under a quarter of the mean cluster size: **8, 65**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v5_subspace_big_d128 --out runs/clustering/v5_subspace_big_d128/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 128 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.483, mean 0.482, max 0.829 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 31 ↔ 48 | 0.829 | 0.756 |
| 5 ↔ 60 | 0.826 | 0.676 |
| 58 ↔ 86 | 0.824 | 0.698 |
| 45 ↔ 101 | 0.824 | 0.731 |
| 9 ↔ 101 | 0.817 | 0.770 |
| 58 ↔ 82 | 0.817 | 0.656 |
| 9 ↔ 45 | 0.816 | 0.696 |
| 40 ↔ 45 | 0.815 | 0.606 |
| 46 ↔ 114 | 0.808 | 0.703 |
| 37 ↔ 58 | 0.804 | 0.825 |
| 22 ↔ 48 | 0.802 | 0.770 |
| 30 ↔ 85 | 0.802 | 0.787 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
