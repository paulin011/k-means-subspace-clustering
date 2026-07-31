# Clustering report (subspace_kmeans) — `runs/clustering/v7_seed1_d64`

*Generated 2026-07-31 18:08 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

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
| seed | 1 |
| chunk_size | 262144 |
| gpus | 2 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `81dd389c7786`
- **Files:** 7000 latent files, 12288 tokens each, seed 1.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v7_seed1_d64/sample.json --seed 1 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v7_seed1_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7606.38 | 100.00% | 109 | 26,015,684 |
| 2 | 2357.22 | 80.25% | 7,327 | 4,637,212 |
| 3 | 2064.40 | 42.92% | 96,907 | 2,414,799 |
| 4 | 1992.52 | 22.05% | 174,816 | 2,378,153 |
| 5 | 1962.79 | 14.62% | 202,141 | 2,156,350 |
| 6 | 1945.15 | 10.97% | 208,597 | 1,948,301 |
| 7 | 1933.86 | 8.57% | 216,360 | 1,866,215 |
| 8 | 1925.94 | 7.09% | 219,017 | 1,806,938 |
| 9 | 1919.82 | 5.95% | 204,263 | 1,767,727 |
| 10 | 1915.19 | 5.04% | 198,657 | 1,742,895 |
| 11 | 1911.60 | 4.34% | 204,579 | 1,720,719 |
| 12 | 1908.84 | 3.79% | 209,341 | 1,702,611 |
| 13 | 1906.55 | 3.34% | 212,906 | 1,688,384 |
| 14 | 1904.73 | 3.02% | 215,578 | 1,674,793 |
| 15 | 1903.11 | 2.81% | 216,608 | 1,667,955 |
| 16 | 1901.64 | 2.62% | 209,457 | 1,660,332 |
| 17 | 1900.28 | 2.45% | 203,755 | 1,652,287 |
| 18 | 1899.13 | 2.27% | 199,846 | 1,644,156 |
| 19 | 1898.10 | 2.13% | 197,053 | 1,636,944 |
| 20 | 1897.13 | 2.03% | 194,861 | 1,630,616 |
| 21 | 1896.20 | 1.93% | 193,232 | 1,626,766 |
| 22 | 1895.31 | 1.85% | 192,066 | 1,625,825 |
| 23 | 1894.43 | 1.77% | 191,951 | 1,626,311 |
| 24 | 1893.60 | 1.65% | 192,225 | 1,627,328 |
| 25 | 1892.94 | 1.55% | 192,632 | 1,627,985 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **6001**, split into:

- **8.5%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **59.9%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 6001, the same denominator as the other two, which is what lets all three add to 100%.
- **31.6%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.662** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 66%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 27 / median 37 / max 40** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=65 = a truncation warning. Your max is 40, below the cap → no cluster truncated, d=64 has headroom.

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

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.22%–1.89% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 1,627,915 | 1.89% | 0.697 | 33 | 117 | 237 | 49.1% | 0.002 | 0.659 |
| 17 | 1,378,083 | 1.60% | 0.683 | 35 | 99 | 205 | 48.5% | 0.005 | 0.657 |
| 118 | 1,374,590 | 1.60% | 0.735 | 29 | 172 | 246 | 43.3% | 0.036 | 0.711 |
| 77 | 1,318,700 | 1.53% | 0.678 | 33 | 99 | 221 | 45.8% | 0.013 | 0.680 |
| 15 | 1,296,281 | 1.51% | 0.739 | 30 | 93 | 185 | 49.3% | 0.003 | 0.717 |
| 21 | 1,248,475 | 1.45% | 0.652 | 34 | 209 | 156 | 27.0% | 0.081 | 0.747 |
| 1 | 1,238,370 | 1.44% | 0.708 | 31 | 160 | 183 | 44.3% | 0.012 | 0.718 |
| 5 | 1,174,919 | 1.37% | 0.674 | 34 | 203 | 244 | 36.6% | 0.044 | 0.735 |
| 66 | 1,170,534 | 1.36% | 0.700 | 31 | 109 | 214 | 37.6% | 0.054 | 0.683 |
| 42 | 1,132,315 | 1.32% | 0.662 | 34 | 84 | 173 | 42.5% | 0.018 | 0.665 |
| 119 | 1,092,384 | 1.27% | 0.713 | 29 | 167 | 106 | 44.7% | 0.029 | 0.718 |
| 24 | 1,088,863 | 1.27% | 0.690 | 29 | 206 | 165 | 26.7% | 0.097 | 0.680 |
| 120 | 1,079,616 | 1.26% | 0.652 | 36 | 139 | 219 | 40.9% | 0.036 | 0.772 |
| 111 | 1,064,530 | 1.24% | 0.690 | 31 | 226 | 142 | 38.7% | 0.035 | 0.715 |
| 71 | 1,027,865 | 1.19% | 0.651 | 37 | 134 | 191 | 30.6% | 0.062 | 0.793 |
| 28 | 1,021,871 | 1.19% | 0.633 | 38 | 181 | 173 | 28.6% | 0.063 | 0.776 |
| 110 | 1,017,130 | 1.18% | 0.760 | 30 | 74 | 152 | 48.0% | 0.008 | 0.717 |
| 50 | 1,015,339 | 1.18% | 0.656 | 36 | 170 | 223 | 34.7% | 0.050 | 0.777 |
| 2 | 1,002,759 | 1.17% | 0.639 | 39 | 85 | 146 | 42.9% | 0.058 | 0.793 |
| 125 | 997,167 | 1.16% | 0.628 | 39 | 81 | 152 | 40.0% | 0.081 | 0.797 |
| 38 | 995,386 | 1.16% | 0.698 | 34 | 72 | 148 | 48.1% | 0.008 | 0.683 |
| 95 | 989,485 | 1.15% | 0.627 | 38 | 139 | 184 | 29.6% | 0.056 | 0.764 |
| 16 | 980,583 | 1.14% | 0.743 | 31 | 71 | 140 | 49.9% | 0.000 | 0.566 |
| 63 | 965,598 | 1.12% | 0.661 | 35 | 125 | 229 | 30.9% | 0.076 | 0.766 |
| 23 | 949,916 | 1.10% | 0.676 | 36 | 99 | 191 | 39.3% | 0.023 | 0.781 |
| 20 | 931,298 | 1.08% | 0.650 | 33 | 244 | 53 | 36.6% | 0.046 | 0.702 |
| 104 | 918,801 | 1.07% | 0.639 | 37 | 110 | 160 | 30.8% | 0.084 | 0.772 |
| 78 | 908,302 | 1.06% | 0.759 | 33 | 65 | 130 | 49.9% | 0.000 | 0.555 |
| 4 | 905,846 | 1.05% | 0.727 | 27 | 234 | 50 | 33.2% | 0.058 | 0.682 |
| 94 | 905,635 | 1.05% | 0.625 | 38 | 155 | 148 | 24.0% | 0.158 | 0.754 |
| 11 | 900,956 | 1.05% | 0.646 | 37 | 157 | 148 | 28.4% | 0.096 | 0.772 |
| 123 | 893,035 | 1.04% | 0.744 | 34 | 64 | 128 | 49.7% | 0.001 | 0.575 |
| 126 | 880,199 | 1.02% | 0.757 | 30 | 63 | 126 | 49.2% | 0.004 | 0.700 |
| 47 | 875,880 | 1.02% | 0.716 | 37 | 63 | 126 | 49.6% | 0.002 | 0.518 |
| 117 | 867,398 | 1.01% | 0.617 | 38 | 120 | 189 | 33.6% | 0.059 | 0.764 |
| 7 | 848,466 | 0.99% | 0.634 | 37 | 184 | 109 | 18.8% | 0.130 | 0.746 |
| 52 | 838,189 | 0.97% | 0.683 | 32 | 220 | 36 | 42.7% | 0.018 | 0.712 |
| 6 | 838,034 | 0.97% | 0.642 | 37 | 60 | 123 | 48.6% | 0.003 | 0.671 |
| 103 | 811,469 | 0.94% | 0.670 | 32 | 224 | 84 | 25.2% | 0.102 | 0.721 |
| 86 | 798,762 | 0.93% | 0.729 | 33 | 58 | 115 | 49.3% | 0.002 | 0.632 |
| 74 | 783,700 | 0.91% | 0.760 | 33 | 56 | 112 | 50.0% | 0.000 | 0.638 |
| 32 | 780,594 | 0.91% | 0.648 | 35 | 58 | 116 | 44.9% | 0.017 | 0.663 |
| 10 | 769,253 | 0.89% | 0.618 | 38 | 140 | 126 | 27.8% | 0.110 | 0.746 |
| 97 | 747,885 | 0.87% | 0.629 | 39 | 73 | 132 | 34.0% | 0.175 | 0.830 |
| 101 | 731,213 | 0.85% | 0.659 | 34 | 225 | 63 | 32.6% | 0.063 | 0.750 |
| 34 | 722,828 | 0.84% | 0.615 | 38 | 80 | 114 | 39.1% | 0.118 | 0.771 |
| 55 | 722,682 | 0.84% | 0.678 | 34 | 77 | 118 | 41.7% | 0.025 | 0.727 |
| 45 | 702,031 | 0.82% | 0.605 | 39 | 53 | 108 | 44.5% | 0.026 | 0.693 |
| 114 | 699,641 | 0.81% | 0.692 | 35 | 63 | 90 | 41.6% | 0.018 | 0.781 |
| 39 | 699,208 | 0.81% | 0.651 | 35 | 178 | 61 | 35.0% | 0.052 | 0.766 |
| 18 | 698,848 | 0.81% | 0.609 | 39 | 137 | 80 | 34.2% | 0.071 | 0.770 |
| 43 | 693,729 | 0.81% | 0.655 | 32 | 272 | 9 | 38.1% | 0.036 | 0.722 |
| 44 | 688,156 | 0.80% | 0.700 | 31 | 51 | 87 | 40.9% | 0.037 | 0.642 |
| 109 | 678,770 | 0.79% | 0.647 | 38 | 72 | 123 | 35.8% | 0.034 | 0.736 |
| 35 | 670,236 | 0.78% | 0.652 | 38 | 57 | 105 | 40.4% | 0.091 | 0.830 |
| 59 | 667,010 | 0.78% | 0.648 | 38 | 52 | 99 | 43.3% | 0.032 | 0.797 |
| 54 | 658,507 | 0.77% | 0.603 | 39 | 108 | 147 | 33.2% | 0.079 | 0.786 |
| 70 | 651,814 | 0.76% | 0.622 | 38 | 207 | 49 | 23.3% | 0.089 | 0.776 |
| 31 | 650,213 | 0.76% | 0.643 | 37 | 47 | 93 | 48.5% | 0.007 | 0.651 |
| 105 | 636,600 | 0.74% | 0.625 | 38 | 132 | 115 | 31.2% | 0.088 | 0.769 |
| 99 | 630,350 | 0.73% | 0.653 | 36 | 46 | 91 | 47.8% | 0.008 | 0.720 |
| 57 | 625,574 | 0.73% | 0.631 | 37 | 256 | 24 | 33.6% | 0.096 | 0.763 |
| 81 | 616,594 | 0.72% | 0.617 | 38 | 132 | 107 | 19.0% | 0.216 | 0.776 |
| 33 | 613,450 | 0.71% | 0.623 | 38 | 138 | 60 | 23.5% | 0.093 | 0.757 |
| 58 | 604,401 | 0.70% | 0.649 | 37 | 46 | 97 | 41.5% | 0.026 | 0.685 |
| 51 | 604,144 | 0.70% | 0.603 | 39 | 149 | 62 | 29.1% | 0.182 | 0.773 |
| 60 | 597,771 | 0.69% | 0.703 | 35 | 47 | 103 | 45.5% | 0.019 | 0.714 |
| 36 | 596,652 | 0.69% | 0.657 | 33 | 354 | 0 | 39.1% | 0.036 | 0.689 |
| 92 | 588,994 | 0.68% | 0.635 | 38 | 89 | 101 | 22.5% | 0.154 | 0.794 |
| 121 | 586,123 | 0.68% | 0.586 | 39 | 81 | 58 | 22.3% | 0.133 | 0.693 |
| 69 | 584,878 | 0.68% | 0.606 | 40 | 155 | 29 | 20.3% | 0.114 | 0.786 |
| 25 | 577,456 | 0.67% | 0.650 | 37 | 46 | 93 | 44.0% | 0.028 | 0.701 |
| 87 | 574,731 | 0.67% | 0.627 | 39 | 54 | 77 | 28.7% | 0.068 | 0.760 |
| 29 | 567,047 | 0.66% | 0.645 | 37 | 49 | 102 | 37.6% | 0.051 | 0.721 |
| 83 | 561,493 | 0.65% | 0.648 | 36 | 96 | 102 | 18.9% | 0.136 | 0.721 |
| 90 | 555,755 | 0.65% | 0.681 | 36 | 41 | 94 | 42.0% | 0.030 | 0.563 |
| 112 | 552,868 | 0.64% | 0.623 | 39 | 45 | 95 | 40.9% | 0.036 | 0.711 |
| 85 | 550,336 | 0.64% | 0.652 | 36 | 48 | 89 | 36.0% | 0.059 | 0.731 |
| 91 | 549,375 | 0.64% | 0.604 | 39 | 63 | 73 | 24.5% | 0.093 | 0.693 |
| 49 | 536,013 | 0.62% | 0.637 | 37 | 42 | 68 | 41.7% | 0.036 | 0.711 |
| 115 | 530,959 | 0.62% | 0.609 | 39 | 135 | 100 | 30.4% | 0.094 | 0.757 |
| 102 | 523,510 | 0.61% | 0.666 | 35 | 38 | 75 | 47.7% | 0.008 | 0.671 |
| 8 | 522,411 | 0.61% | 0.661 | 36 | 38 | 78 | 47.9% | 0.008 | 0.720 |
| 79 | 516,419 | 0.60% | 0.630 | 38 | 134 | 31 | 24.4% | 0.092 | 0.754 |
| 37 | 515,070 | 0.60% | 0.626 | 38 | 67 | 104 | 29.8% | 0.106 | 0.754 |
| 14 | 511,954 | 0.60% | 0.611 | 39 | 47 | 79 | 32.5% | 0.070 | 0.701 |
| 61 | 505,724 | 0.59% | 0.622 | 38 | 73 | 89 | 36.8% | 0.151 | 0.761 |
| 93 | 504,166 | 0.59% | 0.629 | 38 | 89 | 61 | 19.1% | 0.153 | 0.776 |
| 30 | 498,588 | 0.58% | 0.609 | 38 | 142 | 20 | 16.4% | 0.153 | 0.714 |
| 0 | 498,223 | 0.58% | 0.646 | 34 | 152 | 38 | 18.4% | 0.102 | 0.714 |
| 100 | 493,782 | 0.57% | 0.739 | 33 | 36 | 79 | 47.7% | 0.003 | 0.651 |
| 108 | 485,557 | 0.56% | 0.622 | 39 | 45 | 90 | 36.8% | 0.046 | 0.703 |
| 62 | 485,012 | 0.56% | 0.652 | 38 | 36 | 66 | 40.5% | 0.059 | 0.716 |
| 12 | 479,764 | 0.56% | 0.656 | 37 | 39 | 71 | 40.7% | 0.050 | 0.682 |
| 26 | 473,395 | 0.55% | 0.634 | 37 | 91 | 65 | 16.8% | 0.142 | 0.747 |
| 75 | 463,140 | 0.54% | 0.625 | 37 | 339 | 5 | 32.3% | 0.056 | 0.725 |
| 48 | 460,927 | 0.54% | 0.619 | 37 | 84 | 22 | 16.9% | 0.153 | 0.677 |
| 80 | 459,007 | 0.53% | 0.603 | 39 | 256 | 11 | 32.2% | 0.116 | 0.744 |
| 13 | 442,040 | 0.51% | 0.653 | 38 | 33 | 68 | 45.1% | 0.017 | 0.769 |
| 53 | 415,410 | 0.48% | 0.682 | 36 | 30 | 65 | 46.1% | 0.016 | 0.626 |
| 122 | 413,852 | 0.48% | 0.632 | 35 | 254 | 3 | 25.0% | 0.103 | 0.725 |
| 89 | 412,123 | 0.48% | 0.692 | 33 | 45 | 36 | 31.5% | 0.050 | 0.670 |
| 76 | 403,673 | 0.47% | 0.623 | 40 | 30 | 55 | 39.6% | 0.036 | 0.663 |
| 22 | 402,503 | 0.47% | 0.607 | 39 | 35 | 53 | 36.1% | 0.113 | 0.716 |
| 127 | 399,814 | 0.46% | 0.615 | 39 | 82 | 50 | 24.1% | 0.117 | 0.719 |
| 56 | 399,424 | 0.46% | 0.631 | 38 | 36 | 78 | 34.2% | 0.068 | 0.651 |
| 27 | 393,948 | 0.46% | 0.631 | 39 | 59 | 70 | 20.4% | 0.149 | 0.772 |
| 9 | 389,125 | 0.45% | 0.613 | 40 | 29 | 62 | 43.6% | 0.020 | 0.687 |
| 98 | 382,905 | 0.45% | 0.652 | 39 | 29 | 58 | 45.7% | 0.017 | 0.741 |
| 72 | 381,613 | 0.44% | 0.590 | 40 | 307 | 1 | 30.2% | 0.063 | 0.714 |
| 67 | 367,578 | 0.43% | 0.657 | 38 | 27 | 56 | 44.7% | 0.020 | 0.712 |
| 64 | 363,086 | 0.42% | 0.662 | 38 | 28 | 60 | 39.9% | 0.041 | 0.760 |
| 113 | 354,883 | 0.41% | 0.699 | 39 | 26 | 53 | 48.1% | 0.008 | 0.690 |
| 84 | 353,299 | 0.41% | 0.647 | 39 | 28 | 60 | 41.3% | 0.025 | 0.745 |
| 65 | 344,400 | 0.40% | 0.705 | 35 | 26 | 56 | 46.9% | 0.010 | 0.714 |
| 107 | 300,761 | 0.35% | 0.634 | 39 | 26 | 53 | 37.3% | 0.022 | 0.741 |
| 19 | 300,745 | 0.35% | 0.648 | 39 | 24 | 43 | 31.7% | 0.056 | 0.701 |
| 88 | 295,029 | 0.34% | 0.676 | 37 | 22 | 42 | 49.2% | 0.002 | 0.541 |
| 116 | 290,602 | 0.34% | 0.643 | 39 | 21 | 41 | 48.1% | 0.009 | 0.653 |
| 124 | 262,336 | 0.30% | 0.624 | 39 | 27 | 52 | 30.0% | 0.077 | 0.679 |
| 106 | 257,034 | 0.30% | 0.673 | 37 | 19 | 37 | 48.4% | 0.007 | 0.682 |
| 82 | 254,715 | 0.30% | 0.655 | 37 | 19 | 35 | 42.7% | 0.029 | 0.658 |
| 41 | 253,849 | 0.30% | 0.647 | 39 | 19 | 39 | 47.0% | 0.012 | 0.677 |
| 40 | 253,184 | 0.29% | 0.632 | 40 | 19 | 44 | 41.4% | 0.029 | 0.693 |
| 46 | 246,520 | 0.29% | 0.663 | 38 | 19 | 40 | 43.5% | 0.026 | 0.661 |
| 68 | 231,938 | 0.27% | 0.665 | 36 | 23 | 33 | 32.6% | 0.056 | 0.695 |
| 96 | 230,896 | 0.27% | 0.664 | 36 | 26 | 11 | 19.8% | 0.115 | 0.561 |
| 73 | 193,240 | 0.22% | 0.681 | 39 | 14 | 28 | 45.2% | 0.012 | 0.602 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **36**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v7_seed1_d64 --out runs/clustering/v7_seed1_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure; see `runs/clustering/v7_seed1_d64/temporal_report.md`.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.434, mean 0.439, max 0.830 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 35 ↔ 97 | 0.830 | 0.659 |
| 59 ↔ 125 | 0.797 | 0.673 |
| 35 ↔ 92 | 0.794 | 0.649 |
| 2 ↔ 71 | 0.793 | 0.607 |
| 54 ↔ 69 | 0.786 | 0.741 |
| 23 ↔ 114 | 0.781 | 0.455 |
| 97 ↔ 125 | 0.778 | 0.720 |
| 23 ↔ 50 | 0.777 | 0.607 |
| 28 ↔ 70 | 0.776 | 0.264 |
| 81 ↔ 93 | 0.776 | 0.544 |
| 51 ↔ 54 | 0.773 | 0.743 |
| 27 ↔ 81 | 0.772 | 0.543 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
