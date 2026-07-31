# Clustering report (subspace_kmeans) — `runs/clustering/v8_seed2_d64`

*Generated 2026-07-31 18:10 by `analyze_clusters.py`. K=128 affine subspaces of dim 64 in 2048-dim token space, 86,016,000 tokens.*

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
| seed | 2 |
| chunk_size | 262144 |
| gpus | 2 |
| tokens analyzed | 86,016,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `aa5126ce3e0d`
- **Files:** 7000 latent files, 12288 tokens each, seed 2.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v8_seed2_d64/sample.json --seed 2 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v8_seed2_d64/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7414.86 | 100.00% | 301 | 35,852,088 |
| 2 | 2406.77 | 77.03% | 10,909 | 8,970,464 |
| 3 | 2076.64 | 47.40% | 124,957 | 2,296,573 |
| 4 | 1996.48 | 23.18% | 167,076 | 2,109,746 |
| 5 | 1964.61 | 15.12% | 232,104 | 1,974,343 |
| 6 | 1946.07 | 11.20% | 239,253 | 1,890,775 |
| 7 | 1934.51 | 8.80% | 242,395 | 1,867,794 |
| 8 | 1926.51 | 7.23% | 242,834 | 1,852,679 |
| 9 | 1920.39 | 6.12% | 242,547 | 1,863,539 |
| 10 | 1915.50 | 5.28% | 241,325 | 1,877,599 |
| 11 | 1911.65 | 4.59% | 240,251 | 1,885,874 |
| 12 | 1908.52 | 4.09% | 239,867 | 1,900,734 |
| 13 | 1905.82 | 3.72% | 240,808 | 1,916,290 |
| 14 | 1903.45 | 3.41% | 242,178 | 1,923,451 |
| 15 | 1901.33 | 3.13% | 244,741 | 1,931,431 |
| 16 | 1899.43 | 2.88% | 248,024 | 1,937,522 |
| 17 | 1897.75 | 2.68% | 242,921 | 1,944,732 |
| 18 | 1896.17 | 2.48% | 232,709 | 1,947,283 |
| 19 | 1894.73 | 2.27% | 240,502 | 1,937,912 |
| 20 | 1893.54 | 2.11% | 245,635 | 1,930,197 |
| 21 | 1892.48 | 1.99% | 243,364 | 1,928,367 |
| 22 | 1891.37 | 1.88% | 241,706 | 1,922,329 |
| 23 | 1890.32 | 1.74% | 240,025 | 1,919,085 |
| 24 | 1889.50 | 1.58% | 238,120 | 1,918,478 |
| 25 | 1888.82 | 1.43% | 236,222 | 1,918,305 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **6000**, split into:

- **8.5%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **60.1%** is `captured / total`. It is the chunk of `within` that the top-64 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 6000, the same denominator as the other two, which is what lets all three add to 100%.
- **31.5%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-64): 0.664** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 64 subspace directions recover about 66%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 26 / median 37 / max 40** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=65 = a truncation warning. Your max is 40, below the cap → no cluster truncated, d=64 has headroom.

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

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.27%–2.23% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-64) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 63 | 1,918,725 | 2.23% | 0.707 | 32 | 138 | 274 | 49.8% | 0.001 | 0.583 |
| 83 | 1,486,987 | 1.73% | 0.703 | 33 | 107 | 216 | 49.2% | 0.002 | 0.668 |
| 110 | 1,274,386 | 1.48% | 0.743 | 30 | 92 | 184 | 49.4% | 0.001 | 0.727 |
| 106 | 1,222,783 | 1.42% | 0.720 | 31 | 88 | 176 | 49.2% | 0.003 | 0.650 |
| 93 | 1,168,373 | 1.36% | 0.726 | 32 | 84 | 167 | 50.0% | 0.000 | 0.635 |
| 52 | 1,132,871 | 1.32% | 0.670 | 34 | 174 | 225 | 36.2% | 0.041 | 0.784 |
| 46 | 1,098,584 | 1.28% | 0.665 | 34 | 167 | 190 | 28.3% | 0.079 | 0.741 |
| 51 | 1,074,464 | 1.25% | 0.750 | 30 | 81 | 171 | 47.1% | 0.010 | 0.727 |
| 101 | 1,062,764 | 1.24% | 0.756 | 30 | 77 | 156 | 48.3% | 0.009 | 0.727 |
| 56 | 1,044,311 | 1.21% | 0.653 | 37 | 110 | 189 | 39.6% | 0.030 | 0.804 |
| 11 | 1,017,626 | 1.18% | 0.633 | 39 | 81 | 145 | 43.7% | 0.047 | 0.781 |
| 98 | 959,470 | 1.12% | 0.693 | 29 | 199 | 120 | 27.0% | 0.101 | 0.669 |
| 72 | 950,158 | 1.10% | 0.740 | 31 | 79 | 172 | 46.2% | 0.011 | 0.705 |
| 66 | 945,137 | 1.10% | 0.688 | 36 | 81 | 172 | 43.8% | 0.014 | 0.786 |
| 78 | 940,085 | 1.09% | 0.699 | 35 | 68 | 138 | 48.7% | 0.006 | 0.663 |
| 48 | 939,768 | 1.09% | 0.639 | 38 | 119 | 179 | 39.4% | 0.039 | 0.812 |
| 116 | 937,064 | 1.09% | 0.652 | 37 | 148 | 213 | 40.6% | 0.031 | 0.812 |
| 39 | 934,519 | 1.09% | 0.668 | 35 | 118 | 214 | 40.1% | 0.021 | 0.786 |
| 103 | 929,690 | 1.08% | 0.643 | 36 | 67 | 133 | 48.3% | 0.008 | 0.634 |
| 69 | 922,261 | 1.07% | 0.630 | 38 | 166 | 135 | 25.2% | 0.150 | 0.771 |
| 85 | 908,379 | 1.06% | 0.652 | 36 | 183 | 158 | 29.1% | 0.088 | 0.755 |
| 59 | 901,505 | 1.05% | 0.671 | 34 | 67 | 138 | 39.6% | 0.031 | 0.651 |
| 9 | 877,705 | 1.02% | 0.679 | 34 | 65 | 129 | 44.6% | 0.023 | 0.665 |
| 57 | 875,046 | 1.02% | 0.762 | 30 | 64 | 128 | 48.7% | 0.005 | 0.694 |
| 109 | 874,598 | 1.02% | 0.677 | 33 | 172 | 131 | 34.4% | 0.062 | 0.718 |
| 62 | 857,744 | 1.00% | 0.622 | 38 | 122 | 137 | 31.6% | 0.110 | 0.751 |
| 29 | 852,262 | 0.99% | 0.736 | 31 | 73 | 173 | 44.7% | 0.007 | 0.727 |
| 32 | 847,055 | 0.98% | 0.711 | 31 | 74 | 141 | 38.7% | 0.052 | 0.685 |
| 117 | 842,398 | 0.98% | 0.639 | 39 | 67 | 136 | 42.4% | 0.070 | 0.800 |
| 107 | 820,817 | 0.95% | 0.637 | 38 | 95 | 140 | 32.3% | 0.042 | 0.781 |
| 100 | 818,944 | 0.95% | 0.656 | 35 | 80 | 136 | 40.1% | 0.058 | 0.749 |
| 86 | 814,714 | 0.95% | 0.709 | 33 | 59 | 124 | 47.0% | 0.006 | 0.685 |
| 102 | 812,059 | 0.94% | 0.654 | 36 | 193 | 153 | 19.6% | 0.126 | 0.738 |
| 31 | 804,554 | 0.94% | 0.658 | 34 | 65 | 120 | 32.4% | 0.063 | 0.670 |
| 127 | 799,621 | 0.93% | 0.732 | 34 | 58 | 116 | 49.4% | 0.002 | 0.639 |
| 50 | 788,262 | 0.92% | 0.721 | 30 | 129 | 127 | 37.2% | 0.036 | 0.725 |
| 97 | 777,356 | 0.90% | 0.643 | 36 | 56 | 116 | 47.7% | 0.008 | 0.700 |
| 47 | 775,248 | 0.90% | 0.647 | 37 | 70 | 121 | 32.2% | 0.092 | 0.751 |
| 71 | 765,330 | 0.89% | 0.664 | 33 | 168 | 101 | 25.4% | 0.106 | 0.687 |
| 35 | 747,598 | 0.87% | 0.734 | 32 | 56 | 110 | 47.6% | 0.015 | 0.675 |
| 64 | 747,270 | 0.87% | 0.620 | 38 | 109 | 172 | 31.6% | 0.049 | 0.747 |
| 124 | 742,722 | 0.86% | 0.702 | 30 | 182 | 15 | 27.2% | 0.065 | 0.718 |
| 95 | 741,636 | 0.86% | 0.687 | 32 | 201 | 11 | 40.2% | 0.026 | 0.729 |
| 20 | 738,476 | 0.86% | 0.620 | 39 | 155 | 91 | 34.8% | 0.042 | 0.796 |
| 112 | 731,150 | 0.85% | 0.651 | 35 | 303 | 26 | 37.9% | 0.054 | 0.724 |
| 113 | 729,151 | 0.85% | 0.622 | 38 | 138 | 177 | 22.6% | 0.185 | 0.796 |
| 90 | 718,816 | 0.84% | 0.661 | 34 | 228 | 24 | 42.3% | 0.018 | 0.729 |
| 4 | 717,035 | 0.83% | 0.645 | 36 | 57 | 111 | 34.6% | 0.060 | 0.696 |
| 111 | 711,978 | 0.83% | 0.630 | 37 | 210 | 56 | 22.5% | 0.089 | 0.767 |
| 88 | 710,010 | 0.83% | 0.733 | 36 | 51 | 102 | 49.7% | 0.001 | 0.598 |
| 67 | 704,235 | 0.82% | 0.639 | 38 | 96 | 135 | 21.9% | 0.160 | 0.781 |
| 89 | 704,231 | 0.82% | 0.730 | 32 | 51 | 109 | 48.5% | 0.003 | 0.675 |
| 65 | 702,052 | 0.82% | 0.612 | 39 | 123 | 123 | 31.8% | 0.139 | 0.751 |
| 7 | 701,303 | 0.82% | 0.618 | 38 | 76 | 111 | 39.4% | 0.088 | 0.760 |
| 41 | 696,331 | 0.81% | 0.709 | 37 | 50 | 103 | 48.7% | 0.011 | 0.535 |
| 18 | 693,277 | 0.81% | 0.648 | 38 | 53 | 100 | 42.6% | 0.042 | 0.755 |
| 42 | 688,684 | 0.80% | 0.618 | 39 | 91 | 141 | 27.4% | 0.142 | 0.804 |
| 125 | 686,272 | 0.80% | 0.786 | 34 | 50 | 98 | 50.0% | 0.000 | 0.569 |
| 8 | 685,484 | 0.80% | 0.600 | 40 | 57 | 121 | 43.1% | 0.030 | 0.702 |
| 10 | 682,039 | 0.79% | 0.628 | 38 | 72 | 101 | 25.9% | 0.095 | 0.707 |
| 37 | 676,727 | 0.79% | 0.683 | 29 | 206 | 14 | 35.8% | 0.042 | 0.669 |
| 15 | 675,918 | 0.79% | 0.625 | 37 | 162 | 82 | 17.6% | 0.134 | 0.738 |
| 24 | 668,429 | 0.78% | 0.607 | 39 | 116 | 107 | 34.4% | 0.082 | 0.747 |
| 81 | 661,226 | 0.77% | 0.657 | 38 | 57 | 113 | 38.8% | 0.025 | 0.804 |
| 1 | 651,642 | 0.76% | 0.653 | 35 | 53 | 109 | 42.5% | 0.025 | 0.703 |
| 5 | 651,036 | 0.76% | 0.663 | 32 | 222 | 39 | 34.5% | 0.060 | 0.695 |
| 61 | 646,578 | 0.75% | 0.613 | 39 | 134 | 114 | 34.9% | 0.050 | 0.749 |
| 23 | 645,866 | 0.75% | 0.724 | 34 | 47 | 95 | 48.6% | 0.005 | 0.651 |
| 114 | 638,915 | 0.74% | 0.645 | 37 | 66 | 78 | 27.4% | 0.102 | 0.707 |
| 80 | 632,312 | 0.74% | 0.615 | 39 | 118 | 48 | 29.9% | 0.081 | 0.746 |
| 36 | 627,398 | 0.73% | 0.626 | 38 | 137 | 101 | 27.0% | 0.062 | 0.754 |
| 115 | 626,395 | 0.73% | 0.602 | 39 | 245 | 56 | 32.9% | 0.070 | 0.748 |
| 49 | 624,959 | 0.73% | 0.628 | 38 | 147 | 64 | 36.8% | 0.111 | 0.757 |
| 21 | 620,211 | 0.72% | 0.603 | 39 | 68 | 89 | 28.1% | 0.076 | 0.682 |
| 73 | 613,484 | 0.71% | 0.606 | 40 | 161 | 60 | 21.5% | 0.122 | 0.748 |
| 105 | 610,363 | 0.71% | 0.615 | 39 | 130 | 117 | 23.9% | 0.105 | 0.747 |
| 0 | 606,053 | 0.70% | 0.630 | 39 | 50 | 108 | 39.2% | 0.046 | 0.717 |
| 27 | 600,238 | 0.70% | 0.589 | 38 | 80 | 52 | 22.8% | 0.135 | 0.696 |
| 126 | 600,196 | 0.70% | 0.640 | 36 | 100 | 119 | 19.6% | 0.128 | 0.717 |
| 25 | 599,154 | 0.70% | 0.644 | 38 | 62 | 122 | 28.8% | 0.077 | 0.770 |
| 6 | 587,228 | 0.68% | 0.662 | 35 | 43 | 85 | 47.2% | 0.010 | 0.700 |
| 60 | 587,073 | 0.68% | 0.639 | 38 | 110 | 45 | 18.6% | 0.257 | 0.804 |
| 68 | 582,050 | 0.68% | 0.620 | 39 | 79 | 117 | 25.6% | 0.106 | 0.748 |
| 45 | 577,748 | 0.67% | 0.603 | 39 | 164 | 37 | 32.9% | 0.125 | 0.751 |
| 123 | 565,587 | 0.66% | 0.730 | 26 | 308 | 5 | 34.5% | 0.044 | 0.659 |
| 54 | 564,603 | 0.66% | 0.649 | 37 | 44 | 92 | 45.5% | 0.024 | 0.753 |
| 2 | 558,076 | 0.65% | 0.613 | 40 | 51 | 81 | 30.0% | 0.078 | 0.712 |
| 76 | 535,208 | 0.62% | 0.688 | 34 | 39 | 92 | 40.7% | 0.035 | 0.536 |
| 87 | 528,107 | 0.61% | 0.629 | 38 | 115 | 69 | 20.0% | 0.121 | 0.770 |
| 14 | 523,148 | 0.61% | 0.611 | 38 | 117 | 49 | 23.4% | 0.166 | 0.750 |
| 43 | 505,074 | 0.59% | 0.619 | 38 | 81 | 94 | 22.0% | 0.121 | 0.735 |
| 91 | 502,507 | 0.58% | 0.676 | 36 | 41 | 87 | 43.5% | 0.022 | 0.724 |
| 44 | 498,054 | 0.58% | 0.649 | 34 | 356 | 0 | 36.3% | 0.038 | 0.703 |
| 16 | 496,676 | 0.58% | 0.630 | 36 | 272 | 2 | 29.0% | 0.075 | 0.712 |
| 104 | 486,441 | 0.57% | 0.643 | 38 | 35 | 74 | 47.2% | 0.008 | 0.639 |
| 77 | 483,264 | 0.56% | 0.643 | 37 | 40 | 89 | 35.9% | 0.061 | 0.696 |
| 92 | 473,739 | 0.55% | 0.650 | 39 | 38 | 74 | 41.4% | 0.039 | 0.771 |
| 30 | 467,843 | 0.54% | 0.659 | 37 | 37 | 69 | 39.4% | 0.069 | 0.718 |
| 99 | 461,207 | 0.54% | 0.627 | 37 | 317 | 0 | 35.4% | 0.057 | 0.728 |
| 17 | 459,392 | 0.53% | 0.612 | 40 | 55 | 85 | 27.0% | 0.100 | 0.697 |
| 3 | 458,102 | 0.53% | 0.630 | 39 | 54 | 61 | 25.9% | 0.097 | 0.749 |
| 33 | 456,537 | 0.53% | 0.660 | 37 | 35 | 73 | 46.0% | 0.014 | 0.753 |
| 122 | 447,584 | 0.52% | 0.618 | 37 | 127 | 9 | 28.3% | 0.118 | 0.719 |
| 70 | 445,827 | 0.52% | 0.654 | 37 | 33 | 69 | 45.1% | 0.026 | 0.682 |
| 26 | 424,587 | 0.49% | 0.585 | 39 | 270 | 1 | 30.2% | 0.039 | 0.708 |
| 120 | 415,951 | 0.48% | 0.673 | 35 | 30 | 62 | 48.0% | 0.006 | 0.589 |
| 108 | 415,844 | 0.48% | 0.624 | 37 | 75 | 49 | 18.6% | 0.121 | 0.694 |
| 118 | 415,703 | 0.48% | 0.632 | 35 | 221 | 1 | 18.0% | 0.118 | 0.719 |
| 58 | 401,499 | 0.47% | 0.640 | 37 | 33 | 67 | 38.6% | 0.030 | 0.689 |
| 34 | 398,999 | 0.46% | 0.689 | 38 | 30 | 65 | 46.6% | 0.014 | 0.757 |
| 19 | 395,809 | 0.46% | 0.625 | 39 | 32 | 68 | 36.3% | 0.056 | 0.688 |
| 13 | 386,984 | 0.45% | 0.613 | 39 | 66 | 51 | 24.1% | 0.130 | 0.731 |
| 22 | 386,297 | 0.45% | 0.660 | 38 | 29 | 57 | 45.4% | 0.013 | 0.717 |
| 38 | 372,010 | 0.43% | 0.647 | 38 | 27 | 53 | 48.9% | 0.005 | 0.553 |
| 53 | 368,162 | 0.43% | 0.652 | 36 | 48 | 29 | 20.4% | 0.112 | 0.770 |
| 75 | 353,740 | 0.41% | 0.630 | 40 | 27 | 52 | 40.3% | 0.035 | 0.702 |
| 40 | 346,709 | 0.40% | 0.639 | 37 | 41 | 35 | 23.8% | 0.099 | 0.721 |
| 121 | 342,763 | 0.40% | 0.615 | 40 | 26 | 60 | 44.1% | 0.045 | 0.731 |
| 79 | 339,447 | 0.39% | 0.640 | 40 | 25 | 48 | 47.9% | 0.012 | 0.650 |
| 119 | 334,847 | 0.39% | 0.618 | 40 | 25 | 54 | 44.2% | 0.023 | 0.694 |
| 94 | 323,866 | 0.38% | 0.683 | 35 | 24 | 49 | 44.9% | 0.020 | 0.633 |
| 96 | 303,176 | 0.35% | 0.633 | 39 | 23 | 48 | 44.1% | 0.038 | 0.661 |
| 12 | 294,649 | 0.34% | 0.648 | 35 | 56 | 6 | 17.9% | 0.144 | 0.670 |
| 84 | 289,708 | 0.34% | 0.673 | 37 | 21 | 44 | 45.1% | 0.022 | 0.666 |
| 55 | 287,160 | 0.33% | 0.654 | 35 | 53 | 19 | 17.4% | 0.139 | 0.636 |
| 28 | 256,104 | 0.30% | 0.677 | 35 | 41 | 5 | 17.4% | 0.142 | 0.594 |
| 82 | 245,049 | 0.28% | 0.656 | 38 | 26 | 54 | 31.1% | 0.085 | 0.638 |
| 74 | 235,389 | 0.27% | 0.637 | 39 | 18 | 35 | 45.5% | 0.012 | 0.661 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **44, 99**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v8_seed2_d64 --out runs/clustering/v8_seed2_d64/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure; see `runs/clustering/v8_seed2_d64/temporal_report.md`.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 64 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.424, mean 0.435, max 0.812 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 48 ↔ 116 | 0.812 | 0.668 |
| 42 ↔ 60 | 0.804 | 0.577 |
| 56 ↔ 81 | 0.804 | 0.779 |
| 42 ↔ 117 | 0.800 | 0.688 |
| 20 ↔ 113 | 0.796 | 0.708 |
| 39 ↔ 66 | 0.786 | 0.783 |
| 39 ↔ 52 | 0.784 | 0.767 |
| 60 ↔ 67 | 0.781 | 0.338 |
| 11 ↔ 107 | 0.781 | 0.651 |
| 67 ↔ 92 | 0.771 | 0.606 |
| 69 ↔ 113 | 0.771 | 0.663 |
| 53 ↔ 87 | 0.770 | 0.369 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
