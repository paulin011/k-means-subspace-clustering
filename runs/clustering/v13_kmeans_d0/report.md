# Clustering report (subspace_kmeans) — `runs/clustering/v13_kmeans_d0`

*Generated 2026-09-14 15:18 by `analyze_clusters.py`. K=128 point clusters in 2048-dim token space, 86,016,000 tokens.*

## Overview

The model groups the 86,016,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 128 point clusters, each summarised by a single centroid. A token is assigned to the nearest centroid (plain squared distance).

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
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
| dim | 0 |
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
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v13_kmeans_d0/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v13_kmeans_d0/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7659.20 | 100.00% | 127 | 23,702,502 |
| 2 | 5268.38 | 67.64% | 6,236 | 4,481,441 |
| 3 | 5119.52 | 31.40% | 6,424 | 1,647,676 |
| 4 | 5082.36 | 17.15% | 6,603 | 1,375,513 |
| 5 | 5065.63 | 12.09% | 6,691 | 1,447,285 |
| 6 | 5055.93 | 9.38% | 6,720 | 1,475,958 |
| 7 | 5049.63 | 7.65% | 6,731 | 1,483,301 |
| 8 | 5045.13 | 6.49% | 6,739 | 1,477,409 |
| 9 | 5041.70 | 5.66% | 6,744 | 1,461,925 |
| 10 | 5039.03 | 5.03% | 6,752 | 1,439,076 |
| 11 | 5036.84 | 4.53% | 6,758 | 1,410,882 |
| 12 | 5035.00 | 4.13% | 6,763 | 1,379,538 |
| 13 | 5033.44 | 3.78% | 6,776 | 1,346,730 |
| 14 | 5032.13 | 3.48% | 6,780 | 1,312,582 |
| 15 | 5030.98 | 3.23% | 6,781 | 1,279,100 |
| 16 | 5029.97 | 3.02% | 6,782 | 1,283,804 |
| 17 | 5029.09 | 2.83% | 6,785 | 1,285,922 |
| 18 | 5028.29 | 2.68% | 6,792 | 1,286,427 |
| 19 | 5027.58 | 2.54% | 6,800 | 1,286,632 |
| 20 | 5026.91 | 2.42% | 6,802 | 1,285,453 |
| 21 | 5026.30 | 2.31% | 6,804 | 1,283,115 |
| 22 | 5025.74 | 2.22% | 6,807 | 1,279,186 |
| 23 | 5025.23 | 2.13% | 6,810 | 1,275,041 |
| 24 | 5024.72 | 2.05% | 6,809 | 1,270,757 |
| 25 | 5024.28 | 1.99% | 6,808 | 1,265,676 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5999**, split into:

- **16.2%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **83.8%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.01%–1.46% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | cells@50% | owned | files@50% | tCV |
|---|---|---|---|---|---|---|
| 25 | 1,259,517 | 1.46% | 688 | 140 | 24.4% | 0.183 |
| 96 | 1,121,268 | 1.30% | 1791 | 21 | 38.1% | 0.046 |
| 72 | 1,009,782 | 1.17% | 511 | 135 | 31.0% | 0.131 |
| 80 | 1,008,067 | 1.17% | 359 | 533 | 21.6% | 0.127 |
| 81 | 1,008,056 | 1.17% | 1683 | 53 | 36.9% | 0.032 |
| 66 | 1,007,972 | 1.17% | 255 | 214 | 29.2% | 0.122 |
| 59 | 1,007,918 | 1.17% | 1859 | 32 | 37.0% | 0.057 |
| 74 | 1,006,790 | 1.17% | 1618 | 12 | 36.5% | 0.075 |
| 87 | 975,468 | 1.13% | 1229 | 187 | 39.8% | 0.036 |
| 27 | 953,576 | 1.11% | 1352 | 77 | 36.6% | 0.064 |
| 34 | 943,441 | 1.10% | 1198 | 60 | 34.1% | 0.062 |
| 86 | 941,684 | 1.09% | 937 | 203 | 38.8% | 0.054 |
| 20 | 901,266 | 1.05% | 768 | 210 | 27.6% | 0.089 |
| 16 | 892,034 | 1.04% | 1512 | 13 | 35.5% | 0.046 |
| 6 | 888,232 | 1.03% | 1020 | 375 | 35.1% | 0.068 |
| 32 | 881,443 | 1.02% | 1209 | 39 | 35.5% | 0.064 |
| 99 | 867,730 | 1.01% | 1557 | 17 | 37.3% | 0.066 |
| 103 | 861,395 | 1.00% | 1029 | 11 | 35.7% | 0.045 |
| 105 | 859,153 | 1.00% | 930 | 90 | 30.0% | 0.085 |
| 70 | 858,587 | 1.00% | 872 | 407 | 40.6% | 0.029 |
| 109 | 855,236 | 0.99% | 606 | 270 | 25.8% | 0.054 |
| 37 | 849,069 | 0.99% | 837 | 190 | 38.2% | 0.079 |
| 33 | 843,623 | 0.98% | 1232 | 99 | 36.0% | 0.042 |
| 107 | 841,016 | 0.98% | 1131 | 77 | 37.6% | 0.038 |
| 68 | 830,851 | 0.97% | 1838 | 4 | 34.8% | 0.066 |
| 77 | 828,107 | 0.96% | 147 | 192 | 24.2% | 0.197 |
| 42 | 826,526 | 0.96% | 872 | 328 | 39.4% | 0.045 |
| 41 | 820,459 | 0.95% | 2163 | 7 | 38.8% | 0.034 |
| 1 | 793,526 | 0.92% | 709 | 241 | 32.9% | 0.082 |
| 4 | 780,869 | 0.91% | 872 | 308 | 39.4% | 0.063 |
| 115 | 779,102 | 0.91% | 747 | 63 | 34.5% | 0.056 |
| 114 | 767,860 | 0.89% | 176 | 225 | 31.4% | 0.099 |
| 78 | 756,737 | 0.88% | 861 | 188 | 40.8% | 0.025 |
| 75 | 756,636 | 0.88% | 410 | 300 | 31.1% | 0.093 |
| 85 | 741,454 | 0.86% | 437 | 94 | 33.6% | 0.040 |
| 17 | 732,671 | 0.85% | 830 | 55 | 39.5% | 0.046 |
| 67 | 731,140 | 0.85% | 849 | 91 | 33.5% | 0.090 |
| 39 | 729,705 | 0.85% | 1057 | 31 | 37.9% | 0.037 |
| 8 | 727,512 | 0.85% | 1111 | 46 | 35.3% | 0.054 |
| 21 | 726,793 | 0.84% | 1526 | 0 | 38.6% | 0.041 |
| 14 | 726,360 | 0.84% | 1217 | 1 | 35.2% | 0.046 |
| 117 | 723,546 | 0.84% | 1117 | 55 | 34.6% | 0.073 |
| 51 | 717,978 | 0.83% | 909 | 22 | 36.9% | 0.046 |
| 88 | 717,090 | 0.83% | 799 | 184 | 37.7% | 0.039 |
| 93 | 716,848 | 0.83% | 589 | 276 | 40.9% | 0.026 |
| 46 | 711,352 | 0.83% | 1625 | 0 | 30.3% | 0.074 |
| 2 | 709,581 | 0.82% | 1327 | 8 | 38.1% | 0.033 |
| 82 | 709,314 | 0.82% | 693 | 259 | 29.1% | 0.141 |
| 69 | 705,902 | 0.82% | 812 | 79 | 35.5% | 0.039 |
| 58 | 705,525 | 0.82% | 968 | 26 | 38.4% | 0.041 |
| 10 | 705,270 | 0.82% | 1096 | 65 | 35.4% | 0.055 |
| 26 | 698,288 | 0.81% | 1339 | 0 | 38.6% | 0.043 |
| 54 | 693,339 | 0.81% | 549 | 75 | 37.0% | 0.099 |
| 36 | 692,359 | 0.80% | 922 | 116 | 36.0% | 0.079 |
| 11 | 686,573 | 0.80% | 806 | 95 | 28.5% | 0.073 |
| 106 | 683,163 | 0.79% | 887 | 51 | 39.1% | 0.057 |
| 7 | 681,095 | 0.79% | 687 | 34 | 37.2% | 0.112 |
| 29 | 680,691 | 0.79% | 1061 | 60 | 35.9% | 0.089 |
| 38 | 679,026 | 0.79% | 558 | 46 | 27.3% | 0.094 |
| 19 | 674,133 | 0.78% | 422 | 163 | 37.2% | 0.064 |
| 101 | 671,952 | 0.78% | 1208 | 29 | 36.4% | 0.030 |
| 124 | 662,234 | 0.77% | 1482 | 16 | 36.3% | 0.042 |
| 91 | 659,987 | 0.77% | 773 | 89 | 33.3% | 0.066 |
| 55 | 658,410 | 0.77% | 1073 | 0 | 38.3% | 0.028 |
| 15 | 657,973 | 0.76% | 832 | 80 | 39.1% | 0.036 |
| 22 | 657,352 | 0.76% | 426 | 106 | 34.0% | 0.092 |
| 56 | 654,063 | 0.76% | 1196 | 5 | 39.6% | 0.038 |
| 48 | 649,614 | 0.76% | 668 | 80 | 37.9% | 0.084 |
| 98 | 644,284 | 0.75% | 520 | 74 | 37.3% | 0.060 |
| 45 | 643,835 | 0.75% | 1104 | 36 | 35.8% | 0.081 |
| 57 | 643,802 | 0.75% | 954 | 64 | 31.3% | 0.059 |
| 60 | 641,116 | 0.75% | 1027 | 116 | 29.9% | 0.077 |
| 9 | 640,338 | 0.74% | 1132 | 50 | 37.7% | 0.060 |
| 84 | 639,743 | 0.74% | 1643 | 0 | 38.7% | 0.046 |
| 5 | 634,071 | 0.74% | 1071 | 64 | 39.4% | 0.049 |
| 120 | 632,625 | 0.74% | 432 | 48 | 35.9% | 0.078 |
| 65 | 631,598 | 0.73% | 670 | 111 | 35.9% | 0.051 |
| 73 | 628,461 | 0.73% | 941 | 49 | 28.2% | 0.093 |
| 127 | 624,848 | 0.73% | 1195 | 15 | 37.4% | 0.053 |
| 64 | 616,108 | 0.72% | 325 | 118 | 31.6% | 0.137 |
| 108 | 614,906 | 0.71% | 922 | 29 | 36.3% | 0.051 |
| 95 | 613,761 | 0.71% | 425 | 260 | 29.2% | 0.086 |
| 35 | 610,493 | 0.71% | 1714 | 0 | 33.3% | 0.067 |
| 76 | 608,458 | 0.71% | 1065 | 3 | 32.3% | 0.057 |
| 50 | 608,344 | 0.71% | 563 | 65 | 40.8% | 0.036 |
| 121 | 605,036 | 0.70% | 695 | 182 | 29.8% | 0.061 |
| 110 | 603,712 | 0.70% | 649 | 202 | 29.9% | 0.085 |
| 30 | 593,856 | 0.69% | 192 | 207 | 20.9% | 0.192 |
| 90 | 592,591 | 0.69% | 891 | 80 | 30.1% | 0.071 |
| 44 | 592,584 | 0.69% | 470 | 50 | 31.7% | 0.053 |
| 47 | 586,151 | 0.68% | 301 | 196 | 31.5% | 0.107 |
| 23 | 578,567 | 0.67% | 843 | 58 | 34.5% | 0.059 |
| 12 | 575,772 | 0.67% | 640 | 5 | 39.2% | 0.032 |
| 40 | 573,177 | 0.67% | 528 | 157 | 32.9% | 0.139 |
| 119 | 569,331 | 0.66% | 778 | 10 | 36.3% | 0.049 |
| 71 | 566,435 | 0.66% | 169 | 176 | 24.4% | 0.109 |
| 83 | 557,771 | 0.65% | 987 | 1 | 33.1% | 0.038 |
| 123 | 554,839 | 0.65% | 328 | 69 | 38.5% | 0.045 |
| 125 | 544,787 | 0.63% | 802 | 146 | 36.5% | 0.030 |
| 52 | 530,290 | 0.62% | 568 | 132 | 33.5% | 0.055 |
| 113 | 519,684 | 0.60% | 1114 | 8 | 37.3% | 0.042 |
| 126 | 516,602 | 0.60% | 220 | 68 | 16.5% | 0.176 |
| 62 | 514,912 | 0.60% | 1159 | 21 | 37.9% | 0.041 |
| 100 | 512,932 | 0.60% | 647 | 40 | 23.9% | 0.111 |
| 0 | 511,451 | 0.59% | 809 | 0 | 35.6% | 0.029 |
| 79 | 510,543 | 0.59% | 272 | 170 | 29.3% | 0.078 |
| 18 | 505,167 | 0.59% | 1062 | 0 | 33.1% | 0.033 |
| 112 | 503,872 | 0.59% | 1053 | 50 | 35.1% | 0.057 |
| 61 | 496,306 | 0.58% | 355 | 124 | 31.5% | 0.051 |
| 94 | 492,961 | 0.57% | 1068 | 3 | 34.8% | 0.044 |
| 31 | 488,769 | 0.57% | 768 | 27 | 33.5% | 0.151 |
| 53 | 487,969 | 0.57% | 243 | 108 | 35.0% | 0.053 |
| 111 | 486,657 | 0.57% | 1036 | 2 | 35.6% | 0.039 |
| 97 | 484,570 | 0.56% | 250 | 153 | 31.2% | 0.040 |
| 118 | 481,289 | 0.56% | 353 | 111 | 23.3% | 0.105 |
| 13 | 480,621 | 0.56% | 756 | 4 | 39.5% | 0.060 |
| 3 | 477,719 | 0.56% | 129 | 102 | 26.3% | 0.107 |
| 89 | 475,080 | 0.55% | 210 | 181 | 21.0% | 0.111 |
| 102 | 457,854 | 0.53% | 355 | 144 | 21.8% | 0.078 |
| 63 | 455,784 | 0.53% | 929 | 3 | 34.9% | 0.051 |
| 104 | 438,645 | 0.51% | 129 | 123 | 21.9% | 0.155 |
| 49 | 426,695 | 0.50% | 38 | 90 | 40.2% | 0.038 |
| 24 | 403,774 | 0.47% | 670 | 39 | 25.6% | 0.194 |
| 43 | 361,072 | 0.42% | 372 | 93 | 21.3% | 0.134 |
| 122 | 350,209 | 0.41% | 804 | 0 | 32.6% | 0.080 |
| 116 | 323,097 | 0.38% | 69 | 102 | 15.1% | 0.143 |
| 28 | 281,979 | 0.33% | 99 | 90 | 11.5% | 0.227 |
| 92 | 6,808 | 0.01% | 1 | 1 | 48.4% | 0.010 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **0, 18, 21, 26, 35, 46, 55, 84, 122** under a quarter of the mean cluster size: **92**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v13_kmeans_d0 --out runs/clustering/v13_kmeans_d0/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
