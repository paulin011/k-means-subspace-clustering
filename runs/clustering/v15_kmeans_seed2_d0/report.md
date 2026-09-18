# Clustering report (subspace_kmeans) — `runs/clustering/v15_kmeans_seed2_d0`

*Generated 2026-09-14 16:33 by `analyze_clusters.py`. K=128 point clusters in 2048-dim token space, 86,016,000 tokens.*

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
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v15_kmeans_seed2_d0/sample.json --seed 2 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v15_kmeans_seed2_d0/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7414.86 | 100.00% | 301 | 35,852,088 |
| 2 | 5317.50 | 69.28% | 39,492 | 7,056,556 |
| 3 | 5125.82 | 35.88% | 80,350 | 2,188,676 |
| 4 | 5079.52 | 18.63% | 117,016 | 1,611,900 |
| 5 | 5060.88 | 12.45% | 144,358 | 1,514,985 |
| 6 | 5051.20 | 9.30% | 162,045 | 1,421,146 |
| 7 | 5045.10 | 7.56% | 173,659 | 1,346,074 |
| 8 | 5040.71 | 6.42% | 182,287 | 1,288,605 |
| 9 | 5037.40 | 5.54% | 189,431 | 1,245,266 |
| 10 | 5034.91 | 4.82% | 195,876 | 1,220,757 |
| 11 | 5032.98 | 4.25% | 201,305 | 1,223,661 |
| 12 | 5031.42 | 3.80% | 206,723 | 1,229,093 |
| 13 | 5030.15 | 3.45% | 210,470 | 1,234,021 |
| 14 | 5029.06 | 3.16% | 210,275 | 1,237,730 |
| 15 | 5028.13 | 2.94% | 209,866 | 1,239,712 |
| 16 | 5027.31 | 2.75% | 209,089 | 1,241,396 |
| 17 | 5026.61 | 2.59% | 208,322 | 1,242,946 |
| 18 | 5025.93 | 2.44% | 207,577 | 1,244,934 |
| 19 | 5025.35 | 2.32% | 206,843 | 1,247,141 |
| 20 | 5024.81 | 2.21% | 206,159 | 1,247,913 |
| 21 | 5024.34 | 2.11% | 205,501 | 1,249,378 |
| 22 | 5023.89 | 2.02% | 204,926 | 1,249,969 |
| 23 | 5023.47 | 1.94% | 204,411 | 1,249,306 |
| 24 | 5023.07 | 1.86% | 204,132 | 1,249,167 |
| 25 | 5022.72 | 1.80% | 204,001 | 1,248,550 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5999**, split into:

- **16.3%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **83.7%** residual, i.e. within-cluster (point clusters: no subspace basis, so nothing beyond the centroid is captured)

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 86,016,000.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 7000 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 128 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.24%–1.45% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | cells@50% | owned | files@50% | tCV |
|---|---|---|---|---|---|---|
| 88 | 1,247,234 | 1.45% | 1794 | 35 | 37.5% | 0.056 |
| 97 | 1,099,307 | 1.28% | 735 | 340 | 26.9% | 0.079 |
| 39 | 1,095,287 | 1.27% | 1392 | 151 | 38.3% | 0.062 |
| 109 | 1,040,747 | 1.21% | 1045 | 104 | 35.9% | 0.042 |
| 120 | 1,015,404 | 1.18% | 1949 | 22 | 37.3% | 0.057 |
| 78 | 1,003,636 | 1.17% | 1139 | 187 | 35.8% | 0.053 |
| 110 | 978,166 | 1.14% | 783 | 170 | 29.6% | 0.119 |
| 40 | 955,951 | 1.11% | 1412 | 56 | 38.3% | 0.028 |
| 32 | 950,035 | 1.10% | 460 | 156 | 26.9% | 0.139 |
| 29 | 938,383 | 1.09% | 1215 | 269 | 34.8% | 0.039 |
| 116 | 934,071 | 1.09% | 1416 | 35 | 37.4% | 0.042 |
| 4 | 928,320 | 1.08% | 1223 | 140 | 37.9% | 0.066 |
| 58 | 928,156 | 1.08% | 1434 | 38 | 39.3% | 0.050 |
| 76 | 918,541 | 1.07% | 1055 | 67 | 39.9% | 0.059 |
| 34 | 897,220 | 1.04% | 1040 | 205 | 39.6% | 0.029 |
| 18 | 896,603 | 1.04% | 150 | 162 | 24.5% | 0.177 |
| 11 | 894,001 | 1.04% | 353 | 270 | 35.0% | 0.099 |
| 48 | 886,289 | 1.03% | 1389 | 15 | 37.7% | 0.051 |
| 63 | 855,367 | 0.99% | 174 | 190 | 17.4% | 0.275 |
| 31 | 847,932 | 0.99% | 1161 | 115 | 36.2% | 0.037 |
| 100 | 842,227 | 0.98% | 1549 | 0 | 35.5% | 0.053 |
| 7 | 841,390 | 0.98% | 1241 | 71 | 35.6% | 0.062 |
| 66 | 839,997 | 0.98% | 1191 | 77 | 38.8% | 0.025 |
| 56 | 835,094 | 0.97% | 1027 | 66 | 36.3% | 0.041 |
| 62 | 829,199 | 0.96% | 892 | 256 | 35.8% | 0.116 |
| 47 | 828,558 | 0.96% | 926 | 151 | 36.7% | 0.071 |
| 82 | 821,482 | 0.96% | 1155 | 31 | 34.0% | 0.074 |
| 23 | 813,732 | 0.95% | 1643 | 25 | 38.9% | 0.074 |
| 84 | 811,527 | 0.94% | 312 | 321 | 22.5% | 0.114 |
| 114 | 805,665 | 0.94% | 1360 | 69 | 37.3% | 0.074 |
| 9 | 799,209 | 0.93% | 1046 | 81 | 33.1% | 0.054 |
| 20 | 798,955 | 0.93% | 989 | 2 | 33.9% | 0.064 |
| 104 | 789,228 | 0.92% | 328 | 353 | 29.3% | 0.058 |
| 57 | 774,391 | 0.90% | 747 | 346 | 40.4% | 0.019 |
| 77 | 765,805 | 0.89% | 1758 | 7 | 37.7% | 0.059 |
| 55 | 764,807 | 0.89% | 1202 | 42 | 40.6% | 0.021 |
| 83 | 761,769 | 0.89% | 337 | 359 | 19.6% | 0.133 |
| 89 | 756,276 | 0.88% | 840 | 253 | 39.6% | 0.031 |
| 102 | 755,927 | 0.88% | 1347 | 35 | 37.1% | 0.051 |
| 69 | 751,402 | 0.87% | 1383 | 38 | 33.8% | 0.065 |
| 74 | 746,923 | 0.87% | 911 | 46 | 25.7% | 0.095 |
| 41 | 739,912 | 0.86% | 1735 | 4 | 37.2% | 0.031 |
| 42 | 739,351 | 0.86% | 538 | 187 | 31.1% | 0.085 |
| 107 | 735,307 | 0.85% | 424 | 126 | 34.6% | 0.045 |
| 51 | 733,190 | 0.85% | 1110 | 88 | 39.9% | 0.027 |
| 17 | 727,698 | 0.85% | 681 | 154 | 37.5% | 0.038 |
| 37 | 710,370 | 0.83% | 1841 | 4 | 36.8% | 0.045 |
| 35 | 709,444 | 0.82% | 882 | 175 | 37.8% | 0.053 |
| 2 | 706,560 | 0.82% | 937 | 128 | 35.6% | 0.044 |
| 52 | 704,297 | 0.82% | 828 | 201 | 38.9% | 0.034 |
| 81 | 702,468 | 0.82% | 605 | 268 | 27.1% | 0.099 |
| 30 | 702,423 | 0.82% | 125 | 169 | 34.0% | 0.058 |
| 91 | 698,913 | 0.81% | 1154 | 82 | 39.2% | 0.055 |
| 111 | 696,563 | 0.81% | 1275 | 62 | 35.4% | 0.054 |
| 73 | 694,629 | 0.81% | 861 | 38 | 36.6% | 0.051 |
| 96 | 685,712 | 0.80% | 790 | 137 | 39.4% | 0.032 |
| 125 | 675,929 | 0.79% | 733 | 32 | 39.3% | 0.038 |
| 85 | 669,376 | 0.78% | 505 | 288 | 37.3% | 0.063 |
| 6 | 667,000 | 0.78% | 839 | 25 | 35.3% | 0.051 |
| 24 | 665,930 | 0.77% | 768 | 111 | 37.2% | 0.056 |
| 3 | 665,669 | 0.77% | 1132 | 48 | 34.3% | 0.113 |
| 75 | 664,530 | 0.77% | 241 | 206 | 28.2% | 0.110 |
| 0 | 664,131 | 0.77% | 1836 | 0 | 33.1% | 0.069 |
| 68 | 663,386 | 0.77% | 914 | 64 | 35.2% | 0.040 |
| 80 | 654,859 | 0.76% | 1348 | 22 | 36.9% | 0.039 |
| 79 | 654,465 | 0.76% | 467 | 72 | 36.1% | 0.060 |
| 25 | 651,643 | 0.76% | 1482 | 0 | 37.3% | 0.037 |
| 90 | 649,010 | 0.75% | 1043 | 36 | 36.8% | 0.038 |
| 54 | 647,115 | 0.75% | 1103 | 149 | 37.9% | 0.055 |
| 60 | 643,907 | 0.75% | 446 | 152 | 30.8% | 0.140 |
| 19 | 639,978 | 0.74% | 628 | 117 | 35.5% | 0.055 |
| 106 | 639,899 | 0.74% | 301 | 152 | 25.8% | 0.098 |
| 105 | 637,202 | 0.74% | 788 | 102 | 30.9% | 0.072 |
| 87 | 636,865 | 0.74% | 1215 | 5 | 36.9% | 0.071 |
| 36 | 633,353 | 0.74% | 1173 | 26 | 34.4% | 0.061 |
| 117 | 622,903 | 0.72% | 310 | 141 | 26.5% | 0.108 |
| 70 | 612,341 | 0.71% | 501 | 112 | 36.1% | 0.046 |
| 127 | 604,907 | 0.70% | 617 | 12 | 39.3% | 0.029 |
| 95 | 604,730 | 0.70% | 835 | 74 | 36.9% | 0.037 |
| 27 | 602,378 | 0.70% | 735 | 80 | 29.6% | 0.201 |
| 124 | 600,381 | 0.70% | 1209 | 36 | 34.7% | 0.064 |
| 65 | 599,640 | 0.70% | 755 | 47 | 35.0% | 0.060 |
| 71 | 597,163 | 0.69% | 479 | 201 | 23.3% | 0.113 |
| 64 | 590,734 | 0.69% | 373 | 188 | 20.5% | 0.113 |
| 26 | 587,354 | 0.68% | 321 | 155 | 32.2% | 0.095 |
| 101 | 586,451 | 0.68% | 512 | 95 | 38.9% | 0.052 |
| 94 | 584,822 | 0.68% | 976 | 32 | 33.7% | 0.049 |
| 119 | 581,531 | 0.68% | 1110 | 6 | 32.4% | 0.114 |
| 15 | 580,969 | 0.68% | 718 | 48 | 32.7% | 0.136 |
| 49 | 579,917 | 0.67% | 1417 | 9 | 37.0% | 0.036 |
| 72 | 578,796 | 0.67% | 1048 | 0 | 34.4% | 0.038 |
| 10 | 577,768 | 0.67% | 214 | 162 | 33.8% | 0.043 |
| 86 | 573,282 | 0.67% | 1860 | 0 | 38.1% | 0.065 |
| 59 | 562,944 | 0.65% | 938 | 40 | 30.3% | 0.063 |
| 45 | 559,412 | 0.65% | 731 | 47 | 35.2% | 0.043 |
| 121 | 557,307 | 0.65% | 566 | 49 | 35.8% | 0.091 |
| 67 | 556,480 | 0.65% | 365 | 67 | 30.1% | 0.056 |
| 50 | 554,690 | 0.64% | 961 | 1 | 32.2% | 0.052 |
| 112 | 547,251 | 0.64% | 930 | 9 | 36.6% | 0.042 |
| 126 | 542,273 | 0.63% | 901 | 68 | 34.3% | 0.068 |
| 21 | 533,878 | 0.62% | 180 | 172 | 32.1% | 0.060 |
| 53 | 529,423 | 0.62% | 171 | 137 | 28.2% | 0.127 |
| 61 | 526,807 | 0.61% | 183 | 48 | 26.0% | 0.089 |
| 46 | 525,883 | 0.61% | 1402 | 6 | 35.4% | 0.071 |
| 8 | 525,145 | 0.61% | 712 | 9 | 32.4% | 0.144 |
| 118 | 517,505 | 0.60% | 712 | 17 | 39.1% | 0.066 |
| 93 | 516,113 | 0.60% | 878 | 27 | 36.4% | 0.054 |
| 13 | 505,278 | 0.59% | 565 | 51 | 25.4% | 0.088 |
| 92 | 503,923 | 0.59% | 136 | 175 | 19.0% | 0.181 |
| 14 | 503,160 | 0.58% | 888 | 12 | 26.0% | 0.093 |
| 98 | 498,410 | 0.58% | 1022 | 40 | 34.5% | 0.032 |
| 33 | 495,417 | 0.58% | 1001 | 28 | 38.4% | 0.045 |
| 115 | 489,670 | 0.57% | 560 | 106 | 36.2% | 0.104 |
| 44 | 489,278 | 0.57% | 443 | 1 | 37.2% | 0.037 |
| 113 | 470,980 | 0.55% | 521 | 94 | 24.8% | 0.140 |
| 122 | 464,869 | 0.54% | 390 | 123 | 25.4% | 0.102 |
| 1 | 456,574 | 0.53% | 563 | 62 | 29.9% | 0.079 |
| 22 | 454,411 | 0.53% | 623 | 128 | 35.7% | 0.035 |
| 16 | 451,282 | 0.52% | 526 | 67 | 28.2% | 0.068 |
| 123 | 400,865 | 0.47% | 748 | 0 | 34.1% | 0.029 |
| 5 | 400,270 | 0.47% | 598 | 123 | 30.9% | 0.051 |
| 43 | 381,357 | 0.44% | 91 | 139 | 14.7% | 0.146 |
| 103 | 374,163 | 0.43% | 35 | 79 | 37.0% | 0.051 |
| 108 | 319,742 | 0.37% | 460 | 38 | 22.9% | 0.081 |
| 12 | 273,638 | 0.32% | 770 | 3 | 24.8% | 0.142 |
| 28 | 273,031 | 0.32% | 98 | 86 | 11.1% | 0.229 |
| 38 | 229,200 | 0.27% | 21 | 42 | 38.8% | 0.027 |
| 99 | 204,007 | 0.24% | 402 | 10 | 37.9% | 0.056 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **0, 25, 72, 86, 100, 123**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v15_kmeans_seed2_d0 --out runs/clustering/v15_kmeans_seed2_d0/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
