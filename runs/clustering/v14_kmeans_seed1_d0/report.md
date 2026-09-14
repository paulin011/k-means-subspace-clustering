# Clustering report (subspace_kmeans) — `runs/clustering/v14_kmeans_seed1_d0`

*Generated 2026-09-14 15:54 by `analyze_clusters.py`. K=128 point clusters in 2048-dim token space, 86,016,000 tokens.*

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
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v14_kmeans_seed1_d0/sample.json --seed 1 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 7000, full list in `runs/clustering/v14_kmeans_seed1_d0/sample.json`): 0, 1, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21, 22, 24, 25, 26, 27, 28 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 7606.38 | 100.00% | 109 | 26,015,684 |
| 2 | 5283.80 | 67.59% | 9,837 | 5,439,820 |
| 3 | 5121.93 | 32.77% | 35,536 | 2,019,804 |
| 4 | 5081.45 | 17.65% | 104,363 | 1,483,253 |
| 5 | 5064.19 | 12.35% | 155,829 | 1,397,484 |
| 6 | 5054.33 | 9.58% | 167,856 | 1,376,794 |
| 7 | 5047.89 | 7.79% | 175,206 | 1,337,973 |
| 8 | 5043.29 | 6.60% | 179,700 | 1,299,693 |
| 9 | 5039.69 | 5.82% | 182,938 | 1,266,827 |
| 10 | 5036.66 | 5.29% | 185,451 | 1,238,634 |
| 11 | 5034.03 | 4.84% | 187,257 | 1,226,987 |
| 12 | 5031.78 | 4.42% | 189,013 | 1,239,039 |
| 13 | 5029.91 | 4.02% | 190,639 | 1,245,454 |
| 14 | 5028.34 | 3.65% | 192,151 | 1,246,167 |
| 15 | 5027.02 | 3.32% | 193,906 | 1,243,695 |
| 16 | 5025.91 | 3.03% | 195,539 | 1,237,538 |
| 17 | 5024.96 | 2.76% | 197,169 | 1,233,593 |
| 18 | 5024.18 | 2.50% | 199,147 | 1,239,043 |
| 19 | 5023.58 | 2.27% | 201,127 | 1,245,069 |
| 20 | 5023.10 | 2.08% | 203,194 | 1,250,154 |
| 21 | 5022.68 | 1.93% | 205,068 | 1,254,961 |
| 22 | 5022.30 | 1.81% | 206,868 | 1,259,482 |
| 23 | 5021.97 | 1.71% | 208,595 | 1,263,976 |
| 24 | 5021.65 | 1.63% | 210,244 | 1,267,716 |
| 25 | 5021.38 | 1.56% | 211,705 | 1,270,418 |

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

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.25%–1.48% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | cells@50% | owned | files@50% | tCV |
|---|---|---|---|---|---|---|
| 14 | 1,273,209 | 1.48% | 452 | 269 | 20.9% | 0.205 |
| 42 | 1,183,953 | 1.38% | 1446 | 43 | 37.3% | 0.042 |
| 16 | 1,127,474 | 1.31% | 334 | 279 | 26.3% | 0.109 |
| 23 | 1,116,258 | 1.30% | 1538 | 202 | 37.6% | 0.073 |
| 21 | 1,104,206 | 1.28% | 1591 | 46 | 37.5% | 0.034 |
| 28 | 1,087,785 | 1.26% | 1462 | 163 | 38.4% | 0.055 |
| 38 | 1,080,143 | 1.26% | 1841 | 37 | 37.6% | 0.058 |
| 102 | 1,055,267 | 1.23% | 849 | 241 | 29.6% | 0.076 |
| 116 | 1,035,952 | 1.20% | 1287 | 63 | 37.2% | 0.102 |
| 85 | 933,979 | 1.09% | 1131 | 190 | 39.1% | 0.045 |
| 123 | 929,498 | 1.08% | 1033 | 322 | 37.8% | 0.043 |
| 29 | 889,913 | 1.03% | 1370 | 50 | 38.4% | 0.033 |
| 90 | 886,053 | 1.03% | 964 | 108 | 37.1% | 0.048 |
| 47 | 884,600 | 1.03% | 361 | 433 | 20.2% | 0.127 |
| 74 | 881,224 | 1.02% | 2139 | 0 | 37.9% | 0.040 |
| 84 | 877,350 | 1.02% | 1467 | 12 | 37.1% | 0.033 |
| 82 | 875,394 | 1.02% | 727 | 257 | 40.3% | 0.035 |
| 114 | 861,460 | 1.00% | 1520 | 4 | 37.9% | 0.039 |
| 94 | 857,262 | 1.00% | 769 | 182 | 36.1% | 0.053 |
| 66 | 854,276 | 0.99% | 974 | 175 | 37.5% | 0.037 |
| 77 | 853,908 | 0.99% | 1093 | 97 | 34.5% | 0.045 |
| 50 | 848,139 | 0.99% | 900 | 316 | 40.5% | 0.024 |
| 40 | 843,515 | 0.98% | 1160 | 84 | 35.3% | 0.066 |
| 113 | 829,167 | 0.96% | 1738 | 17 | 38.3% | 0.045 |
| 59 | 819,248 | 0.95% | 437 | 203 | 34.8% | 0.088 |
| 65 | 800,227 | 0.93% | 1041 | 107 | 38.9% | 0.045 |
| 100 | 798,556 | 0.93% | 1008 | 210 | 37.3% | 0.051 |
| 2 | 795,529 | 0.92% | 788 | 82 | 34.7% | 0.082 |
| 32 | 795,200 | 0.92% | 1137 | 65 | 39.0% | 0.035 |
| 54 | 794,520 | 0.92% | 1061 | 18 | 35.6% | 0.111 |
| 5 | 793,533 | 0.92% | 769 | 380 | 39.0% | 0.040 |
| 39 | 791,610 | 0.92% | 1045 | 46 | 39.8% | 0.042 |
| 56 | 790,102 | 0.92% | 1192 | 15 | 35.8% | 0.066 |
| 25 | 789,348 | 0.92% | 1580 | 9 | 36.7% | 0.055 |
| 120 | 786,497 | 0.91% | 1248 | 21 | 35.4% | 0.056 |
| 3 | 783,437 | 0.91% | 1354 | 6 | 35.4% | 0.058 |
| 8 | 781,372 | 0.91% | 668 | 170 | 37.2% | 0.064 |
| 27 | 778,571 | 0.91% | 723 | 199 | 36.1% | 0.099 |
| 6 | 775,935 | 0.90% | 713 | 124 | 27.1% | 0.090 |
| 69 | 760,210 | 0.88% | 1138 | 7 | 38.2% | 0.037 |
| 60 | 759,407 | 0.88% | 2167 | 0 | 37.8% | 0.037 |
| 118 | 755,147 | 0.88% | 797 | 164 | 40.3% | 0.017 |
| 95 | 743,707 | 0.86% | 877 | 70 | 37.5% | 0.049 |
| 107 | 733,381 | 0.85% | 1117 | 56 | 37.2% | 0.063 |
| 63 | 729,090 | 0.85% | 974 | 163 | 39.2% | 0.052 |
| 51 | 728,395 | 0.85% | 1076 | 74 | 36.4% | 0.063 |
| 125 | 724,851 | 0.84% | 311 | 108 | 35.8% | 0.031 |
| 86 | 719,521 | 0.84% | 165 | 156 | 19.8% | 0.237 |
| 111 | 712,406 | 0.83% | 1263 | 12 | 38.1% | 0.030 |
| 53 | 710,949 | 0.83% | 701 | 45 | 39.8% | 0.037 |
| 67 | 710,158 | 0.83% | 1077 | 41 | 32.9% | 0.050 |
| 101 | 708,014 | 0.82% | 1657 | 0 | 36.8% | 0.033 |
| 9 | 700,647 | 0.81% | 1115 | 8 | 30.9% | 0.084 |
| 71 | 688,958 | 0.80% | 602 | 213 | 29.0% | 0.086 |
| 103 | 682,759 | 0.79% | 1219 | 6 | 37.3% | 0.042 |
| 124 | 682,308 | 0.79% | 792 | 114 | 40.4% | 0.027 |
| 115 | 675,088 | 0.78% | 760 | 130 | 31.7% | 0.100 |
| 91 | 671,225 | 0.78% | 697 | 58 | 36.7% | 0.099 |
| 37 | 663,031 | 0.77% | 339 | 173 | 32.2% | 0.098 |
| 34 | 660,084 | 0.77% | 263 | 188 | 23.0% | 0.164 |
| 44 | 655,721 | 0.76% | 529 | 160 | 39.5% | 0.047 |
| 109 | 654,968 | 0.76% | 1032 | 75 | 36.3% | 0.075 |
| 13 | 649,484 | 0.76% | 531 | 45 | 26.5% | 0.078 |
| 7 | 642,686 | 0.75% | 1636 | 0 | 37.1% | 0.054 |
| 112 | 641,825 | 0.75% | 975 | 99 | 29.7% | 0.084 |
| 61 | 640,743 | 0.74% | 1456 | 9 | 39.0% | 0.041 |
| 33 | 640,720 | 0.74% | 832 | 40 | 35.7% | 0.055 |
| 58 | 639,890 | 0.74% | 438 | 329 | 37.1% | 0.055 |
| 10 | 639,605 | 0.74% | 1106 | 96 | 36.0% | 0.084 |
| 89 | 636,407 | 0.74% | 1271 | 0 | 34.1% | 0.035 |
| 104 | 632,666 | 0.74% | 1198 | 51 | 33.7% | 0.055 |
| 0 | 632,482 | 0.74% | 1014 | 25 | 34.9% | 0.146 |
| 105 | 629,920 | 0.73% | 761 | 56 | 32.9% | 0.070 |
| 92 | 628,038 | 0.73% | 282 | 82 | 27.3% | 0.100 |
| 97 | 609,199 | 0.71% | 503 | 52 | 30.9% | 0.049 |
| 121 | 608,608 | 0.71% | 618 | 62 | 32.0% | 0.088 |
| 75 | 607,479 | 0.71% | 621 | 31 | 34.2% | 0.161 |
| 87 | 607,196 | 0.71% | 396 | 167 | 33.1% | 0.099 |
| 15 | 605,104 | 0.70% | 507 | 31 | 37.0% | 0.059 |
| 83 | 598,307 | 0.70% | 1390 | 37 | 38.1% | 0.054 |
| 17 | 594,836 | 0.69% | 511 | 219 | 25.0% | 0.086 |
| 1 | 593,639 | 0.69% | 1109 | 0 | 36.5% | 0.037 |
| 117 | 592,742 | 0.69% | 561 | 130 | 27.7% | 0.099 |
| 99 | 592,118 | 0.69% | 1008 | 65 | 36.3% | 0.044 |
| 11 | 588,878 | 0.68% | 1013 | 6 | 30.3% | 0.069 |
| 31 | 588,446 | 0.68% | 269 | 253 | 27.2% | 0.067 |
| 79 | 587,166 | 0.68% | 1504 | 0 | 36.6% | 0.030 |
| 52 | 579,066 | 0.67% | 1087 | 0 | 34.2% | 0.045 |
| 78 | 572,115 | 0.67% | 214 | 73 | 16.1% | 0.246 |
| 110 | 562,009 | 0.65% | 633 | 15 | 38.5% | 0.041 |
| 62 | 558,207 | 0.65% | 105 | 115 | 29.8% | 0.088 |
| 80 | 557,278 | 0.65% | 591 | 85 | 35.6% | 0.102 |
| 24 | 556,066 | 0.65% | 429 | 193 | 21.6% | 0.129 |
| 49 | 550,298 | 0.64% | 717 | 127 | 31.2% | 0.084 |
| 70 | 549,478 | 0.64% | 913 | 96 | 38.2% | 0.054 |
| 36 | 548,304 | 0.64% | 906 | 13 | 36.1% | 0.031 |
| 55 | 547,213 | 0.64% | 1107 | 2 | 35.8% | 0.046 |
| 108 | 546,879 | 0.64% | 454 | 123 | 25.0% | 0.090 |
| 35 | 546,466 | 0.64% | 185 | 55 | 23.9% | 0.101 |
| 45 | 539,550 | 0.63% | 161 | 184 | 34.2% | 0.096 |
| 4 | 529,002 | 0.62% | 1035 | 3 | 34.5% | 0.062 |
| 119 | 522,881 | 0.61% | 744 | 2 | 34.4% | 0.055 |
| 64 | 522,614 | 0.61% | 181 | 125 | 22.8% | 0.086 |
| 81 | 520,662 | 0.61% | 535 | 128 | 25.9% | 0.132 |
| 26 | 516,279 | 0.60% | 172 | 123 | 28.0% | 0.132 |
| 98 | 512,773 | 0.60% | 132 | 164 | 19.6% | 0.177 |
| 41 | 500,323 | 0.58% | 493 | 184 | 16.9% | 0.149 |
| 68 | 490,346 | 0.57% | 300 | 120 | 30.0% | 0.038 |
| 18 | 487,300 | 0.57% | 1319 | 16 | 35.3% | 0.062 |
| 127 | 481,984 | 0.56% | 448 | 63 | 32.9% | 0.048 |
| 93 | 480,694 | 0.56% | 647 | 16 | 28.4% | 0.066 |
| 20 | 479,785 | 0.56% | 675 | 136 | 32.6% | 0.047 |
| 43 | 470,540 | 0.55% | 925 | 13 | 33.8% | 0.049 |
| 30 | 454,458 | 0.53% | 408 | 93 | 21.3% | 0.135 |
| 57 | 449,429 | 0.52% | 1153 | 1 | 36.7% | 0.034 |
| 88 | 445,348 | 0.52% | 40 | 101 | 40.3% | 0.038 |
| 76 | 430,755 | 0.50% | 318 | 44 | 29.9% | 0.050 |
| 22 | 414,393 | 0.48% | 613 | 5 | 36.2% | 0.054 |
| 46 | 400,695 | 0.47% | 295 | 50 | 31.2% | 0.044 |
| 48 | 399,228 | 0.46% | 189 | 179 | 16.1% | 0.099 |
| 72 | 391,476 | 0.46% | 814 | 11 | 26.4% | 0.070 |
| 12 | 382,737 | 0.44% | 77 | 103 | 35.1% | 0.086 |
| 126 | 373,329 | 0.43% | 614 | 0 | 34.3% | 0.036 |
| 73 | 371,211 | 0.43% | 150 | 104 | 21.5% | 0.168 |
| 122 | 333,286 | 0.39% | 720 | 4 | 33.8% | 0.079 |
| 96 | 267,796 | 0.31% | 95 | 105 | 12.0% | 0.211 |
| 19 | 258,776 | 0.30% | 52 | 91 | 16.0% | 0.139 |
| 106 | 213,092 | 0.25% | 68 | 72 | 13.0% | 0.174 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **1, 7, 52, 60, 74, 79, 89, 101, 126**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v14_kmeans_seed1_d0 --out runs/clustering/v14_kmeans_seed1_d0/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

_Skipped: point clusters (d=0) have no subspace basis to compare._

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
