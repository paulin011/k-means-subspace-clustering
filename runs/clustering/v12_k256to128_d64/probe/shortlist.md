# Cluster shortlist — `runs/clustering/v12_k256to128_d64`

*Generated 2026-08-13 15:12 by `cluster_probe.py --rank`. K=128 affine subspaces (d=64), method `subspace_kmeans_merged_residual`. Per-cell arrays from `runs/signatures/v12_k256to128_d64` (13,021 files × 12,288 cells = 160,002,048 tokens); persistence baseline `runs/persistence/v6/err_persist.npy`.*

## What this is

One row per cluster, so you can pick a handful out of 128 instead of reading 128 rows of the main report. **Nothing here duplicates `report.md`**: size, EVR, `cells@50%`, `owned`, `files@50%`, `tCV`, `maxAff` and the affinity table live there, the monthly/seasonal *dominant-cluster* maps live in `temporal_report.md`. These are the axes neither of them has.

Then open one cluster with `python3 src/analysis/cluster_probe.py --dir runs/clustering/v12_k256to128_d64 --cluster <id>`.

## Columns

*How to read this: every column names the array it comes from. `f_j(cell)` below is the cluster's **occupancy field** — the fraction of the 13,021 timesteps at which that HEALPix cell carries label j — reduced from `label_map.npy`. `z` is the **robust per-cell residual anomaly**, `z = (residual_map − median_cell) / (1.4826·MAD_cell)`, i.e. how far a token's residual sits from that cell's own temporal norm; the normalisation is not optional, because the per-cell mean residual spans 15.7× geographically.*

| column | formula / source | reads as |
|---|---|---|
| `tokens` | `label_map.npy` bincount over the full dataset | size (**not** `model['counts']`, which covers only the sampled files) |
| `near%` | `cluster_near_frac` (`signatures.npz`) | share of its own tokens whose runner-up subspace is within 10% — contestedness |
| `zres` | mean of `z` over the tokens this cluster owns | **is this cluster hard *relative to where it sits*** — which the raw residual cannot say |
| `unmodelled` | mean raw residual ÷ mean `err_persist.npy` over its tokens | **anomalous because unmodelled** (high) vs **because rapidly changing** (low) |
| `events` | extreme events hosted (≥2 tokens), host = cluster of the event's peak-z token | the direct outlier axis |
| `tau50` | occupancy at the 50%-mass crossing (`worldmap.core_region`) | **the honest territoriality scalar**: 1.0 = owns its territory outright, <0.1 = itinerant. Not `maxf` — v6 c122 has maxf 1.00 and tau50 0.062 |
| `core` | cells with `f_j >= tau50` | how much territory holds half its mass |
| `rival` / `rival_km` | `cluster_runner_up` + great-circle km between the two tau50 core centroids | **< 2,000 km = local map blur** (the common case); **> 4,000 km = a real embedding-space confusion** |
| `seasonality` | std/mean of the 12 monthly enrichments | 0 = aseasonal |

## Metric audit

*How to read this: the `report-metric-audit` rule — a column that separates fewer than ~40 of 128 clusters is carrying no information and is dropped before shipping, not printed with more decimals.*

| column | distinct values | verdict |
|---|---|---|
| `tokens` | 128 / 128 | keep |
| `near%` | 128 / 128 | keep |
| `zres` | 128 / 128 | keep |
| `unmodelled` | 128 / 128 | keep |
| `events` | 94 / 128 | keep |
| `tau50` | 109 / 128 | keep |
| `core` | 76 / 128 | keep |
| `rival` | 80 / 128 | id column, not a metric |
| `rival_km` | 100 / 128 | keep |
| `seasonality` | 128 / 128 | keep |

### Mutual rank correlation (Spearman)

*A column that duplicates another earns no slot. These are the survivors of the redundancy audit; the rejected candidates are listed in `cluster_probe.py`'s docstring with the correlation that killed each.*

| | `tokens` | `near%` | `zres` | `unmodelled` | `events` | `tau50` | `core` | `rival_km` | `seasonality` |
|---|---|---|---|---|---|---|---|---|---|
| `tokens` | — | +0.48 | -0.04 | -0.03 | +0.09 | -0.39 | +0.87 | +0.05 | +0.25 |
| `near%` | +0.48 | — | -0.20 | +0.16 | -0.37 | -0.92 | +0.58 | +0.62 | +0.85 |
| `zres` | -0.04 | -0.20 | — | -0.22 | +0.74 | +0.10 | +0.12 | -0.28 | -0.18 |
| `unmodelled` | -0.03 | +0.16 | -0.22 | — | -0.21 | -0.01 | -0.06 | +0.31 | +0.14 |
| `events` | +0.09 | -0.37 | +0.74 | -0.21 | — | +0.27 | +0.16 | -0.45 | -0.34 |
| `tau50` | -0.39 | -0.92 | +0.10 | -0.01 | +0.27 | — | -0.59 | -0.58 | -0.92 |
| `core` | +0.87 | +0.58 | +0.12 | -0.06 | +0.16 | -0.59 | — | +0.10 | +0.46 |
| `rival_km` | +0.05 | +0.62 | -0.28 | +0.31 | -0.45 | -0.58 | +0.10 | — | +0.59 |
| `seasonality` | +0.25 | +0.85 | -0.18 | +0.14 | -0.34 | -0.92 | +0.46 | +0.59 | — |

## All 128 clusters (sorted by `events`, descending)

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 70 | 1,122,911 | 46.4% | +1.19 | 0.45 | 1,018 | 0.102 | 279 | 85 | 5,728 | 0.177 |
| 90 | 1,543,022 | 35.9% | +1.08 | 0.42 | 881 | 0.142 | 307 | 49 | 1,840 | 0.181 |
| 106 | 1,276,452 | 43.2% | +1.02 | 0.52 | 653 | 0.097 | 326 | 23 | 476 | 0.727 |
| 110 | 1,746,823 | 39.4% | +1.37 | 0.46 | 566 | 0.132 | 346 | 27 | 3,773 | 0.182 |
| 73 | 398,693 | 27.7% | +0.86 | 0.66 | 532 | 0.647 | 17 | 35 | 3,315 | 0.401 |
| 12 | 1,379,812 | 0.4% | +0.21 | 0.60 | 529 | 1.000 | 56 | 47 | 588 | 0.002 |
| 56 | 1,173,470 | 6.4% | +0.18 | 0.47 | 439 | 0.999 | 48 | 105 | 364 | 0.022 |
| 11 | 1,206,254 | 29.9% | +0.28 | 0.57 | 400 | 0.841 | 51 | 49 | 3,269 | 0.186 |
| 10 | 1,642,382 | 4.3% | +0.23 | 0.66 | 386 | 0.999 | 65 | 67 | 367 | 0.003 |
| 33 | 2,049,131 | 0.5% | +0.28 | 1.39 | 352 | 1.000 | 124 | 37 | 854 | 0.002 |
| 27 | 1,734,405 | 34.2% | +0.62 | 0.39 | 351 | 0.151 | 331 | 110 | 3,773 | 0.520 |
| 105 | 1,512,242 | 5.3% | +0.18 | 0.40 | 348 | 0.995 | 59 | 115 | 1,140 | 0.009 |
| 0 | 1,157,040 | 1.5% | +0.26 | 0.91 | 315 | 1.000 | 62 | 37 | 858 | 0.003 |
| 37 | 931,169 | 1.5% | +0.26 | 1.01 | 298 | 1.000 | 59 | 0 | 858 | 0.003 |
| 48 | 993,772 | 0.4% | +0.21 | 0.39 | 294 | 1.000 | 63 | 111 | 574 | 0.003 |
| 88 | 991,523 | 0.4% | +0.22 | 0.35 | 273 | 1.000 | 45 | 117 | 1,407 | 0.003 |
| 19 | 1,220,084 | 1.1% | +0.21 | 0.62 | 246 | 1.000 | 49 | 12 | 942 | 0.003 |
| 67 | 1,166,714 | 1.0% | +0.22 | 0.54 | 215 | 1.000 | 48 | 10 | 367 | 0.002 |
| 114 | 1,688,276 | 57.6% | +0.79 | 0.65 | 206 | 0.244 | 175 | 23 | 4,896 | 0.388 |
| 15 | 1,158,468 | 0.1% | +0.23 | 0.70 | 191 | 1.000 | 51 | 113 | 1,398 | 0.000 |
| 117 | 1,541,564 | 3.3% | +0.18 | 0.39 | 189 | 0.992 | 60 | 88 | 1,407 | 0.007 |
| 75 | 1,203,647 | 0.6% | +0.20 | 0.35 | 183 | 1.000 | 57 | 48 | 375 | 0.004 |
| 85 | 1,123,200 | 53.3% | +0.42 | 0.57 | 183 | 0.226 | 134 | 70 | 5,728 | 0.755 |
| 113 | 1,041,041 | 0.3% | +0.24 | 0.82 | 181 | 1.000 | 45 | 15 | 1,398 | 0.001 |
| 96 | 989,471 | 0.0% | +0.25 | 0.66 | 179 | 1.000 | 51 | 15 | 350 | 0.000 |
| 111 | 1,436,401 | 1.3% | +0.20 | 0.45 | 177 | 1.000 | 56 | 122 | 524 | 0.003 |
| 29 | 860,403 | 47.8% | +0.71 | 0.67 | 173 | 0.198 | 49 | 5 | 10,050 | 0.345 |
| 13 | 972,068 | 3.4% | +0.20 | 0.71 | 165 | 1.000 | 38 | 36 | 750 | 0.005 |
| 80 | 974,474 | 0.5% | +0.21 | 0.69 | 156 | 1.000 | 42 | 28 | 897 | 0.002 |
| 83 | 1,188,844 | 1.0% | +0.21 | 0.82 | 146 | 1.000 | 69 | 15 | 307 | 0.007 |
| 23 | 1,825,219 | 60.4% | +0.47 | 0.59 | 141 | 0.278 | 182 | 106 | 476 | 0.373 |
| 97 | 1,246,086 | 31.4% | -0.18 | 0.47 | 138 | 0.345 | 92 | 20 | 3,139 | 0.475 |
| 17 | 1,798,380 | 57.8% | +0.27 | 0.66 | 137 | 0.243 | 201 | 101 | 6,653 | 0.524 |
| 6 | 1,348,407 | 9.1% | +0.22 | 0.34 | 128 | 0.996 | 53 | 122 | 878 | 0.037 |
| 36 | 1,508,304 | 2.8% | +0.19 | 0.67 | 125 | 0.999 | 58 | 91 | 1,153 | 0.009 |
| 39 | 1,105,295 | 0.5% | +0.22 | 0.71 | 114 | 1.000 | 47 | 28 | 595 | 0.001 |
| 7 | 726,348 | 18.6% | +0.40 | 0.63 | 106 | 0.404 | 59 | 86 | 1,181 | 1.031 |
| 46 | 654,195 | 1.8% | +0.23 | 0.67 | 106 | 0.999 | 26 | 76 | 1,336 | 0.005 |
| 41 | 2,091,336 | 43.6% | +0.39 | 0.65 | 106 | 0.325 | 211 | 35 | 5,023 | 0.951 |
| 115 | 740,283 | 3.6% | +0.20 | 0.37 | 104 | 0.997 | 29 | 105 | 1,140 | 0.005 |
| 92 | 1,112,083 | 10.0% | +0.16 | 0.68 | 101 | 0.995 | 45 | 46 | 8,487 | 0.026 |
| 47 | 982,483 | 7.0% | +0.13 | 0.57 | 98 | 0.996 | 38 | 12 | 588 | 0.040 |
| 98 | 1,100,819 | 16.2% | +0.10 | 0.68 | 90 | 0.988 | 43 | 31 | 912 | 0.103 |
| 38 | 598,894 | 8.8% | +0.17 | 0.68 | 86 | 0.991 | 24 | 92 | 3,051 | 0.037 |
| 122 | 1,644,076 | 4.2% | +0.16 | 0.35 | 85 | 0.996 | 64 | 6 | 878 | 0.012 |
| 86 | 1,056,528 | 33.3% | +0.04 | 0.59 | 85 | 0.329 | 101 | 14 | 4,752 | 1.026 |
| 25 | 1,043,196 | 0.8% | +0.18 | 0.35 | 80 | 1.000 | 41 | 6 | 312 | 0.002 |
| 71 | 900,860 | 15.6% | +0.17 | 0.74 | 77 | 0.980 | 35 | 31 | 3,030 | 0.096 |
| 72 | 1,059,116 | 6.1% | +0.15 | 0.61 | 75 | 0.998 | 41 | 97 | 1,973 | 0.026 |
| 94 | 872,977 | 11.9% | +0.08 | 0.73 | 74 | 0.960 | 34 | 62 | 2,508 | 0.026 |
| 31 | 1,380,484 | 19.1% | +0.07 | 0.72 | 71 | 0.904 | 56 | 98 | 912 | 0.111 |
| 55 | 961,659 | 19.3% | +0.56 | 0.49 | 71 | 0.470 | 66 | 82 | 1,406 | 0.872 |
| 28 | 988,097 | 0.4% | +0.20 | 0.64 | 69 | 1.000 | 42 | 80 | 897 | 0.001 |
| 126 | 840,666 | 29.3% | +0.19 | 0.79 | 69 | 0.855 | 34 | 22 | 326 | 0.147 |
| 32 | 744,094 | 18.4% | +0.15 | 0.62 | 68 | 0.966 | 29 | 50 | 4,115 | 0.029 |
| 14 | 1,712,666 | 58.8% | +0.26 | 0.55 | 67 | 0.414 | 112 | 95 | 7,419 | 0.203 |
| 53 | 780,517 | 20.2% | +0.34 | 0.67 | 66 | 0.926 | 31 | 124 | 5,259 | 0.136 |
| 66 | 885,938 | 8.0% | +0.11 | 0.50 | 63 | 0.987 | 35 | 44 | 1,696 | 0.027 |
| 120 | 1,451,395 | 22.9% | +0.17 | 0.62 | 54 | 0.928 | 59 | 121 | 3,955 | 0.067 |
| 112 | 2,152,927 | 59.1% | +0.24 | 0.71 | 52 | 0.414 | 141 | 18 | 7,919 | 0.343 |
| 79 | 1,698,460 | 64.6% | +0.47 | 0.59 | 51 | 0.203 | 228 | 35 | 2,362 | 0.229 |
| 24 | 765,604 | 1.1% | +0.21 | 0.57 | 50 | 1.000 | 53 | 46 | 1,085 | 0.003 |
| 9 | 726,957 | 3.9% | +0.16 | 0.60 | 48 | 0.997 | 29 | 107 | 2,493 | 0.025 |
| 76 | 648,679 | 1.4% | +0.16 | 0.71 | 47 | 0.999 | 25 | 46 | 1,336 | 0.007 |
| 118 | 1,836,031 | 50.1% | +0.09 | 0.74 | 46 | 0.537 | 97 | 89 | 2,527 | 0.363 |
| 20 | 1,663,341 | 34.9% | -0.47 | 0.50 | 44 | 0.223 | 191 | 97 | 3,139 | 0.783 |
| 49 | 1,741,555 | 46.1% | +0.23 | 0.60 | 44 | 0.524 | 101 | 118 | 3,158 | 0.471 |
| 103 | 1,952,615 | 48.0% | +0.29 | 0.55 | 43 | 0.336 | 167 | 49 | 1,252 | 0.267 |
| 22 | 1,426,360 | 44.6% | +0.01 | 0.78 | 43 | 0.799 | 60 | 68 | 3,585 | 0.117 |
| 60 | 1,791,860 | 40.2% | +0.05 | 0.77 | 43 | 0.900 | 72 | 43 | 3,240 | 0.061 |
| 74 | 1,875,432 | 11.1% | +0.14 | 0.37 | 43 | 0.982 | 73 | 58 | 792 | 0.013 |
| 64 | 670,566 | 5.3% | +0.16 | 0.44 | 42 | 0.975 | 27 | 125 | 2,708 | 0.010 |
| 116 | 1,566,827 | 37.9% | +0.08 | 0.80 | 41 | 0.648 | 76 | 5 | 3,557 | 0.155 |
| 42 | 1,399,861 | 12.0% | +0.18 | 0.36 | 38 | 0.983 | 55 | 108 | 521 | 0.005 |
| 50 | 983,331 | 35.9% | -0.04 | 0.64 | 37 | 0.795 | 42 | 119 | 2,717 | 0.132 |
| 58 | 1,512,147 | 23.8% | +0.05 | 0.38 | 37 | 0.926 | 61 | 108 | 1,528 | 0.038 |
| 35 | 1,472,645 | 56.9% | +0.36 | 0.54 | 37 | 0.212 | 187 | 79 | 2,362 | 0.769 |
| 84 | 1,277,984 | 24.1% | +0.24 | 0.52 | 37 | 0.643 | 61 | 123 | 4,572 | 0.429 |
| 2 | 587,878 | 2.2% | +0.14 | 0.71 | 37 | 1.000 | 24 | 77 | 8,203 | 0.004 |
| 100 | 2,245,361 | 19.4% | +0.03 | 0.55 | 35 | 0.841 | 96 | 125 | 1,904 | 0.061 |
| 61 | 933,115 | 21.5% | +0.08 | 0.78 | 35 | 0.880 | 38 | 116 | 4,160 | 0.164 |
| 44 | 1,202,520 | 23.5% | +0.10 | 0.62 | 35 | 0.941 | 48 | 107 | 1,550 | 0.028 |
| 65 | 1,024,029 | 2.2% | +0.18 | 0.57 | 34 | 0.998 | 40 | 93 | 1,577 | 0.005 |
| 40 | 736,874 | 12.2% | +0.10 | 0.75 | 33 | 0.987 | 29 | 92 | 8,634 | 0.038 |
| 95 | 1,597,945 | 47.3% | +0.01 | 0.72 | 32 | 0.430 | 112 | 14 | 7,419 | 0.321 |
| 81 | 1,145,722 | 18.5% | +0.11 | 0.73 | 31 | 0.977 | 45 | 11 | 7,667 | 0.028 |
| 78 | 1,640,161 | 42.1% | -0.06 | 0.74 | 31 | 0.512 | 87 | 101 | 8,916 | 0.499 |
| 1 | 1,698,607 | 55.2% | +0.08 | 0.67 | 31 | 0.452 | 106 | 114 | 3,073 | 0.114 |
| 68 | 2,810,654 | 52.5% | -0.22 | 0.73 | 30 | 0.584 | 143 | 22 | 3,585 | 0.277 |
| 91 | 1,042,185 | 9.8% | +0.13 | 0.55 | 29 | 0.988 | 41 | 97 | 2,590 | 0.020 |
| 62 | 2,049,912 | 46.9% | -0.16 | 0.69 | 28 | 0.691 | 93 | 119 | 4,720 | 0.145 |
| 18 | 1,754,598 | 63.8% | +0.27 | 0.61 | 27 | 0.245 | 157 | 112 | 7,919 | 0.106 |
| 45 | 795,409 | 13.3% | +0.08 | 0.54 | 24 | 0.975 | 31 | 103 | 3,348 | 0.072 |
| 108 | 1,485,180 | 24.3% | +0.10 | 0.43 | 23 | 0.790 | 69 | 125 | 888 | 0.122 |
| 124 | 1,878,540 | 48.7% | +0.02 | 0.59 | 22 | 0.360 | 166 | 16 | 4,208 | 0.782 |
| 93 | 893,547 | 4.2% | +0.15 | 0.53 | 22 | 0.992 | 35 | 97 | 3,082 | 0.013 |
| 127 | 832,751 | 22.4% | +0.16 | 0.79 | 20 | 0.590 | 38 | 16 | 9,813 | 0.418 |
| 69 | 484,588 | 4.4% | +0.14 | 0.59 | 19 | 0.999 | 19 | 77 | 3,906 | 0.013 |
| 119 | 2,502,269 | 48.1% | -0.26 | 0.59 | 18 | 0.580 | 140 | 62 | 4,720 | 0.164 |
| 59 | 735,967 | 7.3% | +0.13 | 0.49 | 18 | 0.994 | 29 | 107 | 1,172 | 0.026 |
| 3 | 728,641 | 22.2% | +0.15 | 0.76 | 18 | 0.962 | 29 | 78 | 2,707 | 0.077 |
| 77 | 989,042 | 18.3% | +0.17 | 0.65 | 17 | 0.966 | 39 | 123 | 3,461 | 0.052 |
| 82 | 838,064 | 15.2% | +0.08 | 0.50 | 17 | 0.948 | 33 | 84 | 9,530 | 0.109 |
| 121 | 1,407,202 | 29.7% | +0.02 | 0.69 | 15 | 0.869 | 60 | 120 | 3,955 | 0.120 |
| 104 | 648,616 | 7.5% | +0.09 | 0.41 | 15 | 0.968 | 26 | 108 | 2,193 | 0.012 |
| 102 | 673,040 | 34.2% | +0.19 | 0.62 | 15 | 0.794 | 28 | 78 | 6,030 | 0.263 |
| 109 | 1,922,975 | 47.4% | -0.06 | 0.76 | 14 | 0.466 | 132 | 1 | 11,364 | 0.463 |
| 107 | 970,776 | 20.3% | +0.09 | 0.61 | 12 | 0.949 | 38 | 44 | 1,550 | 0.055 |
| 16 | 969,853 | 47.5% | -0.15 | 0.77 | 12 | 0.390 | 66 | 124 | 4,208 | 0.785 |
| 4 | 1,261,551 | 32.1% | -0.03 | 0.52 | 11 | 0.767 | 57 | 119 | 1,419 | 0.046 |
| 125 | 1,827,665 | 27.9% | -0.01 | 0.49 | 11 | 0.748 | 88 | 100 | 1,904 | 0.064 |
| 89 | 2,238,766 | 51.4% | -0.18 | 0.71 | 11 | 0.523 | 128 | 118 | 2,527 | 0.209 |
| 101 | 1,865,940 | 50.6% | -0.13 | 0.72 | 10 | 0.583 | 96 | 43 | 5,725 | 0.246 |
| 63 | 972,600 | 33.1% | +0.01 | 0.64 | 9 | 0.868 | 40 | 99 | 1,827 | 0.134 |
| 30 | 959,934 | 18.4% | -0.01 | 0.49 | 8 | 0.857 | 41 | 125 | 865 | 0.029 |
| 43 | 1,923,579 | 41.9% | -0.11 | 0.76 | 8 | 0.790 | 85 | 60 | 3,240 | 0.247 |
| 99 | 547,623 | 15.9% | +0.07 | 0.58 | 8 | 0.983 | 22 | 63 | 1,827 | 0.070 |
| 52 | 647,866 | 37.5% | +0.16 | 0.69 | 7 | 0.568 | 31 | 54 | 13,103 | 0.429 |
| 54 | 1,440,093 | 50.7% | +0.15 | 0.80 | 7 | 0.317 | 134 | 5 | 11,409 | 0.885 |
| 8 | 653,517 | 18.6% | +0.08 | 0.64 | 6 | 0.969 | 26 | 52 | 1,576 | 0.062 |
| 26 | 1,010,227 | 18.7% | -0.14 | 0.67 | 5 | 0.520 | 64 | 31 | 1,231 | 0.750 |
| 5 | 1,401,319 | 49.6% | -0.14 | 0.72 | 5 | 0.365 | 104 | 116 | 3,557 | 0.596 |
| 87 | 615,548 | 32.5% | -0.04 | 0.70 | 5 | 0.822 | 26 | 109 | 14,284 | 0.168 |
| 123 | 1,027,927 | 22.9% | -0.02 | 0.67 | 5 | 0.544 | 53 | 121 | 7,268 | 0.502 |
| 51 | 839,378 | 30.3% | +0.00 | 0.67 | 4 | 0.812 | 37 | 121 | 6,012 | 0.214 |
| 57 | 669,754 | 15.6% | -0.03 | 0.70 | 3 | 0.601 | 29 | 95 | 2,644 | 0.610 |
| 34 | 1,253,281 | 50.1% | -0.04 | 0.72 | 2 | 0.646 | 60 | 21 | 9,875 | 0.123 |
| 21 | 1,872,120 | 53.0% | -0.31 | 0.68 | 2 | 0.374 | 150 | 34 | 9,875 | 0.590 |

## Presets

*The same table cut four ways — each is the top 5 on one axis, so a reader with a specific question does not have to re-sort.*

**hardest relative to its geography** — by `zres`:

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 110 | 1,746,823 | 39.4% | +1.37 | 0.46 | 566 | 0.132 | 346 | 27 | 3,773 | 0.182 |
| 70 | 1,122,911 | 46.4% | +1.19 | 0.45 | 1,018 | 0.102 | 279 | 85 | 5,728 | 0.177 |
| 90 | 1,543,022 | 35.9% | +1.08 | 0.42 | 881 | 0.142 | 307 | 49 | 1,840 | 0.181 |
| 106 | 1,276,452 | 43.2% | +1.02 | 0.52 | 653 | 0.097 | 326 | 23 | 476 | 0.727 |
| 73 | 398,693 | 27.7% | +0.86 | 0.66 | 532 | 0.647 | 17 | 35 | 3,315 | 0.401 |

**most unmodelled (hard, but not because it moves)** — by `unmodelled`:

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 33 | 2,049,131 | 0.5% | +0.28 | 1.39 | 352 | 1.000 | 124 | 37 | 854 | 0.002 |
| 37 | 931,169 | 1.5% | +0.26 | 1.01 | 298 | 1.000 | 59 | 0 | 858 | 0.003 |
| 0 | 1,157,040 | 1.5% | +0.26 | 0.91 | 315 | 1.000 | 62 | 37 | 858 | 0.003 |
| 83 | 1,188,844 | 1.0% | +0.21 | 0.82 | 146 | 1.000 | 69 | 15 | 307 | 0.007 |
| 113 | 1,041,041 | 0.3% | +0.24 | 0.82 | 181 | 1.000 | 45 | 15 | 1,398 | 0.001 |

**most itinerant (a moving weather state, not a region)** — by `tau50` (ascending):

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 106 | 1,276,452 | 43.2% | +1.02 | 0.52 | 653 | 0.097 | 326 | 23 | 476 | 0.727 |
| 70 | 1,122,911 | 46.4% | +1.19 | 0.45 | 1,018 | 0.102 | 279 | 85 | 5,728 | 0.177 |
| 110 | 1,746,823 | 39.4% | +1.37 | 0.46 | 566 | 0.132 | 346 | 27 | 3,773 | 0.182 |
| 90 | 1,543,022 | 35.9% | +1.08 | 0.42 | 881 | 0.142 | 307 | 49 | 1,840 | 0.181 |
| 27 | 1,734,405 | 34.2% | +0.62 | 0.39 | 351 | 0.151 | 331 | 110 | 3,773 | 0.520 |

**most remotely confused (`near%` > 30%)** — by `rival_km`:

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 87 | 615,548 | 32.5% | -0.04 | 0.70 | 5 | 0.822 | 26 | 109 | 14,284 | 0.168 |
| 52 | 647,866 | 37.5% | +0.16 | 0.69 | 7 | 0.568 | 31 | 54 | 13,103 | 0.429 |
| 54 | 1,440,093 | 50.7% | +0.15 | 0.80 | 7 | 0.317 | 134 | 5 | 11,409 | 0.885 |
| 109 | 1,922,975 | 47.4% | -0.06 | 0.76 | 14 | 0.466 | 132 | 1 | 11,364 | 0.463 |
| 29 | 860,403 | 47.8% | +0.71 | 0.67 | 173 | 0.198 | 49 | 5 | 10,050 | 0.345 |

## Global findings (not per-cluster columns)

- **Two kinds of cluster.** 98 of 128 are *territorial* (`tau50 >= 0.5` — a fixed geographic region) and 1 are *itinerant* (`tau50 < 0.1` — a moving weather state whose mass is smeared over hundreds of cells). `maxf` ↔ `near%` correlate at Spearman -0.833: crisp territory ⇒ crisp assignment. That is one axis, not two, and `near%` already carries it — which is why `maxf` is not a column here. What it does dictate is *how to draw the map*.
- **Presence is useless as a map.** Cells where a cluster appears at all span 60 to 4155 (mean 1004) — up to 34% of the globe. Hence the mass-coverage threshold rather than an occupancy one.
- **No cluster is an isolated blob.** `iso = ‖μⱼ−μ_global‖²/trace[j]` maxes out at 0.19 across all 128 clusters: the between-cluster offset is at most that fraction of a cluster's own internal spread. Combined with the near-tie rate and the knee-free merge cascade, **"outlier cluster" is not a meaningful category in this embedding** — outliers exist at the *token* level, which is what the `events` column and each cluster's P3 panel are for.
- **Rivals are usually neighbours.** Median `rival_km` 2,527 km against 9,958 km for a random cluster pair; 58/128 are within 2,000 km. But 25 of 48 contested clusters have a rival more than 4,000 km away — those are real embedding-space confusions, not geographic blur, and nothing else in the repo distinguishes the two cases.
- **No diurnal cycle.** Largest deviation of any UTC-hour share from 0.25, over all 128 clusters: **0.016**. The 6-hourly latents carry a seasonal cycle and essentially no diurnal one, so there is no diurnal panel.

## Extreme events

*How to read this: `residual_map.npy` normalised per cell by median/MAD, cut at the top 0.1%, then grouped into space-time connected components — adjacency is a real HEALPix NESTED neighbour at the same timestep, or the same cell at t+1.*

- threshold: robust **z ≥ 5.73**, **160,000** tokens (0.1% of 160,002,048).
- **14,578** events of ≥2 tokens (of 44,605 components); largest **1,010** tokens; 81.2% of flagged tokens sit inside a multi-token event.

Largest events in the run:

| event | when | where | steps | tokens | peak z | peak residual | host |
|---|---|---|---|---|---|---|---|
| 13602 | 2016-08-26T00 → 2016-08-30T00 | 79.7°N 28.0°E | 17 | 1,010 | 19.0 | 2,700 | c96 |
| 9179 | 2015-12-28T12 → 2016-01-03T12 | 80.9°N 131.9°W | 25 | 953 | 18.9 | 3,834 | c83 |
| 8059 | 2015-09-28T06 → 2015-10-04T18 | 76.2°S 153.6°W | 27 | 894 | 20.4 | 3,012 | c33 |
| 40900 | 2022-03-16T00 → 2022-03-20T12 | 76.1°S 53.5°W | 19 | 883 | 16.2 | 1,387 | c33 |
| 20663 | 2018-02-22T06 → 2018-02-27T06 | 82.1°N 152.0°E | 21 | 798 | 22.6 | 4,217 | c83 |
| 40210 | 2022-01-19T00 → 2022-01-25T00 | 78.7°S 157.5°W | 25 | 716 | 13.5 | 1,620 | c33 |
| 39323 | 2021-11-17T06 → 2021-11-22T18 | 76.7°S 132.6°W | 23 | 704 | 16.3 | 1,973 | c48 |
| 32992 | 2020-07-19T00 → 2020-07-25T00 | 69.8°N 35.9°W | 25 | 701 | 13.3 | 3,019 | c103 |
| 32673 | 2020-06-29T12 → 2020-07-05T06 | 62.9°N 76.5°E | 24 | 640 | 16.2 | 3,227 | c11 |
| 32013 | 2020-05-09T18 → 2020-05-14T18 | 81.0°N 11.5°E | 21 | 638 | 13.5 | 2,343 | c83 |

## Validation

*Run on every `--rank` invocation; these are the invariants that catch a regression.*

| # | check | result |
|---|---|---|
| 1 | f_j.sum(0)*N == full-dataset counts | ✔ 160,002,048 tokens, exact integer match=True |
| 2 | tau_q covers >= q, and q=0.5 == report cells@50% on the sampled files | ✔ min (covered - q) = +7.07e-06; cells@50% identical for 128/128 clusters |
| 3 | seasonal fields weighted by files/season sum back to the annual field | ✔ max |recon - f_annual| = 0.00e+00 (files/season [3124, 3312, 3312, 3273]) |
| 4 | HEALPix-neighbour adjacency merges the two named events | ✔ 2022-03-17..19: largest component 883 tokens (next 32, 23); 2018-02-23..25: largest component 798 tokens (next 11, 8) |
| 5 | zres un-normalised reproduces the model objective | ✔ count-weighted mean residual 1877.22 vs model final_obj_per_token 1877.04 (0.01%); z*1.4826*MAD+median re-sums to 1877.22 (9.4e-14 rel) |
