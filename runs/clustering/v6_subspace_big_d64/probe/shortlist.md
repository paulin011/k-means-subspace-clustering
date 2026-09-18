# Cluster shortlist — `runs/clustering/v6_subspace_big_d64`

*Generated 2026-08-13 16:28 by `cluster_probe.py --rank`. K=128 affine subspaces (d=64), method `subspace_kmeans`. Per-cell arrays from `runs/signatures/v6_d64_margin` (13,021 files × 12,288 cells = 160,002,048 tokens); persistence baseline `runs/persistence/v6/err_persist.npy`.*

## What this is

One row per cluster, so you can pick a handful out of 128 instead of reading 128 rows of the main report. **Nothing here duplicates `report.md`**: size, EVR, `cells@50%`, `owned`, `files@50%`, `tCV`, `maxAff` and the affinity table live there, the monthly/seasonal *dominant-cluster* maps live in `temporal_report.md`. These are the axes neither of them has.

Then open one cluster with `python3 src/analysis/cluster_probe.py --dir runs/clustering/v6_subspace_big_d64 --cluster <id>`.

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
| `events` | 84 / 128 | keep |
| `tau50` | 124 / 128 | keep |
| `core` | 94 / 128 | keep |
| `rival` | 88 / 128 | id column, not a metric |
| `rival_km` | 99 / 128 | keep |
| `seasonality` | 128 / 128 | keep |

### Mutual rank correlation (Spearman)

*A column that duplicates another earns no slot. These are the survivors of the redundancy audit; the rejected candidates are listed in `cluster_probe.py`'s docstring with the correlation that killed each.*

| | `tokens` | `near%` | `zres` | `unmodelled` | `events` | `tau50` | `core` | `rival_km` | `seasonality` |
|---|---|---|---|---|---|---|---|---|---|
| `tokens` | — | +0.13 | -0.12 | -0.11 | +0.13 | -0.08 | +0.56 | -0.26 | -0.12 |
| `near%` | +0.13 | — | +0.08 | -0.15 | -0.25 | -0.89 | +0.70 | +0.46 | +0.67 |
| `zres` | -0.12 | +0.08 | — | -0.32 | +0.75 | -0.26 | +0.29 | +0.01 | +0.03 |
| `unmodelled` | -0.11 | -0.15 | -0.32 | — | -0.29 | +0.36 | -0.34 | -0.20 | -0.04 |
| `events` | +0.13 | -0.25 | +0.75 | -0.29 | — | +0.04 | +0.23 | -0.20 | -0.28 |
| `tau50` | -0.08 | -0.89 | -0.26 | +0.36 | +0.04 | — | -0.80 | -0.48 | -0.73 |
| `core` | +0.56 | +0.70 | +0.29 | -0.34 | +0.23 | -0.80 | — | +0.17 | +0.43 |
| `rival_km` | -0.26 | +0.46 | +0.01 | -0.20 | -0.20 | -0.48 | +0.17 | — | +0.45 |
| `seasonality` | -0.12 | +0.67 | +0.03 | -0.04 | -0.28 | -0.73 | +0.43 | +0.45 | — |

## All 128 clusters (sorted by `events`, descending)

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 122 | 698,543 | 45.7% | +1.52 | 0.56 | 1,027 | 0.062 | 262 | 82 | 5,849 | 0.663 |
| 13 | 898,423 | 48.1% | +1.17 | 0.44 | 731 | 0.042 | 545 | 54 | 3,365 | 0.062 |
| 127 | 1,422,753 | 2.9% | +0.20 | 0.39 | 669 | 1.000 | 57 | 107 | 379 | 0.002 |
| 66 | 2,716,397 | 0.9% | +0.27 | 1.45 | 530 | 1.000 | 165 | 77 | 371 | 0.001 |
| 40 | 918,171 | 59.6% | +0.85 | 0.51 | 501 | 0.159 | 150 | 54 | 4,989 | 0.339 |
| 0 | 1,661,809 | 3.6% | +0.24 | 0.92 | 497 | 1.000 | 65 | 17 | 858 | 0.002 |
| 36 | 906,282 | 43.4% | +1.57 | 0.47 | 495 | 0.088 | 274 | 98 | 3,496 | 0.381 |
| 21 | 1,573,383 | 0.9% | +0.20 | 0.65 | 491 | 0.999 | 61 | 90 | 461 | 0.001 |
| 27 | 2,749,899 | 1.3% | +0.22 | 0.92 | 416 | 1.000 | 108 | 14 | 115 | 0.004 |
| 98 | 1,278,935 | 42.3% | +1.05 | 0.38 | 378 | 0.118 | 308 | 36 | 3,496 | 0.235 |
| 25 | 1,417,460 | 4.7% | +0.21 | 0.91 | 354 | 1.000 | 55 | 74 | 1,704 | 0.018 |
| 35 | 1,824,143 | 5.2% | +0.16 | 0.70 | 347 | 0.995 | 71 | 74 | 335 | 0.018 |
| 93 | 1,751,390 | 3.5% | +0.17 | 0.40 | 339 | 0.995 | 68 | 50 | 402 | 0.010 |
| 31 | 981,031 | 58.8% | +0.89 | 0.58 | 331 | 0.147 | 142 | 48 | 3,263 | 0.351 |
| 113 | 1,440,113 | 39.5% | +0.31 | 0.47 | 315 | 0.232 | 145 | 121 | 3,076 | 0.268 |
| 94 | 1,034,446 | 45.6% | +0.66 | 0.43 | 313 | 0.107 | 187 | 78 | 3,989 | 0.308 |
| 77 | 762,400 | 1.8% | +0.27 | 0.87 | 312 | 1.000 | 50 | 0 | 437 | 0.012 |
| 102 | 753,653 | 34.9% | +0.73 | 0.55 | 300 | 0.405 | 56 | 95 | 4,179 | 0.867 |
| 17 | 2,006,132 | 4.4% | +0.21 | 0.54 | 273 | 0.998 | 78 | 123 | 760 | 0.005 |
| 123 | 1,245,206 | 1.3% | +0.20 | 0.49 | 264 | 0.999 | 50 | 17 | 760 | 0.003 |
| 18 | 1,349,217 | 43.4% | +0.81 | 0.41 | 240 | 0.176 | 223 | 52 | 6,293 | 0.096 |
| 52 | 1,303,730 | 47.5% | +0.32 | 0.49 | 198 | 0.211 | 162 | 18 | 6,293 | 0.781 |
| 54 | 1,123,680 | 63.8% | +0.54 | 0.50 | 187 | 0.126 | 198 | 73 | 4,079 | 0.412 |
| 111 | 1,451,746 | 39.2% | +0.56 | 0.29 | 167 | 0.167 | 249 | 119 | 2,500 | 0.110 |
| 57 | 926,189 | 60.1% | +0.78 | 0.66 | 158 | 0.121 | 180 | 9 | 7,693 | 0.602 |
| 34 | 1,483,992 | 37.0% | +0.61 | 0.66 | 143 | 0.344 | 88 | 76 | 3,068 | 0.729 |
| 61 | 1,671,345 | 22.6% | +0.13 | 0.72 | 143 | 0.889 | 68 | 47 | 2,400 | 0.118 |
| 50 | 2,526,437 | 8.8% | +0.09 | 0.41 | 142 | 0.965 | 99 | 17 | 786 | 0.032 |
| 74 | 2,169,784 | 7.1% | +0.16 | 0.78 | 142 | 0.992 | 85 | 35 | 335 | 0.022 |
| 89 | 1,493,525 | 7.0% | +0.17 | 0.76 | 142 | 0.996 | 58 | 16 | 1,244 | 0.016 |
| 124 | 865,987 | 10.0% | +0.14 | 0.48 | 131 | 0.973 | 34 | 4 | 2,607 | 0.019 |
| 80 | 1,301,815 | 2.4% | +0.19 | 0.73 | 127 | 0.998 | 52 | 56 | 318 | 0.009 |
| 107 | 1,422,546 | 26.9% | +0.24 | 0.41 | 127 | 0.732 | 62 | 39 | 2,340 | 0.059 |
| 7 | 971,261 | 54.3% | +0.52 | 0.57 | 122 | 0.136 | 155 | 48 | 1,646 | 1.014 |
| 56 | 1,685,942 | 3.9% | +0.17 | 0.73 | 104 | 0.996 | 65 | 80 | 318 | 0.013 |
| 20 | 731,845 | 31.4% | +0.31 | 0.64 | 103 | 0.927 | 29 | 81 | 1,475 | 0.076 |
| 48 | 1,296,960 | 63.2% | +0.27 | 0.56 | 98 | 0.248 | 143 | 31 | 3,263 | 0.321 |
| 49 | 601,787 | 3.8% | +0.09 | 0.61 | 96 | 0.942 | 24 | 28 | 2,945 | 0.171 |
| 104 | 1,057,208 | 42.3% | +0.20 | 0.62 | 96 | 0.274 | 106 | 120 | 4,050 | 0.885 |
| 16 | 1,276,746 | 11.3% | +0.16 | 0.70 | 92 | 0.989 | 50 | 89 | 1,244 | 0.046 |
| 24 | 1,074,553 | 27.9% | +0.18 | 0.62 | 90 | 0.347 | 100 | 73 | 6,603 | 1.122 |
| 119 | 1,926,037 | 36.2% | +0.39 | 0.40 | 87 | 0.407 | 143 | 70 | 2,252 | 0.120 |
| 99 | 1,178,608 | 66.1% | +0.71 | 0.68 | 87 | 0.177 | 175 | 65 | 1,606 | 0.625 |
| 116 | 657,401 | 28.9% | +0.61 | 0.70 | 84 | 0.405 | 45 | 51 | 1,950 | 0.721 |
| 28 | 422,223 | 4.4% | +0.42 | 0.52 | 83 | 1.000 | 17 | 49 | 2,945 | 0.257 |
| 87 | 876,012 | 14.9% | +0.17 | 0.68 | 82 | 0.990 | 34 | 34 | 2,119 | 0.097 |
| 105 | 1,519,789 | 20.8% | +0.29 | 0.52 | 81 | 0.625 | 76 | 53 | 2,100 | 0.449 |
| 55 | 637,053 | 17.2% | +0.11 | 0.52 | 81 | 0.983 | 25 | 62 | 3,305 | 0.111 |
| 64 | 1,048,150 | 32.5% | +0.62 | 0.55 | 79 | 0.366 | 82 | 7 | 17,078 | 0.863 |
| 96 | 683,145 | 21.0% | +0.20 | 0.70 | 78 | 0.902 | 27 | 110 | 219 | 0.078 |
| 32 | 632,276 | 6.0% | +0.15 | 0.75 | 78 | 0.997 | 25 | 89 | 2,303 | 0.087 |
| 14 | 687,137 | 3.5% | +0.16 | 0.70 | 75 | 1.000 | 30 | 27 | 115 | 0.033 |
| 90 | 1,335,381 | 14.1% | +0.05 | 0.62 | 68 | 0.984 | 52 | 121 | 2,271 | 0.079 |
| 60 | 1,416,878 | 54.2% | +0.40 | 0.69 | 67 | 0.230 | 152 | 5 | 5,185 | 0.858 |
| 88 | 1,664,659 | 45.1% | +0.37 | 0.60 | 67 | 0.287 | 146 | 42 | 2,425 | 0.664 |
| 84 | 1,127,565 | 62.1% | +0.27 | 0.61 | 67 | 0.305 | 91 | 126 | 6,857 | 0.084 |
| 70 | 1,701,883 | 39.4% | +0.25 | 0.41 | 62 | 0.520 | 105 | 39 | 2,963 | 0.192 |
| 46 | 1,188,475 | 35.0% | +0.16 | 0.81 | 61 | 0.843 | 48 | 3 | 1,500 | 0.192 |
| 22 | 1,225,682 | 55.1% | +0.22 | 0.55 | 61 | 0.307 | 99 | 71 | 4,449 | 0.198 |
| 43 | 1,591,716 | 32.9% | -0.49 | 0.46 | 59 | 0.223 | 192 | 121 | 2,574 | 0.717 |
| 19 | 1,221,936 | 34.8% | +0.06 | 0.81 | 57 | 0.564 | 62 | 108 | 2,060 | 0.154 |
| 30 | 1,217,109 | 50.3% | +0.21 | 0.72 | 55 | 0.292 | 139 | 126 | 224 | 0.867 |
| 26 | 1,254,224 | 64.1% | +0.25 | 0.57 | 52 | 0.235 | 141 | 40 | 14,342 | 0.430 |
| 5 | 1,309,410 | 61.1% | +0.62 | 0.59 | 48 | 0.153 | 258 | 33 | 1,509 | 0.157 |
| 95 | 1,455,949 | 50.1% | +0.43 | 0.58 | 46 | 0.262 | 147 | 42 | 5,094 | 0.776 |
| 91 | 933,161 | 14.4% | +0.10 | 0.63 | 45 | 0.862 | 38 | 115 | 10,386 | 0.128 |
| 78 | 2,212,501 | 28.6% | -0.32 | 0.37 | 45 | 0.355 | 169 | 111 | 1,284 | 0.158 |
| 68 | 1,178,252 | 42.9% | -0.20 | 0.69 | 42 | 0.620 | 53 | 117 | 4,254 | 0.391 |
| 73 | 1,289,064 | 65.0% | +0.29 | 0.57 | 42 | 0.190 | 195 | 54 | 4,079 | 0.913 |
| 79 | 541,330 | 4.1% | +0.14 | 0.64 | 42 | 0.995 | 21 | 32 | 5,046 | 0.012 |
| 83 | 1,711,174 | 21.3% | +0.06 | 0.36 | 40 | 0.844 | 70 | 78 | 928 | 0.015 |
| 82 | 1,564,155 | 47.6% | +0.17 | 0.66 | 39 | 0.290 | 176 | 95 | 922 | 1.022 |
| 100 | 1,072,847 | 54.0% | +0.39 | 0.71 | 39 | 0.222 | 130 | 101 | 12,831 | 0.863 |
| 41 | 1,457,744 | 47.5% | +0.11 | 0.44 | 39 | 0.109 | 364 | 6 | 4,056 | 0.449 |
| 33 | 1,479,280 | 62.6% | +0.40 | 0.57 | 39 | 0.195 | 179 | 5 | 1,509 | 0.173 |
| 62 | 620,760 | 15.2% | +0.12 | 0.60 | 39 | 0.971 | 25 | 69 | 5,916 | 0.032 |
| 47 | 860,785 | 20.6% | +0.04 | 0.70 | 38 | 0.908 | 35 | 53 | 3,530 | 0.160 |
| 120 | 1,031,918 | 31.6% | +0.00 | 0.76 | 38 | 0.793 | 44 | 126 | 7,390 | 0.223 |
| 114 | 2,095,562 | 41.3% | -0.03 | 0.73 | 37 | 0.845 | 88 | 85 | 998 | 0.036 |
| 106 | 1,120,285 | 70.1% | +0.33 | 0.66 | 36 | 0.164 | 162 | 45 | 4,113 | 0.602 |
| 4 | 1,802,181 | 55.1% | +0.36 | 0.56 | 35 | 0.275 | 172 | 59 | 1,899 | 0.370 |
| 81 | 917,924 | 17.9% | +0.08 | 0.72 | 34 | 0.974 | 36 | 20 | 1,475 | 0.039 |
| 51 | 826,468 | 16.8% | +0.09 | 0.77 | 34 | 0.954 | 33 | 116 | 1,950 | 0.075 |
| 2 | 494,147 | 3.3% | +0.15 | 0.70 | 33 | 1.000 | 19 | 15 | 1,339 | 0.022 |
| 110 | 1,234,452 | 32.1% | -0.04 | 0.66 | 32 | 0.815 | 52 | 117 | 1,352 | 0.066 |
| 3 | 1,012,487 | 29.0% | +0.13 | 0.81 | 29 | 0.907 | 41 | 46 | 1,500 | 0.104 |
| 39 | 1,879,851 | 47.8% | -0.29 | 0.51 | 27 | 0.309 | 176 | 70 | 2,963 | 0.093 |
| 71 | 1,250,228 | 46.3% | -0.05 | 0.68 | 26 | 0.349 | 110 | 22 | 4,449 | 0.586 |
| 10 | 1,632,620 | 25.6% | -0.06 | 0.55 | 24 | 0.740 | 75 | 59 | 1,121 | 0.068 |
| 1 | 1,392,003 | 53.1% | -0.08 | 0.68 | 21 | 0.542 | 77 | 85 | 7,753 | 0.136 |
| 63 | 577,921 | 15.7% | +0.15 | 0.64 | 21 | 0.960 | 23 | 11 | 6,628 | 0.094 |
| 58 | 1,577,928 | 60.3% | -0.00 | 0.67 | 19 | 0.316 | 133 | 86 | 2,246 | 0.593 |
| 6 | 1,976,640 | 44.4% | -0.52 | 0.53 | 19 | 0.307 | 187 | 4 | 4,311 | 0.316 |
| 38 | 438,164 | 19.1% | +0.17 | 0.58 | 17 | 0.955 | 18 | 105 | 2,131 | 0.092 |
| 53 | 1,234,630 | 22.4% | +0.04 | 0.64 | 15 | 0.635 | 60 | 47 | 3,530 | 0.422 |
| 11 | 1,193,307 | 40.4% | -0.33 | 0.61 | 15 | 0.422 | 89 | 43 | 3,309 | 0.664 |
| 85 | 1,796,568 | 38.4% | -0.03 | 0.76 | 15 | 0.828 | 76 | 114 | 998 | 0.166 |
| 72 | 745,480 | 14.7% | +0.08 | 0.63 | 15 | 0.979 | 29 | 23 | 8,326 | 0.101 |
| 42 | 2,056,975 | 49.2% | -0.07 | 0.64 | 13 | 0.475 | 119 | 95 | 5,094 | 0.157 |
| 115 | 1,303,852 | 22.7% | +0.06 | 0.62 | 13 | 0.881 | 55 | 69 | 7,593 | 0.088 |
| 76 | 1,518,382 | 48.6% | -0.04 | 0.59 | 12 | 0.362 | 108 | 121 | 3,699 | 0.327 |
| 15 | 1,057,912 | 26.4% | +0.12 | 0.54 | 12 | 0.843 | 44 | 23 | 3,309 | 0.149 |
| 59 | 1,393,419 | 37.4% | +0.06 | 0.47 | 10 | 0.726 | 69 | 10 | 1,121 | 0.100 |
| 121 | 1,516,429 | 34.9% | -0.41 | 0.61 | 10 | 0.499 | 90 | 76 | 3,699 | 0.526 |
| 101 | 1,024,117 | 65.3% | -0.10 | 0.66 | 9 | 0.213 | 146 | 40 | 1,981 | 0.645 |
| 86 | 1,738,271 | 52.5% | -0.16 | 0.74 | 9 | 0.510 | 105 | 58 | 2,246 | 0.247 |
| 112 | 654,170 | 16.5% | +0.00 | 0.67 | 9 | 0.898 | 27 | 23 | 4,951 | 0.163 |
| 125 | 1,838,880 | 22.1% | -0.01 | 0.64 | 8 | 0.899 | 75 | 113 | 2,209 | 0.087 |
| 108 | 859,362 | 40.6% | -0.02 | 0.74 | 8 | 0.427 | 59 | 9 | 1,786 | 0.640 |
| 117 | 1,941,600 | 53.6% | -0.20 | 0.67 | 8 | 0.486 | 109 | 68 | 4,254 | 0.036 |
| 44 | 863,104 | 40.9% | -0.04 | 0.76 | 8 | 0.535 | 42 | 71 | 4,094 | 0.220 |
| 8 | 897,898 | 22.0% | -0.06 | 0.71 | 7 | 0.770 | 40 | 97 | 11,520 | 0.262 |
| 103 | 711,025 | 28.8% | +0.08 | 0.81 | 7 | 0.511 | 43 | 29 | 2,395 | 0.650 |
| 45 | 1,048,592 | 66.4% | +0.18 | 0.61 | 7 | 0.252 | 104 | 106 | 4,113 | 0.313 |
| 92 | 597,220 | 12.4% | +0.10 | 0.69 | 7 | 0.959 | 24 | 15 | 1,591 | 0.118 |
| 97 | 867,631 | 28.6% | -0.07 | 0.65 | 7 | 0.743 | 42 | 12 | 3,919 | 0.297 |
| 12 | 1,244,618 | 62.9% | -0.23 | 0.74 | 6 | 0.290 | 107 | 65 | 3,788 | 0.603 |
| 109 | 606,493 | 12.6% | +0.02 | 0.67 | 6 | 0.942 | 24 | 44 | 837 | 0.559 |
| 37 | 639,311 | 41.1% | +0.00 | 0.67 | 5 | 0.657 | 28 | 67 | 9,366 | 0.227 |
| 29 | 1,045,702 | 44.3% | -0.33 | 0.82 | 5 | 0.368 | 89 | 103 | 2,395 | 0.955 |
| 118 | 827,760 | 24.7% | +0.03 | 0.62 | 2 | 0.900 | 34 | 22 | 1,728 | 0.072 |
| 126 | 1,593,218 | 43.5% | -0.26 | 0.72 | 2 | 0.490 | 108 | 30 | 224 | 0.715 |
| 65 | 1,530,460 | 55.2% | -0.28 | 0.73 | 2 | 0.437 | 109 | 12 | 3,788 | 0.534 |
| 69 | 705,110 | 34.5% | -0.03 | 0.59 | 2 | 0.805 | 31 | 4 | 1,023 | 0.118 |
| 9 | 1,245,143 | 52.9% | -0.26 | 0.75 | 1 | 0.316 | 110 | 57 | 7,693 | 0.536 |
| 23 | 974,940 | 32.0% | +0.00 | 0.62 | 1 | 0.509 | 53 | 15 | 3,309 | 0.413 |
| 67 | 1,558,684 | 51.6% | -0.48 | 0.72 | 0 | 0.287 | 168 | 58 | 6,219 | 0.896 |
| 75 | 1,184,391 | 42.7% | -0.13 | 0.76 | 0 | 0.450 | 80 | 99 | 1,389 | 0.720 |

## Presets

*The same table cut four ways — each is the top 5 on one axis, so a reader with a specific question does not have to re-sort.*

**hardest relative to its geography** — by `zres`:

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 36 | 906,282 | 43.4% | +1.57 | 0.47 | 495 | 0.088 | 274 | 98 | 3,496 | 0.381 |
| 122 | 698,543 | 45.7% | +1.52 | 0.56 | 1,027 | 0.062 | 262 | 82 | 5,849 | 0.663 |
| 13 | 898,423 | 48.1% | +1.17 | 0.44 | 731 | 0.042 | 545 | 54 | 3,365 | 0.062 |
| 98 | 1,278,935 | 42.3% | +1.05 | 0.38 | 378 | 0.118 | 308 | 36 | 3,496 | 0.235 |
| 31 | 981,031 | 58.8% | +0.89 | 0.58 | 331 | 0.147 | 142 | 48 | 3,263 | 0.351 |

**most unmodelled (hard, but not because it moves)** — by `unmodelled`:

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 66 | 2,716,397 | 0.9% | +0.27 | 1.45 | 530 | 1.000 | 165 | 77 | 371 | 0.001 |
| 0 | 1,661,809 | 3.6% | +0.24 | 0.92 | 497 | 1.000 | 65 | 17 | 858 | 0.002 |
| 27 | 2,749,899 | 1.3% | +0.22 | 0.92 | 416 | 1.000 | 108 | 14 | 115 | 0.004 |
| 25 | 1,417,460 | 4.7% | +0.21 | 0.91 | 354 | 1.000 | 55 | 74 | 1,704 | 0.018 |
| 77 | 762,400 | 1.8% | +0.27 | 0.87 | 312 | 1.000 | 50 | 0 | 437 | 0.012 |

**most itinerant (a moving weather state, not a region)** — by `tau50` (ascending):

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 898,423 | 48.1% | +1.17 | 0.44 | 731 | 0.042 | 545 | 54 | 3,365 | 0.062 |
| 122 | 698,543 | 45.7% | +1.52 | 0.56 | 1,027 | 0.062 | 262 | 82 | 5,849 | 0.663 |
| 36 | 906,282 | 43.4% | +1.57 | 0.47 | 495 | 0.088 | 274 | 98 | 3,496 | 0.381 |
| 94 | 1,034,446 | 45.6% | +0.66 | 0.43 | 313 | 0.107 | 187 | 78 | 3,989 | 0.308 |
| 41 | 1,457,744 | 47.5% | +0.11 | 0.44 | 39 | 0.109 | 364 | 6 | 4,056 | 0.449 |

**most remotely confused (`near%` > 30%)** — by `rival_km`:

| cluster | tokens | near% | zres | unmodelled | events | tau50 | core | rival | rival_km | seasonality |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 1,048,150 | 32.5% | +0.62 | 0.55 | 79 | 0.366 | 82 | 7 | 17,078 | 0.863 |
| 26 | 1,254,224 | 64.1% | +0.25 | 0.57 | 52 | 0.235 | 141 | 40 | 14,342 | 0.430 |
| 100 | 1,072,847 | 54.0% | +0.39 | 0.71 | 39 | 0.222 | 130 | 101 | 12,831 | 0.863 |
| 37 | 639,311 | 41.1% | +0.00 | 0.67 | 5 | 0.657 | 28 | 67 | 9,366 | 0.227 |
| 1 | 1,392,003 | 53.1% | -0.08 | 0.68 | 21 | 0.542 | 77 | 85 | 7,753 | 0.136 |

## Global findings (not per-cluster columns)

- **Two kinds of cluster.** 69 of 128 are *territorial* (`tau50 >= 0.5` — a fixed geographic region) and 3 are *itinerant* (`tau50 < 0.1` — a moving weather state whose mass is smeared over hundreds of cells). `maxf` ↔ `near%` correlate at Spearman -0.822: crisp territory ⇒ crisp assignment. That is one axis, not two, and `near%` already carries it — which is why `maxf` is not a column here. What it does dictate is *how to draw the map*.
- **Presence is useless as a map.** Cells where a cluster appears at all span 67 to 4396 (mean 1259) — up to 36% of the globe. Hence the mass-coverage threshold rather than an occupancy one.
- **No cluster is an isolated blob.** `iso = ‖μⱼ−μ_global‖²/trace[j]` maxes out at 0.25 across all 128 clusters: the between-cluster offset is at most that fraction of a cluster's own internal spread. Combined with the near-tie rate and the knee-free merge cascade, **"outlier cluster" is not a meaningful category in this embedding** — outliers exist at the *token* level, which is what the `events` column and each cluster's P3 panel are for.
- **Rivals are usually neighbours.** Median `rival_km` 2,537 km against 10,171 km for a random cluster pair; 48/128 are within 2,000 km. But 30 of 72 contested clusters have a rival more than 4,000 km away — those are real embedding-space confusions, not geographic blur, and nothing else in the repo distinguishes the two cases.
- **No diurnal cycle.** Largest deviation of any UTC-hour share from 0.25, over all 128 clusters: **0.023**. The 6-hourly latents carry a seasonal cycle and essentially no diurnal one, so there is no diurnal panel.

## Extreme events

*How to read this: `residual_map.npy` normalised per cell by median/MAD, cut at the top 0.1%, then grouped into space-time connected components — adjacency is a real HEALPix NESTED neighbour at the same timestep, or the same cell at t+1.*

- threshold: robust **z ≥ 5.58**, **160,000** tokens (0.1% of 160,002,048).
- **14,623** events of ≥2 tokens (of 44,980 components); largest **1,030** tokens; 81.0% of flagged tokens sit inside a multi-token event.

Largest events in the run:

| event | when | where | steps | tokens | peak z | peak residual | host |
|---|---|---|---|---|---|---|---|
| 13596 | 2016-08-26T00 → 2016-08-30T12 | 80.2°N 32.4°E | 19 | 1,030 | 17.3 | 2,746 | c18 |
| 8014 | 2015-09-28T06 → 2015-10-04T18 | 76.3°S 153.6°W | 27 | 947 | 20.2 | 2,952 | c66 |
| 9283 | 2015-12-28T12 → 2016-01-03T06 | 81.4°N 131.2°W | 24 | 936 | 19.5 | 4,007 | c27 |
| 41276 | 2022-03-16T06 → 2022-03-20T12 | 76.0°S 53.5°W | 18 | 895 | 15.0 | 1,074 | c77 |
| 20904 | 2018-02-22T06 → 2018-02-27T06 | 82.6°N 151.2°E | 21 | 852 | 22.4 | 4,628 | c113 |
| 40502 | 2022-01-19T00 → 2022-01-24T12 | 78.6°S 157.7°W | 23 | 718 | 12.9 | 1,726 | c66 |
| 39525 | 2021-11-17T00 → 2021-11-22T00 | 76.5°S 132.9°W | 21 | 663 | 15.6 | 3,907 | c123 |
| 32326 | 2020-05-09T18 → 2020-05-14T18 | 81.0°N 8.1°E | 21 | 649 | 13.8 | 2,586 | c14 |
| 4341 | 2015-01-13T06 → 2015-01-20T12 | 35.7°S 86.0°W | 30 | 581 | 15.9 | 7,152 | c13 |
| 12848 | 2016-07-15T18 → 2016-07-18T06 | 68.9°S 131.3°E | 11 | 572 | 21.2 | 2,075 | c77 |

## Validation

*Run on every `--rank` invocation; these are the invariants that catch a regression.*

| # | check | result |
|---|---|---|
| 1 | f_j.sum(0)*N == full-dataset counts | ✔ 160,002,048 tokens, exact integer match=True |
| 2 | tau_q covers >= q, and q=0.5 == report cells@50% on the sampled files | ✔ min (covered - q) = +2.65e-06; cells@50% identical for 128/128 clusters |
| 3 | seasonal fields weighted by files/season sum back to the annual field | ✔ max |recon - f_annual| = 0.00e+00 (files/season [3124, 3312, 3312, 3273]) |
| 4 | HEALPix-neighbour adjacency merges the two named events | ✔ 2022-03-17..19: largest component 895 tokens (next 21, 20); 2018-02-23..25: largest component 852 tokens (next 13, 4) |
| 5 | zres un-normalised reproduces the model objective | ✔ count-weighted mean residual 1888.12 vs model final_obj_per_token 1887.83 (0.02%); z*1.4826*MAD+median re-sums to 1888.12 (6.4e-14 rel) |
