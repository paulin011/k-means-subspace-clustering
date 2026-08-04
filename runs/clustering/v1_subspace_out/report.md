# Clustering report (subspace_kmeans) — `runs/clustering/v1_subspace_out`

*Generated 2026-08-04 13:32 by `analyze_clusters.py`. K=64 affine subspaces of dim 16 in 2048-dim token space, 18,432,000 tokens.*

## Overview

The model groups the 18,432,000 sampled tokens (each a 2048-dim weather-encoder embedding) into 64 clusters, and fits a 16-dimensional flat (an *affine subspace*: a centroid plus a basis of directions) through each one. A token is assigned to whichever cluster leaves the smallest **orthogonal residual** — the part of the token that its cluster's subspace cannot reconstruct.

The core quantities, defined once here:

- **μⱼ** (`model['means'][j]`): the centroid (mean token) of cluster *j*.
- **Uⱼ** (`model['U'][j]`, shape `[2048, 16]`): an orthonormal basis for cluster *j*'s subspace; its columns are PC directions in descending eigenvalue order.
- **Orthogonal residual** of a token *x* under cluster *j*: `‖x − μⱼ‖² − ‖Uⱼᵀ(x − μⱼ)‖²`. The first term is the squared distance to the centroid; the second is the part of that distance the subspace *captures*. What's left is the unexplained residual that the assignment minimises.
- **eigvals** (`model['eigvals'][j]`): the top-16 eigenvalues of cluster *j*'s within-cluster covariance — variance along each kept PC direction.
- **trace** (`model['trace'][j]`): mean squared distance of cluster *j*'s tokens to its centroid μⱼ — the cluster's total within-cluster variance.
- **counts** (`model['counts'][j]`): number of tokens in cluster *j*; **wⱼ = counts[j] / Σcounts** is its population share, used to weight every global average.

## Configuration

*How to read this: these are the run's input settings, taken from `model['config']` (plus `model['sampled_files']` for the true file count). `clusters` is K, `dim` is the subspace dimension d (`dim=0` ⇒ plain k-means). `iters` is the maximum number of training iterations (each one reassigns every token to its best cluster, then refits the centroids and subspaces); `tol` is the convergence threshold: once the fraction of tokens that change cluster in an iteration falls below it, the run stops early instead of using all `iters`. Together they bound how long the run takes. `seed` fixes both the token sample and the cluster initialisation so a run is reproducible.*

| parameter | value |
|---|---|
| src | latents_2 |
| num_files | 1500 |
| tokens_per_file | 12288 |
| clusters | 64 |
| dim | 16 |
| iters | 25 |
| tol | 0.001 |
| linear | False |
| seed | 0 |
| chunk_size | 262144 |
| gpus | 2 |
| tokens analyzed | 18,432,000 |

## Token sample

*How to read this: the model was fit on tokens sampled from a subset of the 13021 latent files. The **fingerprint** is a hash of (tokens-per-file, seed, sorted file ids): two runs with the same fingerprint saw the identical token set and so their metrics can be compared directly. Use the reproduce command to fit a new K or d on exactly these tokens.*

- **Sample fingerprint:** `696d43894e89`
- **Files:** 1500 latent files, 12288 tokens each, seed 0.
- **Reproduce this exact sample** for a new run, with this or any other `cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):

  ```bash
  python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v1_subspace_out/sample.json --seed 0 --tokens-per-file 12288 \
      --clusters <K> --out <new_dir>
  ```
- File ids (first 20 of 1500, full list in `runs/clustering/v1_subspace_out/sample.json`): 21, 25, 50, 77, 100, 102, 106, 109, 114, 117, 122, 126, 130, 140, 142, 150, 156, 158, 160, 167 …

## Convergence

*How to read this: one row per training iteration, from `model['history']`. **objective/token** is the quantity the algorithm minimises — the total reconstruction error divided by the token count, i.e. the average per-token residual the subspaces leave unexplained (for d=0 just the mean squared distance to the nearest centroid); it should fall monotonically and flatten. **labels changed** is the fraction of tokens that switched cluster this iteration. **min/max size** are the smallest and largest cluster token counts that iteration; a min that recovers from a tiny value shows the re-seed guard rescuing a collapsing cluster.*

| iter | objective/token | labels changed | min size | max size |
|---|---|---|---|---|
| 1 | 8056.01 | 100.00% | 153 | 5,261,611 |
| 2 | 3867.60 | 65.48% | 6,139 | 1,529,903 |
| 3 | 3663.00 | 40.02% | 36,265 | 924,498 |
| 4 | 3589.31 | 23.34% | 78,771 | 989,734 |
| 5 | 3556.03 | 16.24% | 93,768 | 967,585 |
| 6 | 3537.25 | 12.59% | 102,951 | 913,302 |
| 7 | 3524.10 | 10.40% | 109,800 | 854,765 |
| 8 | 3513.86 | 8.73% | 114,951 | 802,072 |
| 9 | 3506.54 | 7.34% | 118,064 | 757,457 |
| 10 | 3501.60 | 6.16% | 119,088 | 725,064 |
| 11 | 3497.92 | 5.28% | 119,722 | 700,389 |
| 12 | 3495.07 | 4.60% | 120,554 | 681,003 |
| 13 | 3492.83 | 4.03% | 120,897 | 665,461 |
| 14 | 3491.10 | 3.58% | 121,062 | 652,968 |
| 15 | 3489.68 | 3.22% | 121,268 | 642,510 |
| 16 | 3488.52 | 2.93% | 121,531 | 633,881 |
| 17 | 3487.54 | 2.69% | 121,847 | 626,253 |
| 18 | 3486.70 | 2.49% | 122,414 | 619,300 |
| 19 | 3485.98 | 2.30% | 123,185 | 613,110 |
| 20 | 3485.34 | 2.14% | 123,901 | 607,460 |
| 21 | 3484.80 | 1.98% | 124,697 | 602,101 |
| 22 | 3484.33 | 1.84% | 125,243 | 597,422 |
| 23 | 3483.94 | 1.72% | 125,737 | 592,889 |
| 24 | 3483.58 | 1.61% | 126,322 | 588,483 |
| 25 | 3483.27 | 1.52% | 126,764 | 584,417 |

## Global variance decomposition

*How to read this: the **law of total variance** lets us cut the single, uninterpretable total spread of the tokens into perpendicular pieces that each audit a different part of the model. Writing μ_global for the population-weighted mean of all centroids, the **total variance** splits as:*

*`E‖x − μ_global‖² = between + within`*  *(centroids vs. inside clusters), and `within` splits again into `captured + residual` (along the subspaces vs. off them). The pieces are perpendicular, so their squared lengths add to the whole.*

- *`between = Σⱼ wⱼ ‖μⱼ − μ_global‖²` — spread of the cluster centroids (from `means`, `counts`).*
- *`within  = Σⱼ wⱼ · trace[j]` — average spread of tokens around their own centroid (from `trace`, `counts`).*
- *`captured = Σⱼ wⱼ · Σ eigvals[j]` — the slice of `within` that the subspaces reconstruct (from `eigvals`); `residual = within − captured` is what's left over.*

*The point of the split is to read the total as a **budget**: how much variation is explained by **which** cluster a token is in, how much by **where it sits inside** its cluster's subspace, and how much the model **misses**. The model's objective is to minimise that last piece (residual).*

Total token variance E‖x−μ_global‖² = **5999**, split into:

- **8.5%** is `between / total`. It is variance explained purely by **which** cluster a token is in, before looking at anything inside the cluster.
- **33.5%** is `captured / total`. It is the chunk of `within` that the top-16 subspace directions reconstruct, expressed as a fraction of the grand total. Note it is **not** `captured / within`; it is divided by 5999, the same denominator as the other two, which is what lets all three add to 100%.
- **58.1%** is `residual / total`, the leftover within-cluster variance no subspace direction reaches. This is exactly what the assignment rule minimises.

**Count-weighted within-cluster EVR(top-16): 0.374** — population-weighted average of the per-cluster EVR in the table below: of a cluster's *own* internal variance, its 16 subspace directions recover about 37%. (This is `captured / within`; the **captured** line above was `captured / total`, hence larger here.)

**Dimensions for 80% of within-cluster variance: min 10 / median 11 / max 12** — the **d80** column below. A PC direction is one of PCA's perpendicular axes of variation inside a cluster (columns of `U`, most-spread first); d80 counts how many reach 80% of the kept total. Capped at d+1=17 = a truncation warning. Your max is 12, below the cap → no cluster truncated, d=16 has headroom.

## Clusters (sorted by size)

*How to read this: one row per cluster, largest first. Each column is computed from `assignments.pt` (the per-token `label` / `cell_id` / `file_id`) and `model.pt`. The columns, with their formulas:*

- *`tokens` = `counts[j]`; `share` = wⱼ = tokens / 18,432,000.*
- *`EVR(top-16)` = `Σ eigvals[j] / trace[j]` — fraction of this cluster's own variance captured by its subspace (1.0 = the subspace explains the cluster perfectly; near the global average ⇒ d truncates the spectrum).*
- *`d80` = smallest number of leading PC directions whose eigenvalues reach 80% of `Σ eigvals[j]` (capped at d+1=17 when even all 16 fall short). Low d80 ⇒ a few directions dominate; d80 ≈ d ⇒ a flat spectrum the subspace truncates.*
- *`cells@50%` = how many of the 12288 HEALPix grid cells hold the top 50% of this cluster's tokens. **Low = geographically localized**, high = spread over the globe.*
- *`owned` = number of cells where this cluster is the single most common label (the cell's *dominant* cluster). A cluster can be present everywhere yet own few cells.*
- *`files@50%` = share of the 1500 sampled time steps (latent files, 6-hourly) holding the top 50% of this cluster's tokens — the time-axis twin of `cells@50%`. **50% = spread perfectly evenly over time; lower = concentrated into fewer snapshots (bursty / seasonal).** This replaces an earlier `files` column that counted time steps where the cluster appeared *at all*: with 12,288 cells over 64 clusters that is true almost everywhere, so it read 100% for most clusters and carried no information.*
- *`maxAff` = this cluster's subspace affinity to its **nearest** neighbour, `maxⱼ≠ᵢ ‖UᵢᵀUⱼ‖²_F / d` ∈ [0,1]. A per-cluster separation score: **high ⇒ some other cluster spans nearly the same directions**, so this row is a merge candidate. The affinity table below lists only the top pairs, so a near-duplicate cluster is invisible there unless its pair happens to rank; this column always shows it.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data. `share` is printed to 2 decimals because the whole range is narrow (0.69%–3.17% on this run) and 1 decimal collapses distinct clusters onto the same value.

| cluster | tokens | share | EVR(top-16) | d80 | cells@50% | owned | files@50% | tCV | maxAff |
|---|---|---|---|---|---|---|---|---|---|
| 45 | 584,417 | 3.17% | 0.415 | 10 | 304 | 595 | 25.1% | 0.093 | 0.633 |
| 63 | 470,382 | 2.55% | 0.458 | 10 | 638 | 363 | 43.2% | 0.012 | 0.651 |
| 21 | 451,865 | 2.45% | 0.386 | 10 | 632 | 129 | 33.3% | 0.074 | 0.618 |
| 6 | 449,124 | 2.44% | 0.447 | 11 | 178 | 361 | 34.6% | 0.088 | 0.654 |
| 32 | 430,777 | 2.34% | 0.410 | 10 | 502 | 515 | 44.7% | 0.025 | 0.614 |
| 25 | 428,682 | 2.33% | 0.413 | 10 | 501 | 378 | 28.7% | 0.069 | 0.671 |
| 59 | 402,332 | 2.18% | 0.436 | 10 | 498 | 275 | 40.2% | 0.027 | 0.636 |
| 0 | 397,508 | 2.16% | 0.441 | 10 | 133 | 266 | 48.5% | 0.012 | 0.530 |
| 14 | 394,686 | 2.14% | 0.432 | 10 | 590 | 222 | 30.7% | 0.084 | 0.671 |
| 53 | 370,627 | 2.01% | 0.359 | 11 | 394 | 245 | 27.0% | 0.086 | 0.613 |
| 28 | 369,699 | 2.01% | 0.488 | 10 | 422 | 185 | 37.7% | 0.112 | 0.654 |
| 36 | 363,697 | 1.97% | 0.458 | 10 | 719 | 40 | 40.2% | 0.025 | 0.651 |
| 42 | 360,709 | 1.96% | 0.364 | 11 | 564 | 357 | 36.5% | 0.053 | 0.579 |
| 54 | 345,673 | 1.88% | 0.458 | 10 | 500 | 143 | 40.6% | 0.030 | 0.626 |
| 57 | 343,451 | 1.86% | 0.394 | 11 | 433 | 259 | 41.2% | 0.035 | 0.613 |
| 26 | 343,357 | 1.86% | 0.373 | 11 | 375 | 327 | 30.9% | 0.089 | 0.663 |
| 10 | 343,079 | 1.86% | 0.397 | 11 | 720 | 199 | 39.4% | 0.077 | 0.595 |
| 39 | 337,447 | 1.83% | 0.334 | 12 | 186 | 235 | 41.2% | 0.070 | 0.625 |
| 31 | 334,066 | 1.81% | 0.393 | 11 | 605 | 262 | 41.4% | 0.026 | 0.663 |
| 56 | 331,721 | 1.80% | 0.381 | 11 | 719 | 74 | 38.8% | 0.050 | 0.623 |
| 50 | 310,330 | 1.68% | 0.329 | 12 | 162 | 243 | 35.9% | 0.091 | 0.557 |
| 22 | 296,399 | 1.61% | 0.400 | 10 | 579 | 189 | 43.3% | 0.026 | 0.614 |
| 52 | 296,298 | 1.61% | 0.341 | 11 | 130 | 269 | 37.4% | 0.047 | 0.559 |
| 5 | 288,995 | 1.57% | 0.319 | 12 | 575 | 147 | 33.7% | 0.076 | 0.652 |
| 19 | 284,986 | 1.55% | 0.361 | 11 | 410 | 293 | 36.1% | 0.042 | 0.608 |
| 44 | 284,571 | 1.54% | 0.337 | 12 | 353 | 200 | 29.3% | 0.056 | 0.625 |
| 40 | 282,583 | 1.53% | 0.407 | 10 | 674 | 63 | 40.9% | 0.034 | 0.578 |
| 18 | 280,628 | 1.52% | 0.339 | 11 | 155 | 263 | 40.2% | 0.053 | 0.600 |
| 47 | 279,483 | 1.52% | 0.332 | 12 | 511 | 135 | 30.6% | 0.090 | 0.652 |
| 15 | 278,233 | 1.51% | 0.330 | 12 | 470 | 159 | 30.5% | 0.117 | 0.610 |
| 58 | 272,910 | 1.48% | 0.332 | 12 | 290 | 270 | 32.9% | 0.089 | 0.597 |
| 23 | 266,842 | 1.45% | 0.352 | 12 | 323 | 195 | 27.6% | 0.109 | 0.548 |
| 35 | 266,476 | 1.45% | 0.325 | 12 | 304 | 253 | 31.3% | 0.058 | 0.584 |
| 62 | 266,305 | 1.44% | 0.335 | 12 | 173 | 249 | 27.2% | 0.137 | 0.566 |
| 24 | 265,301 | 1.44% | 0.367 | 11 | 380 | 102 | 22.7% | 0.128 | 0.620 |
| 7 | 263,725 | 1.43% | 0.347 | 12 | 480 | 62 | 24.7% | 0.103 | 0.577 |
| 17 | 262,072 | 1.42% | 0.328 | 12 | 501 | 91 | 40.7% | 0.037 | 0.597 |
| 41 | 257,959 | 1.40% | 0.325 | 12 | 333 | 220 | 39.6% | 0.075 | 0.643 |
| 12 | 256,461 | 1.39% | 0.342 | 12 | 360 | 181 | 25.1% | 0.060 | 0.560 |
| 43 | 255,453 | 1.39% | 0.322 | 12 | 274 | 188 | 37.8% | 0.104 | 0.540 |
| 61 | 251,433 | 1.36% | 0.361 | 11 | 188 | 228 | 32.2% | 0.094 | 0.600 |
| 8 | 248,605 | 1.35% | 0.294 | 12 | 172 | 219 | 39.1% | 0.076 | 0.574 |
| 30 | 247,270 | 1.34% | 0.354 | 11 | 129 | 289 | 39.5% | 0.055 | 0.647 |
| 4 | 245,235 | 1.33% | 0.362 | 11 | 208 | 202 | 37.7% | 0.065 | 0.647 |
| 11 | 244,152 | 1.32% | 0.320 | 12 | 537 | 103 | 37.7% | 0.037 | 0.634 |
| 13 | 236,495 | 1.28% | 0.361 | 11 | 499 | 125 | 36.5% | 0.078 | 0.638 |
| 2 | 233,132 | 1.26% | 0.396 | 11 | 673 | 52 | 38.8% | 0.097 | 0.609 |
| 60 | 219,217 | 1.19% | 0.313 | 12 | 187 | 167 | 41.5% | 0.019 | 0.545 |
| 38 | 218,717 | 1.19% | 0.339 | 12 | 194 | 206 | 22.9% | 0.129 | 0.566 |
| 34 | 216,743 | 1.18% | 0.304 | 12 | 333 | 86 | 37.6% | 0.095 | 0.564 |
| 16 | 215,797 | 1.17% | 0.339 | 12 | 379 | 104 | 28.1% | 0.122 | 0.620 |
| 46 | 214,159 | 1.16% | 0.352 | 11 | 438 | 63 | 32.9% | 0.071 | 0.638 |
| 48 | 208,742 | 1.13% | 0.350 | 12 | 153 | 175 | 35.6% | 0.056 | 0.549 |
| 37 | 207,942 | 1.13% | 0.319 | 12 | 355 | 34 | 25.3% | 0.064 | 0.643 |
| 1 | 207,897 | 1.13% | 0.330 | 12 | 290 | 80 | 19.4% | 0.110 | 0.508 |
| 55 | 205,599 | 1.12% | 0.310 | 12 | 134 | 162 | 41.4% | 0.041 | 0.502 |
| 51 | 203,764 | 1.11% | 0.351 | 11 | 244 | 151 | 19.7% | 0.127 | 0.611 |
| 20 | 203,453 | 1.10% | 0.341 | 11 | 210 | 225 | 16.9% | 0.172 | 0.593 |
| 33 | 200,349 | 1.09% | 0.398 | 10 | 628 | 2 | 40.6% | 0.029 | 0.548 |
| 9 | 196,636 | 1.07% | 0.360 | 11 | 325 | 104 | 19.5% | 0.122 | 0.620 |
| 29 | 192,338 | 1.04% | 0.318 | 12 | 187 | 138 | 22.7% | 0.146 | 0.597 |
| 3 | 163,374 | 0.89% | 0.361 | 11 | 905 | 0 | 42.3% | 0.030 | 0.583 |
| 49 | 154,878 | 0.84% | 0.378 | 11 | 52 | 104 | 48.6% | 0.005 | 0.420 |
| 27 | 126,764 | 0.69% | 0.357 | 11 | 125 | 67 | 17.1% | 0.163 | 0.486 |

**Cluster health flags.** 
`owned == 0` (never the majority label in any cell, so it exists only as a minority everywhere — check it is a real mode and not a leftover): **3**

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v1_subspace_out --out runs/clustering/v1_subspace_out/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files@50%` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

## Subspace affinity between clusters

*How to read this: a similarity score between every pair of cluster subspaces. For clusters *i*, *j*, **Affinity(i,j) = ‖UᵢᵀUⱼ‖²_F / d**, the mean of the squared cosines of the principal angles between the two subspaces: **1 = identical span, 0 = orthogonal (completely different directions of variation).** It's computed from the bases in `model['U']` (the centroids are not involved; the side column **mean-vector cosine** = cos∠(μᵢ, μⱼ) compares the centroids separately). A high affinity pair is a candidate for **merging** — a hint K may be too large; if all off-diagonal values are low, the clusters are genuinely distinct regimes.*

Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / 16 ∈ [0,1]: mean squared cosine of the principal angles between the two subspaces (1 = identical span, 0 = orthogonal). High-affinity pairs are candidates for merging (K may be too large); uniformly low values mean genuinely distinct regimes.

Off-diagonal affinity: median 0.325, mean 0.341, max 0.671 (a low median with a higher max ⇒ most clusters are distinct, only a handful overlap).

| pair | subspace affinity | mean-vector cosine |
|---|---|---|
| 14 ↔ 25 | 0.671 | 0.269 |
| 26 ↔ 31 | 0.663 | 0.571 |
| 6 ↔ 28 | 0.654 | 0.522 |
| 5 ↔ 47 | 0.652 | 0.757 |
| 36 ↔ 63 | 0.651 | 0.261 |
| 4 ↔ 30 | 0.647 | 0.448 |
| 37 ↔ 41 | 0.643 | 0.891 |
| 13 ↔ 46 | 0.638 | 0.444 |
| 14 ↔ 59 | 0.636 | -0.023 |
| 5 ↔ 11 | 0.634 | 0.660 |
| 25 ↔ 45 | 0.633 | 0.238 |
| 36 ↔ 54 | 0.626 | 0.528 |

## Interpretation notes

- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the cluster is a **geographic regime** (region/surface type), stable in time.
- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the temporal & spatial report (`temporal_spatial.py`).
- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the spectrum; re-run with larger `--dim` to capture more structure.
- *High subspace affinity between two clusters* ⇒ they vary along nearly the same directions; consider lowering K or merging that pair.
- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.
