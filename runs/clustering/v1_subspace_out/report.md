# Clustering report (subspace_kmeans) — `runs/clustering/v1_subspace_out`

*Generated 2026-07-31 15:25 by `analyze_clusters.py`. K=64 affine subspaces of dim 16 in 2048-dim token space, 18,432,000 tokens.*

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
- *`files` = share of the 1500 sampled time steps (latent files, 6-hourly) in which the cluster appears at least once. ≈100% ⇒ always present in time.*
- *`tCV` = coefficient of variation (std / mean) of the cluster's token share across the 10 time deciles. **0 = perfectly constant over time; high ⇒ seasonal or trending.** Computed over populated deciles only, so a sparse sample can't fake a signal.*

Spatial columns are over the 12288 HEALPix cells with data; `cells@50%` = number of cells holding half the cluster's tokens (low = localized); `owned` = cells where this cluster is the most common label; `files` = share of the 1500 sampled time steps where the cluster appears; `tCV` = coefficient of variation of its share across time deciles (0 = constant in time).

| cluster | tokens | share | EVR(top-16) | d80 | cells@50% | owned | files | tCV |
|---|---|---|---|---|---|---|---|---|
| 45 | 584,417 | 3.2% | 0.415 | 10 | 304 | 595 | 100% | 0.09 |
| 63 | 470,382 | 2.6% | 0.458 | 10 | 638 | 363 | 100% | 0.01 |
| 21 | 451,865 | 2.5% | 0.386 | 10 | 632 | 129 | 100% | 0.07 |
| 6 | 449,124 | 2.4% | 0.447 | 11 | 178 | 361 | 100% | 0.09 |
| 32 | 430,777 | 2.3% | 0.410 | 10 | 502 | 515 | 100% | 0.03 |
| 25 | 428,682 | 2.3% | 0.413 | 10 | 501 | 378 | 100% | 0.07 |
| 59 | 402,332 | 2.2% | 0.436 | 10 | 498 | 275 | 100% | 0.03 |
| 0 | 397,508 | 2.2% | 0.441 | 10 | 133 | 266 | 100% | 0.01 |
| 14 | 394,686 | 2.1% | 0.432 | 10 | 590 | 222 | 100% | 0.08 |
| 53 | 370,627 | 2.0% | 0.359 | 11 | 394 | 245 | 100% | 0.09 |
| 28 | 369,699 | 2.0% | 0.488 | 10 | 422 | 185 | 100% | 0.11 |
| 36 | 363,697 | 2.0% | 0.458 | 10 | 719 | 40 | 100% | 0.03 |
| 42 | 360,709 | 2.0% | 0.364 | 11 | 564 | 357 | 100% | 0.05 |
| 54 | 345,673 | 1.9% | 0.458 | 10 | 500 | 143 | 100% | 0.03 |
| 57 | 343,451 | 1.9% | 0.394 | 11 | 433 | 259 | 100% | 0.03 |
| 26 | 343,357 | 1.9% | 0.373 | 11 | 375 | 327 | 100% | 0.09 |
| 10 | 343,079 | 1.9% | 0.397 | 11 | 720 | 199 | 100% | 0.08 |
| 39 | 337,447 | 1.8% | 0.334 | 12 | 186 | 235 | 100% | 0.07 |
| 31 | 334,066 | 1.8% | 0.393 | 11 | 605 | 262 | 100% | 0.03 |
| 56 | 331,721 | 1.8% | 0.381 | 11 | 719 | 74 | 100% | 0.05 |
| 50 | 310,330 | 1.7% | 0.329 | 12 | 162 | 243 | 100% | 0.09 |
| 22 | 296,399 | 1.6% | 0.400 | 10 | 579 | 189 | 100% | 0.03 |
| 52 | 296,298 | 1.6% | 0.341 | 11 | 130 | 269 | 100% | 0.05 |
| 5 | 288,995 | 1.6% | 0.319 | 12 | 575 | 147 | 100% | 0.08 |
| 19 | 284,986 | 1.5% | 0.361 | 11 | 410 | 293 | 100% | 0.04 |
| 44 | 284,571 | 1.5% | 0.337 | 12 | 353 | 200 | 100% | 0.06 |
| 40 | 282,583 | 1.5% | 0.407 | 10 | 674 | 63 | 100% | 0.03 |
| 18 | 280,628 | 1.5% | 0.339 | 11 | 155 | 263 | 100% | 0.05 |
| 47 | 279,483 | 1.5% | 0.332 | 12 | 511 | 135 | 100% | 0.09 |
| 15 | 278,233 | 1.5% | 0.330 | 12 | 470 | 159 | 100% | 0.12 |
| 58 | 272,910 | 1.5% | 0.332 | 12 | 290 | 270 | 100% | 0.09 |
| 23 | 266,842 | 1.4% | 0.352 | 12 | 323 | 195 | 100% | 0.11 |
| 35 | 266,476 | 1.4% | 0.325 | 12 | 304 | 253 | 100% | 0.06 |
| 62 | 266,305 | 1.4% | 0.335 | 12 | 173 | 249 | 100% | 0.14 |
| 24 | 265,301 | 1.4% | 0.367 | 11 | 380 | 102 | 99% | 0.13 |
| 7 | 263,725 | 1.4% | 0.347 | 12 | 480 | 62 | 100% | 0.10 |
| 17 | 262,072 | 1.4% | 0.328 | 12 | 501 | 91 | 100% | 0.04 |
| 41 | 257,959 | 1.4% | 0.325 | 12 | 333 | 220 | 100% | 0.07 |
| 12 | 256,461 | 1.4% | 0.342 | 12 | 360 | 181 | 100% | 0.06 |
| 43 | 255,453 | 1.4% | 0.322 | 12 | 274 | 188 | 100% | 0.10 |
| 61 | 251,433 | 1.4% | 0.361 | 11 | 188 | 228 | 100% | 0.09 |
| 8 | 248,605 | 1.3% | 0.294 | 12 | 172 | 219 | 100% | 0.08 |
| 30 | 247,270 | 1.3% | 0.354 | 11 | 129 | 289 | 100% | 0.06 |
| 4 | 245,235 | 1.3% | 0.362 | 11 | 208 | 202 | 100% | 0.06 |
| 11 | 244,152 | 1.3% | 0.320 | 12 | 537 | 103 | 100% | 0.04 |
| 13 | 236,495 | 1.3% | 0.361 | 11 | 499 | 125 | 100% | 0.08 |
| 2 | 233,132 | 1.3% | 0.396 | 11 | 673 | 52 | 100% | 0.10 |
| 60 | 219,217 | 1.2% | 0.313 | 12 | 187 | 167 | 100% | 0.02 |
| 38 | 218,717 | 1.2% | 0.339 | 12 | 194 | 206 | 100% | 0.13 |
| 34 | 216,743 | 1.2% | 0.304 | 12 | 333 | 86 | 100% | 0.10 |
| 16 | 215,797 | 1.2% | 0.339 | 12 | 379 | 104 | 100% | 0.12 |
| 46 | 214,159 | 1.2% | 0.352 | 11 | 438 | 63 | 100% | 0.07 |
| 48 | 208,742 | 1.1% | 0.350 | 12 | 153 | 175 | 100% | 0.06 |
| 37 | 207,942 | 1.1% | 0.319 | 12 | 355 | 34 | 100% | 0.06 |
| 1 | 207,897 | 1.1% | 0.330 | 12 | 290 | 80 | 100% | 0.11 |
| 55 | 205,599 | 1.1% | 0.310 | 12 | 134 | 162 | 100% | 0.04 |
| 51 | 203,764 | 1.1% | 0.351 | 11 | 244 | 151 | 97% | 0.13 |
| 20 | 203,453 | 1.1% | 0.341 | 11 | 210 | 225 | 99% | 0.17 |
| 33 | 200,349 | 1.1% | 0.398 | 10 | 628 | 2 | 100% | 0.03 |
| 9 | 196,636 | 1.1% | 0.360 | 11 | 325 | 104 | 100% | 0.12 |
| 29 | 192,338 | 1.0% | 0.318 | 12 | 187 | 138 | 100% | 0.15 |
| 3 | 163,374 | 0.9% | 0.361 | 11 | 905 | 0 | 100% | 0.03 |
| 49 | 154,878 | 0.8% | 0.378 | 11 | 52 | 104 | 100% | 0.00 |
| 27 | 126,764 | 0.7% | 0.357 | 11 | 125 | 67 | 87% | 0.16 |

## Temporal & spatial analysis

The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar (monthly) resolution with continent outlines and a smooth heatmap — clearer than a single 12,288-pixel map. Generate it from this run's frozen model + assignments:

  ```bash
  python3 src/analysis/temporal_spatial.py --dir runs/clustering/v1_subspace_out --out runs/clustering/v1_subspace_out/temporal_report.md
  ```

The per-cluster `cells@50%` / `owned` / `files` / `tCV` columns above are the compact in-report summary of that same spatial/temporal structure.

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
