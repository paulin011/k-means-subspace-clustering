# Per-file regime signature + residual — `runs/signatures/v12_k256to128_d64`

*Generated 2026-08-13 13:12 by `file_signature.py`. Frozen model `runs/clustering/v12_k256to128_d64` (K=128, d=64, affine=True, fingerprint `82ca602ed7e7`). 13,021 files × 12288 cells = 160,002,048 tokens assigned on a single GPU.*

## What this is

For every latent timestep the frozen v6 model assigns each of its 12,288 cells to the nearest affine subspace (`residual_j = ‖x−μ_j‖²−‖Uⱼᵀ(x−μ_j)‖²`) and reduces the result to a compact per-file summary. This is the coordinate system for **goal 1 (which timestamps to train on)**: `mix` is a low-dim fingerprint of each snapshot, `residual_frac` flags the anomalous / hard-to-encode ones, and `rare_exposure` flags snapshots rich in rare regimes.

## Correctness check

The count-weighted mean of `mean_residual` over all 13,021 files is **1877.22** (model `final_obj_per_token` = 1877.04) and the mean `residual_frac` is **31.3%** (in-sample / held-out = 31.5% in the run report). A full-dataset pass that reproduces the trained objective confirms the assignment kernel is correct.  ✔ matches

## Per-file residual fraction (anomaly / hardness)

`residual_frac` = the share of a snapshot's variance the regime+subspace model leaves unexplained (globally ~31.5%). High outliers are the **anomalous timestamps** — candidates for hard-example upweighting.

- distribution (min/p1/p25/median/p75/p99/max): 30.1% / 30.4% / 31.0% / 31.3% / 31.6% / 32.2% / 32.5%
- monthly mean `residual_frac` (does anomaly concentrate seasonally?): Jan 31.4%, Feb 31.4%, Mar 31.7%, Apr 31.7%, May 31.5%, Jun 31.1%, Jul 30.9%, Aug 31.0%, Sep 31.0%, Oct 31.4%, Nov 31.5%, Dec 31.3%

| rank | file | datetime | resid_frac | rare_exp | dom_cluster |
|---|---|---|---|---|---|
| 1 | 6063 | 2018-02-24T18 | 32.5% | 14.0% | 17 |
| 2 | 6051 | 2018-02-21T18 | 32.4% | 14.1% | 112 |
| 3 | 10532 | 2021-03-18T00 | 32.4% | 14.3% | 124 |
| 4 | 6053 | 2018-02-22T06 | 32.4% | 14.2% | 54 |
| 5 | 6056 | 2018-02-23T00 | 32.4% | 14.2% | 89 |
| 6 | 6065 | 2018-02-25T06 | 32.4% | 14.1% | 124 |
| 7 | 6052 | 2018-02-22T00 | 32.4% | 14.2% | 54 |
| 8 | 6066 | 2018-02-25T12 | 32.4% | 14.1% | 112 |
| 9 | 6067 | 2018-02-25T18 | 32.4% | 13.8% | 124 |
| 10 | 4596 | 2017-02-23T00 | 32.4% | 13.8% | 54 |
| 11 | 4595 | 2017-02-22T18 | 32.4% | 13.6% | 54 |
| 12 | 6064 | 2018-02-25T00 | 32.4% | 14.1% | 124 |

## Rare-regime exposure

`rare_exposure` = share of a snapshot's tokens in the 32 globally-rarest clusters (bottom quartile, share ≤ 0.562%: 2, 3, 7, 8, 9, 24, 29, 32, 38, 40, 45, 46, 51, 52, 53, 57, 59, 64, 66, 69, 73, 76, 82, 87, 93, 94, 99, 102, 104, 115, 126, 127). High values mark snapshots rich in rare regimes — candidates for class-balancing.

| rank | file | datetime | rare_exposure | resid_frac | dom_cluster |
|---|---|---|---|---|---|
| 1 | 2576 | 2015-10-07T00 | 15.7% | 31.4% | 109 |
| 2 | 2573 | 2015-10-06T06 | 15.6% | 31.4% | 109 |
| 3 | 849 | 2014-08-01T06 | 15.6% | 31.1% | 41 |
| 4 | 853 | 2014-08-02T06 | 15.6% | 31.0% | 41 |
| 5 | 2571 | 2015-10-05T18 | 15.5% | 31.5% | 109 |
| 6 | 879 | 2014-08-08T18 | 15.5% | 30.6% | 41 |
| 7 | 8373 | 2019-09-25T06 | 15.5% | 31.8% | 41 |
| 8 | 8378 | 2019-09-26T12 | 15.5% | 32.1% | 41 |
| 9 | 2572 | 2015-10-06T00 | 15.5% | 31.5% | 109 |
| 10 | 2575 | 2015-10-06T18 | 15.5% | 31.4% | 109 |
| 11 | 8374 | 2019-09-25T12 | 15.5% | 31.8% | 41 |
| 12 | 850 | 2014-08-01T12 | 15.5% | 31.0% | 41 |

## Per-cell outlier map

*How to read this: `residual_frac` above scores a whole snapshot. `residual_map.npy` `[N, 12288]` keeps the **same residual per HEALPix cell**, so an outlier can be localised to where on the globe it happened instead of only when. It is the identical quantity — averaging a row reproduces that row's `mean_residual` exactly — just not reduced over cells.*

Raw residuals are **not comparable between cells**: an intrinsically hard cell (storm track) outscores an easy one (subtropical ocean) in every snapshot. Score outliers against each cell's own temporal norm, using the climatology vectors stored in `signatures.npz`:

```python
import numpy as np
R = np.load('runs/signatures/v12_k256to128_d64/residual_map.npy', mmap_mode='r')   # [N, 12288]
s = np.load('runs/signatures/v12_k256to128_d64/signatures.npz')
z = (R - s['cell_mean_residual']) / np.maximum(s['cell_std_residual'], 1e-6)
# z[t, cell] > 4  ->  this cell is far outside its own normal range at step t
```

- per-cell mean residual across the 13,021 steps: min **369** / median **1945** / max **3528** (9.6× spread — this is exactly why the normalisation is needed).
- per-cell temporal std: median **359** (median std/mean = 19.2%).
- hardest cells (highest mean residual, NESTED ids): 8064, 5611, 5946, 5554, 12254, 7505, 12252, 5951, 7659, 6871.
- easiest cells: 10306, 10270, 10269, 10309, 9282, 11294, 10424, 11293, 9285, 11448.

`label_map.npy` `[N, 12288]` int16 carries the assigned cluster per cell over the **whole dataset** — the dynamic counterpart to `assignments.pt`, which only covers the sampled files. Both arrays sit on the same NESTED `[T, 12288]` grid as `err_persist.npy`, so they gather onto each other with no remapping: intersect them to separate *anomalous because rapidly changing* (high persistence error) from *anomalous because unmodelled* (high residual, low persistence error).


## Assignment margin (per-cluster separation)

*How to read this: every token is ranked against all K subspaces, so the **runner-up is free**. `margin = (R₂−R₁)/R₁` asks how much worse the second-best subspace is; a token with `margin < 10%` is a **near-tie** — the partition assigns it, but barely. This is the K-subspaces analogue of the silhouette coefficient, which does not transfer here because it assumes Euclidean distance to a centroid and spherical clusters — the wrong geometry for subspaces.*

- **28.0%** of all 160,002,048 tokens are near-ties (margin < 10%); median per-token margin per cluster ranges **0.09 … 1.06**.
- This decomposes that single global ambiguity figure **by cluster**: it is not spread evenly, so the worst rows below are the ones where hard labels actually throw information away (and where the soft/MPPCA responsibilities matter most).
- `maxAff` in the run report measures the *geometric* angle between two subspaces without touching data; `near_frac` measures whether they actually **contest the same tokens**. The two **correlate strongly** (v6: Pearson +0.77) — overlapping orientation does tend to mean contested tokens — but `near_frac` still adds information: 41% of its variance is unexplained by `maxAff`, its range is far wider, and it names a different closest rival for 52% of clusters. `maxAff` sees only orientation, not where the means sit or where the data is dense.

Least-separated clusters (highest near-tie share of their own tokens):

| rank | cluster | tokens | share | margin_mean | near_frac | runner-up | runner-up share |
|---|---|---|---|---|---|---|---|
| 1 | 79 | 1,698,460 | 1.06% | 0.088 | 64.6% | 35 | 22.3% |
| 2 | 18 | 1,754,598 | 1.10% | 0.093 | 63.8% | 112 | 25.1% |
| 3 | 23 | 1,825,219 | 1.14% | 0.093 | 60.4% | 106 | 19.0% |
| 4 | 112 | 2,152,927 | 1.35% | 0.103 | 59.1% | 18 | 25.5% |
| 5 | 14 | 1,712,666 | 1.07% | 0.096 | 58.8% | 95 | 19.1% |
| 6 | 17 | 1,798,380 | 1.12% | 0.105 | 57.8% | 101 | 21.0% |
| 7 | 114 | 1,688,276 | 1.06% | 0.103 | 57.6% | 23 | 19.7% |
| 8 | 35 | 1,472,645 | 0.92% | 0.106 | 56.9% | 79 | 27.5% |
| 9 | 1 | 1,698,607 | 1.06% | 0.108 | 55.2% | 114 | 18.9% |
| 10 | 85 | 1,123,200 | 0.70% | 0.117 | 53.3% | 70 | 22.4% |
| 11 | 21 | 1,872,120 | 1.17% | 0.113 | 53.0% | 34 | 23.5% |
| 12 | 68 | 2,810,654 | 1.76% | 0.111 | 52.5% | 22 | 25.2% |
| 13 | 89 | 2,238,766 | 1.40% | 0.118 | 51.4% | 118 | 27.7% |
| 14 | 54 | 1,440,093 | 0.90% | 0.117 | 50.7% | 5 | 20.1% |
| 15 | 101 | 1,865,940 | 1.17% | 0.114 | 50.6% | 43 | 33.7% |

Best-separated clusters (lowest near-tie share):

| rank | cluster | tokens | share | margin_mean | near_frac | runner-up | runner-up share |
|---|---|---|---|---|---|---|---|
| 1 | 96 | 989,471 | 0.62% | 0.959 | 0.0% | 15 | 37.0% |
| 2 | 15 | 1,158,468 | 0.72% | 0.714 | 0.1% | 113 | 52.2% |
| 3 | 113 | 1,041,041 | 0.65% | 0.692 | 0.3% | 15 | 37.1% |
| 4 | 28 | 988,097 | 0.62% | 0.641 | 0.4% | 80 | 35.5% |
| 5 | 88 | 991,523 | 0.62% | 0.781 | 0.4% | 117 | 55.8% |

Per-file: `mean_margin` **0.300** (p1 0.281 / p99 0.318), `near_frac` **28.0%** (min 23.1% / max 33.1%) — a timestep whose cells are unusually contested is one the partition describes poorly, an independent hardness axis alongside `residual_frac` (corr = +0.81).

Full table: `cluster_margin.csv` (all 128 clusters). Merge it into the run report with `analyze_clusters.py --margins runs/signatures/v12_k256to128_d64/cluster_margin.csv`.


## How to use for goal 1 (timestamp selection)

- **Diversity / coverage sampling** (anti-redundancy): `cluster_mix.csv` is the feature matrix; run k-center / farthest-point in mix-space to pick a minimal covering subset — consecutive timesteps are near-duplicates in mix (clusters are geographic & time-stable).
- **Seasonal stratification**: the `month` column + the seasonal clusters in `temporal_report.md` give per-month upweighting (shoulder months Apr→May, Oct→Nov).
- **Rare-regime balancing**: upweight files with high `rare_exposure`.
- **Hard-example upweighting**: upweight high-`residual_frac` rows.
- **The closed loop (most principled)**: re-run this script with the model's *forecast errors* as input instead of embeddings — the resulting signatures flag the regimes where the model fails, i.e. the timestamps that most deserve training weight.

## Files

- `signatures.npz` — canonical arrays (mix[N,K], all per-file stats, datetime).
- `file_summary.csv` — one row per file (human-readable, sorted by file_id).
- `cluster_mix.csv` — wide mix matrix for pandas / sklearn timestamp selection.
- `manifest.json` — provenance, formulas, rare-cluster list.
- `cluster_margin.csv` — per-cluster separation (margin / near-tie / runner-up).
- `residual_map.npy` — `[13021, 12288]` float32 per-cell residual (0.60 GiB; load with `mmap_mode='r'`).
- `label_map.npy` — `[13021, 12288]` int16 per-cell assigned cluster.
- `margin_map.npy` — `[13021, 12288]` float16 per-cell assignment margin (0.30 GiB; low = contested cell).