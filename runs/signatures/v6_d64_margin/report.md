# Per-file regime signature + residual — `runs/signatures/v6_d64_margin`

*Generated 2026-08-04 10:39 by `file_signature.py`. Frozen model `runs/clustering/v6_subspace_big_d64` (K=128, d=64, affine=True, fingerprint `82ca602ed7e7`). 13,021 files × 12288 cells = 160,002,048 tokens assigned on a single GPU.*

## What this is

For every latent timestep the frozen v6 model assigns each of its 12,288 cells to the nearest affine subspace (`residual_j = ‖x−μ_j‖²−‖Uⱼᵀ(x−μ_j)‖²`) and reduces the result to a compact per-file summary. This is the coordinate system for **goal 1 (which timestamps to train on)**: `mix` is a low-dim fingerprint of each snapshot, `residual_frac` flags the anomalous / hard-to-encode ones, and `rare_exposure` flags snapshots rich in rare regimes.

## Correctness check

The count-weighted mean of `mean_residual` over all 13,021 files is **1888.12** (model `final_obj_per_token` = 1887.83) and the mean `residual_frac` is **31.5%** (in-sample / held-out = 31.5% in the run report). A full-dataset pass that reproduces the trained objective confirms the assignment kernel is correct.  ✔ matches

## Per-file residual fraction (anomaly / hardness)

`residual_frac` = the share of a snapshot's variance the regime+subspace model leaves unexplained (globally ~31.5%). High outliers are the **anomalous timestamps** — candidates for hard-example upweighting.

- distribution (min/p1/p25/median/p75/p99/max): 30.2% / 30.6% / 31.2% / 31.5% / 31.8% / 32.4% / 32.8%
- monthly mean `residual_frac` (does anomaly concentrate seasonally?): Jan 31.4%, Feb 31.5%, Mar 31.8%, Apr 31.9%, May 31.7%, Jun 31.3%, Jul 31.1%, Aug 31.1%, Sep 31.3%, Oct 31.7%, Nov 31.7%, Dec 31.4%

| rank | file | datetime | resid_frac | rare_exp | dom_cluster |
|---|---|---|---|---|---|
| 1 | 10686 | 2021-04-25T12 | 32.8% | 14.0% | 27 |
| 2 | 10685 | 2021-04-25T06 | 32.7% | 14.1% | 27 |
| 3 | 6209 | 2018-04-02T06 | 32.6% | 14.7% | 60 |
| 4 | 4592 | 2017-02-22T00 | 32.6% | 12.8% | 60 |
| 5 | 4593 | 2017-02-22T06 | 32.6% | 12.8% | 60 |
| 6 | 6210 | 2018-04-02T12 | 32.6% | 14.7% | 60 |
| 7 | 10687 | 2021-04-25T18 | 32.6% | 14.1% | 27 |
| 8 | 10684 | 2021-04-25T00 | 32.6% | 13.9% | 27 |
| 9 | 4590 | 2017-02-21T12 | 32.6% | 13.0% | 67 |
| 10 | 4596 | 2017-02-23T00 | 32.6% | 12.7% | 60 |
| 11 | 6053 | 2018-02-22T06 | 32.6% | 13.4% | 60 |
| 12 | 10532 | 2021-03-18T00 | 32.6% | 13.6% | 60 |

## Rare-regime exposure

`rare_exposure` = share of a snapshot's tokens in the 32 globally-rarest clusters (bottom quartile, share ≤ 0.561%: 2, 8, 14, 20, 28, 32, 37, 38, 44, 47, 49, 51, 55, 62, 63, 69, 72, 77, 79, 87, 92, 96, 97, 102, 103, 108, 109, 112, 116, 118, 122, 124). High values mark snapshots rich in rare regimes — candidates for class-balancing.

| rank | file | datetime | rare_exposure | resid_frac | dom_cluster |
|---|---|---|---|---|---|
| 1 | 11031 | 2021-07-20T18 | 16.3% | 31.0% | 88 |
| 2 | 11035 | 2021-07-21T18 | 16.2% | 30.7% | 88 |
| 3 | 11029 | 2021-07-20T06 | 16.2% | 30.9% | 88 |
| 4 | 11033 | 2021-07-21T06 | 16.2% | 31.0% | 88 |
| 5 | 11030 | 2021-07-20T12 | 16.2% | 31.1% | 88 |
| 6 | 5209 | 2017-07-26T06 | 16.2% | 30.9% | 82 |
| 7 | 5213 | 2017-07-27T06 | 16.2% | 30.8% | 82 |
| 8 | 11034 | 2021-07-21T12 | 16.1% | 30.9% | 88 |
| 9 | 11036 | 2021-07-22T00 | 16.1% | 30.8% | 88 |
| 10 | 6649 | 2018-07-21T06 | 16.1% | 30.7% | 88 |
| 11 | 6653 | 2018-07-22T06 | 16.1% | 30.8% | 88 |
| 12 | 11042 | 2021-07-23T12 | 16.1% | 30.6% | 88 |

## Per-cell outlier map

*How to read this: `residual_frac` above scores a whole snapshot. `residual_map.npy` `[N, 12288]` keeps the **same residual per HEALPix cell**, so an outlier can be localised to where on the globe it happened instead of only when. It is the identical quantity — averaging a row reproduces that row's `mean_residual` exactly — just not reduced over cells.*

Raw residuals are **not comparable between cells**: an intrinsically hard cell (storm track) outscores an easy one (subtropical ocean) in every snapshot. Score outliers against each cell's own temporal norm, using the climatology vectors stored in `signatures.npz`:

```python
import numpy as np
R = np.load('runs/signatures/v6_d64_margin/residual_map.npy', mmap_mode='r')   # [N, 12288]
s = np.load('runs/signatures/v6_d64_margin/signatures.npz')
z = (R - s['cell_mean_residual']) / np.maximum(s['cell_std_residual'], 1e-6)
# z[t, cell] > 4  ->  this cell is far outside its own normal range at step t
```

- per-cell mean residual across the 13,021 steps: min **273** / median **1923** / max **3386** (12.4× spread — this is exactly why the normalisation is needed).
- per-cell temporal std: median **347** (median std/mean = 18.5%).
- hardest cells (highest mean residual, NESTED ids): 6928, 6586, 6871, 7601, 5558, 8064, 6524, 6919, 6046, 12254.
- easiest cells: 11313, 10289, 11310, 10290, 10305, 11314, 10369, 11393, 10285, 11309.

`label_map.npy` `[N, 12288]` int16 carries the assigned cluster per cell over the **whole dataset** — the dynamic counterpart to `assignments.pt`, which only covers the sampled files. Both arrays sit on the same NESTED `[T, 12288]` grid as `err_persist.npy`, so they gather onto each other with no remapping: intersect them to separate *anomalous because rapidly changing* (high persistence error) from *anomalous because unmodelled* (high residual, low persistence error).


## Assignment margin (per-cluster separation)

*How to read this: every token is ranked against all K subspaces, so the **runner-up is free**. `margin = (R₂−R₁)/R₁` asks how much worse the second-best subspace is; a token with `margin < 10%` is a **near-tie** — the partition assigns it, but barely. This is the K-subspaces analogue of the silhouette coefficient, which does not transfer here because it assumes Euclidean distance to a centroid and spherical clusters — the wrong geometry for subspaces.*

- **33.4%** of all 160,002,048 tokens are near-ties (margin < 10%); median per-token margin per cluster ranges **0.08 … 1.34**.
- This decomposes that single global ambiguity figure **by cluster**: it is not spread evenly, so the worst rows below are the ones where hard labels actually throw information away (and where the soft/MPPCA responsibilities matter most).
- `maxAff` in the run report measures the *geometric* angle between two subspaces without touching data; `near_frac` measures whether they actually **contest the same tokens**. They are independent — two well-separated subspaces can still split a continuum.

Least-separated clusters (highest near-tie share of their own tokens):

| rank | cluster | tokens | share | margin_mean | near_frac | runner-up | runner-up share |
|---|---|---|---|---|---|---|---|
| 1 | 106 | 1,120,285 | 0.70% | 0.083 | 70.1% | 45 | 19.2% |
| 2 | 45 | 1,048,592 | 0.66% | 0.093 | 66.4% | 106 | 25.6% |
| 3 | 99 | 1,178,608 | 0.74% | 0.088 | 66.1% | 65 | 21.7% |
| 4 | 101 | 1,024,117 | 0.64% | 0.087 | 65.3% | 40 | 18.6% |
| 5 | 73 | 1,289,064 | 0.81% | 0.087 | 65.0% | 54 | 28.9% |
| 6 | 26 | 1,254,224 | 0.78% | 0.092 | 64.1% | 40 | 15.7% |
| 7 | 54 | 1,123,680 | 0.70% | 0.091 | 63.8% | 73 | 30.2% |
| 8 | 48 | 1,296,960 | 0.81% | 0.089 | 63.2% | 31 | 23.3% |
| 9 | 12 | 1,244,618 | 0.78% | 0.095 | 62.9% | 65 | 20.9% |
| 10 | 33 | 1,479,280 | 0.92% | 0.096 | 62.6% | 5 | 17.6% |
| 11 | 84 | 1,127,565 | 0.70% | 0.096 | 62.1% | 126 | 17.3% |
| 12 | 5 | 1,309,410 | 0.82% | 0.099 | 61.1% | 33 | 20.4% |
| 13 | 58 | 1,577,928 | 0.99% | 0.101 | 60.3% | 86 | 29.8% |
| 14 | 57 | 926,189 | 0.58% | 0.105 | 60.1% | 9 | 18.2% |
| 15 | 40 | 918,171 | 0.57% | 0.109 | 59.6% | 54 | 25.2% |

Best-separated clusters (lowest near-tie share):

| rank | cluster | tokens | share | margin_mean | near_frac | runner-up | runner-up share |
|---|---|---|---|---|---|---|---|
| 1 | 21 | 1,573,383 | 0.98% | 0.633 | 0.9% | 90 | 23.9% |
| 2 | 66 | 2,716,397 | 1.70% | 0.835 | 0.9% | 77 | 58.2% |
| 3 | 123 | 1,245,206 | 0.78% | 0.521 | 1.3% | 17 | 57.9% |
| 4 | 27 | 2,749,899 | 1.72% | 0.519 | 1.3% | 14 | 42.2% |
| 5 | 77 | 762,400 | 0.48% | 1.337 | 1.8% | 0 | 48.4% |

Per-file: `mean_margin` **0.247** (p1 0.231 / p99 0.265), `near_frac` **33.4%** (min 28.3% / max 40.2%) — a timestep whose cells are unusually contested is one the partition describes poorly, an independent hardness axis alongside `residual_frac` (corr = +0.79).

Full table: `cluster_margin.csv` (all 128 clusters). Merge it into the run report with `analyze_clusters.py --margins runs/signatures/v6_d64_margin/cluster_margin.csv`.


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