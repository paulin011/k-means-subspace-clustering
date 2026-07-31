# Per-file regime signature + residual — `runs/signatures/v6_d64_new`

*Generated 2026-07-31 17:41 by `file_signature.py`. Frozen model `runs/clustering/v6_subspace_big_d64` (K=128, d=64, affine=True, fingerprint `82ca602ed7e7`). 13,021 files × 12288 cells = 160,002,048 tokens assigned on a single GPU.*

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
R = np.load('runs/signatures/v6_d64_new/residual_map.npy', mmap_mode='r')   # [N, 12288]
s = np.load('runs/signatures/v6_d64_new/signatures.npz')
z = (R - s['cell_mean_residual']) / np.maximum(s['cell_std_residual'], 1e-6)
# z[t, cell] > 4  ->  this cell is far outside its own normal range at step t
```

- per-cell mean residual across the 13,021 steps: min **273** / median **1923** / max **3386** (12.4× spread — this is exactly why the normalisation is needed).
- per-cell temporal std: median **347** (median std/mean = 18.5%).
- hardest cells (highest mean residual, NESTED ids): 6928, 6586, 6871, 7601, 5558, 8064, 6524, 6919, 6046, 12254.
- easiest cells: 11313, 10289, 11310, 10290, 10305, 11314, 10369, 11393, 10285, 11309.

`label_map.npy` `[N, 12288]` int16 carries the assigned cluster per cell over the **whole dataset** — the dynamic counterpart to `assignments.pt`, which only covers the sampled files. Both arrays sit on the same NESTED `[T, 12288]` grid as `err_persist.npy`, so they gather onto each other with no remapping: intersect them to separate *anomalous because rapidly changing* (high persistence error) from *anomalous because unmodelled* (high residual, low persistence error).


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
- `residual_map.npy` — `[13021, 12288]` float32 per-cell residual (0.60 GiB; load with `mmap_mode='r'`).
- `label_map.npy` — `[13021, 12288]` int16 per-cell assigned cluster.