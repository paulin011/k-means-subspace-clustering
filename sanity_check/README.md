# Sanity check: does clustering recover a known ground truth?

The smoke tests (`smoke_kmeans/`, `smoke_kcenter/`) only check that `subspace_kmeans.py`
and `kcenter.py` run to completion on real data. Real latents have no known "true"
cluster structure, so a smoke test can't tell you whether the recovered clusters are
*correct* — only that the code didn't crash.

This folder generates synthetic tokens from `--k-true` known Gaussian blobs in the
2048-dim token space, clusters them with the real algorithms, and checks the recovered
labels against the planted ground truth.

## Usage

```bash
# 1. generate synthetic latent_{i}.pt files + ground_truth.pt (default: 4 files, K_true=6)
python3 sanity_check/generate_synthetic_data.py --out sanity_check/data

# 2. cluster them with the real algorithms (same sample -> directly comparable)
python3 subspace_kmeans.py --files-from sanity_check/data/sample.json --dim 0 \
    --clusters 6 --iters 20 --src sanity_check/data --out sanity_check/kmeans_out
python3 kcenter.py --files-from sanity_check/data/sample.json \
    --clusters 6 --src sanity_check/data --out sanity_check/kcenter_out

# 3. check recovery against the planted ground truth
python3 sanity_check/check_recovery.py --dir sanity_check/kmeans_out
python3 sanity_check/check_recovery.py --dir sanity_check/kcenter_out
```

`check_recovery.py` reports Adjusted Rand Index / Normalized Mutual Info / purity
between recovered and true labels (permutation-invariant — recovered cluster 3 need
not be true cluster 3), plus the contingency table, so a merge or split is visible
directly.

## Finding (2026-07-08): `kcenter.py` recovers cleanly, `--dim 0` k-means does not

On the default synthetic data (centers separated ~320 apart vs. a per-token noise
scale of ~45 — trivially separable):

- **`kcenter.py`**: ARI = 1.0 — perfect recovery, every run.
- **`subspace_kmeans.py --dim 0`**: ARI 0.58–0.83 across seeds 0–3, every single seed
  tested. The contingency table shows two true blobs merged into one recovered cluster
  and a third blob split across two or three recovered clusters.

This is **not a code bug** — it's the classic failure mode of Lloyd's algorithm with
uniform-random-token initialization: if two of the K initial seeds happen to land in
the same true blob (leaving another blob unseeded), the resulting partition is a
locally-stable fixed point that Lloyd iteration alone won't escape. The existing
tiny-cluster re-seed guard (`subspace_kmeans.py`'s `update_model`) only fires below
`max(2*d, 64)` tokens, so it doesn't rescue a "wrong but not tiny" cluster like the
merged one here (16k+ tokens). `kcenter.py`'s farthest-point-greedy seeding avoids this
by construction — it explicitly picks maximally-separated seeds, so it can't leave a
well-separated blob unclaimed. Real analysis runs use `subspace_kmeans.py` with `d > 0`
(the `subspace_kmeans_runs/` results), where the richer per-cluster subspaces plus the
split-largest re-seed guard reduce this risk and the v6/v7/v8 seed-robustness sweep
(README.md) already checked reproducibility there — this specifically flags the plain
`--dim 0` / point-cluster path, which hasn't had that same scrutiny.

If plain k-means recovery quality matters for a future run, consider: multiple random
seeds + keep the lowest `final_obj_per_token`, or borrowing `kcenter.py`'s farthest-point
seeding to initialize `subspace_kmeans.py`'s `--dim 0` case instead of uniform-random
tokens.
