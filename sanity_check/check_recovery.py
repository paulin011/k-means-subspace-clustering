#!/usr/bin/env python3
"""Check whether a kmeans/kcenter run recovered the planted ground-truth clusters.

Pairs a run's `assignments.pt` (recovered `label` per token) with
`generate_synthetic_data.py`'s `ground_truth.pt` (`true_label` per token), joining on
(file_id, cell_id), then reports label-agreement metrics that are invariant to the
arbitrary cluster numbering (recovered cluster 3 need not correspond to true cluster 3):

  - Adjusted Rand Index (ARI): 1.0 = perfect agreement, ~0.0 = random labeling.
  - Normalized Mutual Information (NMI): 1.0 = perfect agreement, 0.0 = independent.
  - Purity: mean over recovered clusters of their majority true label's share --
    high purity with low NMI would mean recovered clusters are pure but fragmented
    (K_pred > K_true), which the printed contingency table makes visible.

Usage:
  python3 sanity_check/check_recovery.py --dir sanity_check/kmeans_out
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from cluster_io import N_CELLS


def contingency(true, pred):
    k_true, k_pred = int(true.max()) + 1, int(pred.max()) + 1
    C = np.zeros((k_true, k_pred), dtype=np.int64)
    np.add.at(C, (true, pred), 1)
    return C


def adjusted_rand_index(C):
    n = C.sum()
    comb2 = lambda x: x * (x - 1) // 2
    sum_c = sum(comb2(int(x)) for x in C.sum(1))
    sum_k = sum(comb2(int(x)) for x in C.sum(0))
    sum_both = sum(comb2(int(x)) for x in C.flatten())
    expected = sum_c * sum_k / comb2(int(n)) if n > 1 else 0.0
    max_index = 0.5 * (sum_c + sum_k)
    denom = max_index - expected
    return 1.0 if denom == 0 else (sum_both - expected) / denom


def normalized_mutual_info(C):
    n = C.sum()
    pxy = C / n
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / (px * py), 1.0)
        mi = np.nansum(pxy * np.log(ratio))
    hx = -np.nansum(np.where(px > 0, px * np.log(px), 0.0))
    hy = -np.nansum(np.where(py > 0, py * np.log(py), 0.0))
    return 1.0 if hx + hy == 0 else 2 * mi / (hx + hy)


def purity(C):
    return C.max(axis=0).sum() / C.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run directory with assignments.pt")
    ap.add_argument("--ground-truth", default="sanity_check/data/ground_truth.pt")
    ap.add_argument("--ari-threshold", type=float, default=0.9,
                    help="ARI below this is flagged as a mismatch")
    args = ap.parse_args()

    a = torch.load(os.path.join(args.dir, "assignments.pt"), map_location="cpu", weights_only=False)
    gt = torch.load(args.ground_truth, map_location="cpu", weights_only=False)

    gid = (a["file_id"].long() * N_CELLS + a["cell_id"].long()).numpy()
    gt_gid = (gt["file_id"].long() * N_CELLS + gt["cell_id"].long()).numpy()
    lookup = np.full(int(gt_gid.max()) + 1, -1, dtype=np.int64)
    lookup[gt_gid] = gt["true_label"].numpy()
    true_label = lookup[gid]
    missing = int((true_label < 0).sum())
    if missing:
        raise SystemExit(f"{missing}/{len(gid)} assigned tokens have no matching ground-truth "
                          f"cell -- did --ground-truth come from the run that produced {args.dir}?")

    pred_label = a["label"].long().numpy()
    C = contingency(true_label, pred_label)
    ari = adjusted_rand_index(C)
    nmi = normalized_mutual_info(C)
    pur = purity(C)

    print(f"Run: {args.dir}  ({len(gid):,} tokens, K_true={C.shape[0]}, K_pred={C.shape[1]})")
    print(f"  Adjusted Rand Index : {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}")
    print(f"  Purity              : {pur:.4f}")
    print("\n  contingency table (rows=true cluster, cols=recovered cluster):")
    header = "        " + "".join(f"{j:>8}" for j in range(C.shape[1]))
    print(header)
    for i, row in enumerate(C):
        print(f"  true{i:>2} " + "".join(f"{x:>8}" for x in row))

    verdict = "RECOVERED" if ari >= args.ari_threshold else "MISMATCH"
    print(f"\nVerdict: {verdict} (ARI {'>=' if ari >= args.ari_threshold else '<'} "
          f"{args.ari_threshold} threshold)")
    if verdict == "MISMATCH":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
