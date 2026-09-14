#!/usr/bin/env python3
"""Normalized mutual information between two runs' partitions.

Both runs must share the token sample (same sample fingerprint, e.g. via
--files-from): assignments.pt rows then align 1:1, which is asserted on
file_id/cell_id before comparing. NMI uses the arithmetic-mean normalisation
(same convention as analyze_forecast_error.py's static-vs-dynamic gate and the
historical v6/v7/v8 seed-sweep numbers: seed-to-seed ~0.68-0.72, chance ~0.13).
Label-permutation invariant, so no cluster matching is needed. Reads only the
two assignments.pt files -- no data pass, runs in seconds.

Run:  python3 src/analysis/partition_nmi.py --a <run_a> --b <run_b>
"""

import argparse
import os

import torch


def nmi(a, b, na, nb):
    """Normalized mutual information (arithmetic-mean norm) of integer labels."""
    joint = torch.bincount(a * nb + b, minlength=na * nb).double().view(na, nb)
    p = joint / joint.sum()
    pa, pb = p.sum(1), p.sum(0)
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    outer = pa[:, None] * pb[None, :]
    m = p > 0
    mi = (p[m] * (p[m] / outer[m]).log()).sum()
    return float(mi / ((ha + hb) / 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="first run directory")
    p.add_argument("--b", required=True, help="second run directory")
    args = p.parse_args()

    aa = torch.load(os.path.join(args.a, "assignments.pt"), map_location="cpu", weights_only=False)
    ab = torch.load(os.path.join(args.b, "assignments.pt"), map_location="cpu", weights_only=False)
    assert torch.equal(aa["file_id"], ab["file_id"]) and torch.equal(aa["cell_id"], ab["cell_id"]), \
        "runs do not share the token sample (row order differs); NMI would be meaningless"
    la, lb = aa["label"].long(), ab["label"].long()
    print(f"NMI({os.path.basename(args.a.rstrip('/'))}, {os.path.basename(args.b.rstrip('/'))}) "
          f"= {nmi(la, lb, int(la.max()) + 1, int(lb.max()) + 1):.4f}  "
          f"({la.numel():,} tokens)")


if __name__ == "__main__":
    main()
