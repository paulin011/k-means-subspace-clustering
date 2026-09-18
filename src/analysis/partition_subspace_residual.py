#!/usr/bin/env python3
"""Post-hoc subspace residual of a run's PARTITION, at fixed labels.

Answers: how good is this run's partition *as a subspace clustering*? Reload the
run's exact token sample, keep its labels frozen (no reassignment), fit each
cluster's best d-dimensional affine subspace by exact PCA, and report the mean
squared residual per token
    resid/token = sum_j n_j (tr C_j - sum_{i<=d} lambda_ji) / T .
This is the apples-to-apples number for comparing a d=0 partition (plain k-means /
k-center) against a K-subspaces run at the same K and d: both get the identical
model class per cluster; only the partition differs. No second data pass is needed
after the moment accumulation, because the residual is a function of the
per-cluster covariance alone.

Also reports: the d=0 residual (= count-weighted mean trace; for a d=0 run this
must reproduce model['final_obj_per_token'] -- printed as an invariant check) and
the total token variance about the global mean (the denominator for "residual as
% of total variance").

One streamed pass over the run's sample (~352 GB RAM for a 7000-file run -- do
not overlap with another full-sample job), single GPU. Writes
<dir>/posthoc_subspace.json.

Run:  python3 src/analysis/partition_subspace_residual.py --dir <run> --dim 64
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.cluster_io import DIM, load_tokens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="run directory (model.pt/assignments.pt/sample.json)")
    p.add_argument("--dim", type=int, default=64, help="subspace dimension of the post-hoc PCA fit")
    p.add_argument("--src", default="latents_2")
    p.add_argument("--chunk-size", type=int, default=262144)
    p.add_argument("--load-workers", type=int, default=16)
    p.add_argument("--max-ram-gb", type=float, default=420.0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev, cs, d = "cuda:0", args.chunk_size, args.dim

    model = torch.load(os.path.join(args.dir, "model.pt"), map_location="cpu", weights_only=False)
    assign = torch.load(os.path.join(args.dir, "assignments.pt"), map_location="cpu", weights_only=False)
    K = model["means"].shape[0]
    with open(os.path.join(args.dir, "sample.json")) as f:
        sample = json.load(f)

    # Reload the exact token sample; the manifest goes to a throwaway subdir so the
    # run's own sample.json is never touched (same pattern as holdout_eval.py).
    la = SimpleNamespace(src=args.src, out=os.path.join(args.dir, "posthoc"),
                         num_files=0, files_from=os.path.join(args.dir, "sample.json"),
                         tokens_per_file=sample["tokens_per_file"], seed=sample["seed"],
                         load_workers=args.load_workers, max_ram_gb=args.max_ram_gb)
    data, file_ids, cell_ids, _ = load_tokens(la)
    T = data.shape[0]
    # Token order must match assignments.pt row-for-row (load_tokens places file
    # `pos` at rows [pos*tpf, (pos+1)*tpf) regardless of thread completion order).
    assert torch.equal(file_ids, assign["file_id"]) and torch.equal(cell_ids, assign["cell_id"]), \
        "reloaded token order does not match assignments.pt"
    labels = assign["label"].long()

    t0 = time.time()
    S = torch.zeros(K, DIM, DIM, device=dev)
    msum = torch.zeros(K, DIM, device=dev)
    cnt = torch.zeros(K, device=dev)
    for i0 in range(0, T, cs):
        X = data[i0:i0 + cs].to(dev, non_blocking=True).float()
        a = labels[i0:i0 + cs].to(dev)
        for j in torch.unique(a).tolist():
            Xj = X[a == j]
            S[j].addmm_(Xj.T, Xj)
            msum[j] += Xj.sum(0)
            cnt[j] += Xj.shape[0]
        if (i0 // cs) % 40 == 0:
            print(f"  moments {i0:,}/{T:,} tokens ({time.time() - t0:.0f}s)", flush=True)

    n = cnt.clamp(min=1.0)
    mu = msum / n[:, None]
    C = S / n[:, None, None] - mu.unsqueeze(-1) * mu.unsqueeze(1)
    trace = C.diagonal(dim1=-2, dim2=-1).sum(-1)                     # [K]
    evals = torch.linalg.eigvalsh(C)                                 # ascending
    top = evals[:, DIM - d:].clamp_min(0).sum(1) if d > 0 else torch.zeros(K, device=dev)

    w = cnt / T
    resid_d = float((w * (trace - top)).sum())                       # post-hoc d-dim residual
    resid_0 = float((w * trace).sum())                               # = within variance (d=0 objective)
    gmu = msum.sum(0) / T
    total_var = float(S.diagonal(dim1=-2, dim2=-1).sum() / T - (gmu * gmu).sum())

    ref = model.get("final_obj_per_token")
    print(f"partition: {args.dir}  (method={model['config'].get('method')}, K={K})")
    print(f"d=0 residual/token   : {resid_0:.2f}"
          + (f"  (run's final_obj_per_token {ref:.2f}, rel diff {abs(resid_0 - ref) / ref:.1e})"
             if ref is not None and model["U"].shape[2] == 0 else ""))
    print(f"post-hoc d={d} residual/token: {resid_d:.2f}")
    print(f"total variance/token : {total_var:.2f}  "
          f"(residual fractions: d=0 {resid_0 / total_var:.1%}, d={d} {resid_d / total_var:.1%})")

    out = {"dir": args.dir, "dim": d, "resid_per_token_d": resid_d,
           "resid_per_token_d0": resid_0, "total_var_per_token": total_var,
           "final_obj_per_token_ref": ref, "K": K, "tokens": T}
    with open(os.path.join(args.dir, "posthoc_subspace.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.dir}/posthoc_subspace.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
