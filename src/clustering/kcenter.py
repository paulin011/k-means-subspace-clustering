#!/usr/bin/env python3
"""Greedy k-center (Gonzalez farthest-point) baseline on weather-encoder tokens.

K-center optimizes the *minimax* objective -- minimize the largest distance from any
token to its cluster center -- instead of k-means' sum of squared distances. The classic
greedy 2-approximation: start from a random token, then repeatedly add the token
farthest from all chosen centers. Centers are data points; there is no iterative
refinement, so a "run" is one selection + one assignment pass.

Everything around the algorithm is shared infrastructure from common/cluster_io.py:
the token sample (use --files-from <run>/sample.json for a directly comparable sample),
and the model.pt / assignments.pt schema (U has d=0: point clusters, no subspace).
The optional `radius [K]` field (max member distance per cluster, Euclidean not
squared) is k-center's native objective; `final_obj_per_token` is the mean squared
distance to the assigned center, comparable to a d=0 k-means objective on the same
sample fingerprint.

`means` holds the chosen center TOKENS, not cluster centroids -- k-center never runs an
M-step. Downstream readers must not treat the pair (`means`, `trace`) as a
law-of-total-variance decomposition: for any anchor c, E||x-c||^2 = E||x-mu||^2 +
||c-mu||^2, so `trace` overstates the true within-cluster variance by the squared
centroid offset and `between + within` is not the sample variance. `analyze_clusters.py`
detects this (`config['method'] == 'kcenter'`) and prints a dispersion section instead.

Numerics: the per-cluster SSE accumulator is float64. In float32 it silently stagnates
-- 86M tokens contributing ~5000 each drive one bin past 2.7e11, where the ulp exceeds
the addend and further adds round away -- which under-reported `trace` by ~3x in the
original v16 run while the double-precision `obj` stayed correct. assign_pass()
reconciles the two and raises if they disagree, so the failure cannot recur silently.
TF32 is disabled for the assignment matmul: it costs nothing here (the pass is bound by
the ~352 GB host-to-device stream, not by 45 TFLOP of matmul) and removes a ~3e-5 bias
from the headline objective. Center *selection* runs in fp16 on the subsample, where a
~3e-4 relative error only perturbs which extreme point is picked.

Scale note: the full 7000-file sample (~86M tokens, 352 GB fp16) exceeds GPU memory,
so center *selection* runs on a GPU-resident uniform subsample (--select-tokens,
default 8M ~ 30 GiB on one A40); the final assignment, counts, radius and objective
are computed over the FULL sample in one streamed pass. Selection on a subsample
only weakens the greedy's choice of extreme points; every reported number is
full-sample.

Multiple init seeds amortize the ~352 GB data load: --seeds 0 1 2 writes
<out>_seed<s>/ for each (selection is seconds per seed), all sharing one sample.json.
Single GPU (cuda:0); the broken P2P path never enters.

Smoke tests (no test framework in this repo): src/clustering/smoke_kcenter.py.

Run:  python3 src/clustering/kcenter.py --files-from <run>/sample.json -K 128 \
          --seeds 0 1 2 --max-ram-gb 420 --out runs/clustering/vNN_kcenter
"""

import argparse
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.cluster_io import (DIM, N_CELLS, load_tokens, sample_fingerprint,
                               save_assignments, save_model)

# Reconciliation tolerance for sum_j counts_j*trace[j] == T*obj_per_token. The two sides
# are the same `vals`, summed in a different order, and `trace` is additionally rounded
# to float32 for the schema -- so only rounding separates them at ~1e-7. The float32
# accumulator bug this guards against was off by a factor of 3.
RECON_RTOL = 1e-6


def greedy_centers(Xs, xn, K, init_seed):
    """Gonzalez farthest-point selection on a GPU-resident fp16 token block.

    Xs [n, D] half, xn [n] float32 = its squared norms. Returns (centers [K, D] float32,
    radius) where radius is the final minimax distance ON THE SUBSAMPLE. The first center
    is drawn with `init_seed`; everything after it is deterministic, so a seed sweep
    varies exactly one token.
    """
    n, D = Xs.shape
    gs = torch.Generator().manual_seed(init_seed)
    centers = torch.empty(K, D, device=Xs.device)
    centers[0] = Xs[int(torch.randint(n, (1,), generator=gs))].float()

    def dist_to(c):                                   # [n] squared distance to one center
        return (xn - 2.0 * (Xs @ c.half()).float() + (c * c).sum()).clamp_min_(0)

    mind = dist_to(centers[0])
    for k in range(1, K):
        centers[k] = Xs[int(mind.argmax())].float()
        mind = torch.minimum(mind, dist_to(centers[k]))
    return centers, float(mind.max().sqrt())


def assign_pass(data, centers, chunk_size, device):
    """One streamed full-sample assignment pass against fixed centers.

    Returns (labels [T] int32, counts [K] int64, radius [K] float32 Euclidean,
    trace [K] float32 = mean squared distance to the assigned center, obj_per_token).
    Accumulators are float64 (see the module docstring); the two independent routes to
    the objective are reconciled before returning.
    """
    K = centers.shape[0]
    T = data.shape[0]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False     # exact enough to quote 4 sig figs
    try:
        C = centers.to(device, torch.float32)
        cn = (C * C).sum(1)                           # [K]
        labels = torch.empty(T, dtype=torch.int32)
        rad2 = torch.zeros(K, device=device)
        ssum = torch.zeros(K, device=device, dtype=torch.float64)
        obj = 0.0
        for i0 in range(0, T, chunk_size):
            X = data[i0:i0 + chunk_size].to(device, non_blocking=True).float()
            d2 = (X * X).sum(1, keepdim=True) - 2.0 * (X @ C.T) + cn
            vals, a = d2.min(1)
            vals.clamp_min_(0)
            obj += vals.double().sum().item()
            labels[i0:i0 + chunk_size] = a.to("cpu", torch.int32)
            rad2.scatter_reduce_(0, a, vals, reduce="amax")
            ssum.index_add_(0, a, vals.double())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    counts = torch.bincount(labels.long(), minlength=K)
    radius = rad2.sqrt().cpu()
    trace = (ssum.cpu() / counts.clamp(min=1)).float()
    obj_per_token = obj / T

    # The two routes to the objective must agree: sum_j counts_j*trace[j] is the same set
    # of `vals` regrouped by cluster. A float32 `ssum` fails this by ~3x at 86M tokens.
    recon = float((counts.double() * trace.double()).sum() / T)
    assert abs(recon - obj_per_token) <= RECON_RTOL * max(obj_per_token, 1.0), (
        f"per-cluster SSE does not reconcile with the objective: {recon} vs "
        f"{obj_per_token} (rel {abs(recon - obj_per_token) / max(obj_per_token, 1.0):.3e}) "
        f"-- accumulator precision loss")
    return labels, counts, radius, trace, obj_per_token


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="latents_2", help="directory with latent_{i}.pt files")
    p.add_argument("--out", default="runs/clustering/kcenter_run",
                   help="output directory (suffix _seed<s> is appended when --seeds has >1 entry)")
    p.add_argument("--num-files", type=int, default=1500, help="random latent files to sample")
    p.add_argument("--files-from", default=None,
                   help="reuse the exact file list from a previous run's sample.json or model.pt")
    p.add_argument("--tokens-per-file", type=int, default=N_CELLS, help="random cells kept per file")
    p.add_argument("--clusters", "-K", type=int, default=64, help="number of centers")
    p.add_argument("--seed", type=int, default=0, help="token-sampling seed (fingerprint); NOT the init seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="greedy init seeds (first center choice); one output dir per seed")
    p.add_argument("--select-tokens", type=int, default=8_000_000,
                   help="GPU-resident subsample used for greedy center selection")
    p.add_argument("--chunk-size", type=int, default=262144, help="tokens per GPU chunk")
    p.add_argument("--load-workers", type=int, default=16, help="parallel file-loading threads")
    p.add_argument("--max-ram-gb", type=float, default=350.0, help="abort if sampled tokens exceed this")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    K, cs, dev = args.clusters, args.chunk_size, "cuda:0"
    outs = {s: args.out + (f"_seed{s}" if len(args.seeds) > 1 else "") for s in args.seeds}

    args.out = outs[args.seeds[0]]                    # load_tokens writes sample.json here
    data, file_ids, cell_ids, sampled = load_tokens(args)
    T = data.shape[0]
    fp = sample_fingerprint(sampled, min(args.tokens_per_file, N_CELLS), args.seed)

    # GPU-resident selection subsample (fp16) + its fp32 squared norms.
    n_sel = min(args.select_tokens, T)
    g = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(T, generator=g)[:n_sel] if n_sel < T else torch.arange(T)
    print(f"Selection subsample: {n_sel:,}/{T:,} tokens "
          f"({n_sel * DIM * 2 / 2**30:.1f} GiB on {dev})", flush=True)
    Xs = data[sel].to(dev)
    xn = torch.empty(n_sel, device=dev)
    for i0 in range(0, n_sel, cs):
        xb = Xs[i0:i0 + cs].float()
        xn[i0:i0 + cs] = (xb * xb).sum(1)

    for s in args.seeds:
        t0 = time.time()
        centers, sub_radius = greedy_centers(Xs, xn, K, s)
        print(f"seed {s}: selected {K} centers in {time.time() - t0:.1f}s "
              f"(subsample radius {sub_radius:.1f})", flush=True)

        labels, counts, radius, trace, obj_per_token = assign_pass(data, centers, cs, dev)
        print(f"seed {s}: obj/token={obj_per_token:.4f}  minimax radius={radius.max():.1f}  "
              f"sizes[min/med/max]={int(counts.min())}/{int(counts.median())}/{int(counts.max())}  "
              f"({time.time() - t0:.1f}s total)", flush=True)

        out = outs[s]
        config = {**vars(args), "out": out, "method": "kcenter", "init_seed": s}
        # Standard convergence-stat keys (analyze_clusters.py renders them), plus
        # k-center's native minimax numbers. One entry: there is no refinement loop,
        # so frac_changed is trivially 1.0 (every label assigned in this single sweep).
        history = [{"iter": 1, "obj_per_token": obj_per_token, "frac_changed": 1.0,
                    "size_min": int(counts.min()), "size_max": int(counts.max()),
                    "radius_max": float(radius.max()), "radius_median": float(radius.median())}]
        save_model(out, U=torch.zeros(K, DIM, 0), means=centers.cpu(),
                   eigvals=torch.zeros(K, 0), trace=trace, counts=counts,
                   config=config, history=history, sampled_files=sampled,
                   sample_fingerprint=fp, radius=radius, final_obj_per_token=obj_per_token)
        save_assignments(out, file_id=file_ids, cell_id=cell_ids, label=labels)
        if out != outs[args.seeds[0]]:                # every run dir carries the manifest
            shutil.copy(os.path.join(outs[args.seeds[0]], "sample.json"),
                        os.path.join(out, "sample.json"))
        print(f"Saved model.pt + assignments.pt to {out}/", flush=True)


if __name__ == "__main__":
    main()
