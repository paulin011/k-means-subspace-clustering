#!/usr/bin/env python3
"""Held-out generalization check for a clustering run.

The variance decomposition in `analyze_clusters.py` (between / within / residual) is
measured on the *same* tokens the model was fit on. With d-dim subspaces each cluster
fits K*(2048*d) basis parameters (e.g. K=128, d=128 -> ~33.5 M numbers), so an
in-sample residual of "31.5%" could in principle be optimistic -- the bases might be
partly memorising the training tokens rather than capturing reusable structure.

This script freezes the trained model (means mu_j, bases U_j) and replays the *exact*
assignment rule the algorithm optimises --

    residual_j(x) = ||x - mu_j||^2 - ||U_j^T (x - mu_j)||^2     (affine, the default)
                  = ||x||^2 - ||U_j^T x||^2                     (linear, --linear: subspace
                                                                 through the origin, mu_j = 0)
                  = ||x - mu_j||^2                              (point cluster, d=0)

-- on tokens drawn from latent files the run never saw. It then reports the held-out
residual as a fraction of held-out total variance, alongside the in-sample numbers, so
you can read off whether the structure generalises:

  * held-out residual % ~= in-sample residual %   -> the subspaces generalise.
  * held-out residual % >> in-sample residual %    -> overfitting; the bases are too
                                                      expensive for the signal (lower d
                                                      or raise the token count).

Definitions (all expectations are per-token means; mu_global is the trained global
mean Sum_j w_j mu_j, frozen, so the denominator matches the in-sample report):
  total    = E||x - mu_global||^2
  residual = E[ residual_a(x) ]   (a = argmin assignment)   == the objective/token
  modeled  = total - residual

We deliberately do NOT re-split held-out variance into between/within: with frozen
means that ANOVA identity no longer holds exactly, and `residual / total` is the one
number that is both well-defined and directly comparable to the in-sample report.

Writes <dir>/holdout.json; `analyze_clusters.py` renders a "Held-out generalization"
section from it automatically on the next run.

Usage:
  python3 holdout_eval.py --dir subspace_kmeans_runs/v6_subspace_big_d64 --num-files 200
"""

import argparse
import json
import os
import types

import torch

from cluster_io import (DIM, N_CELLS, N_FILES_TOTAL, OVERFIT_GAP_THRESHOLD,
                        load_tokens)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="run directory with model.pt")
    p.add_argument("--src", default=None, help="latent dir (default: the run's config src)")
    p.add_argument("--num-files", type=int, default=200,
                   help="held-out latent files to evaluate (none overlap the training sample)")
    p.add_argument("--seed", type=int, default=12345,
                   help="seed for choosing held-out files (kept distinct from training seeds)")
    p.add_argument("--chunk-size", type=int, default=262144, help="tokens per GPU chunk")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--load-workers", type=int, default=16)
    p.add_argument("--max-ram-gb", type=float, default=350.0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{args.gpu}"

    m = torch.load(os.path.join(args.dir, "model.pt"), map_location="cpu", weights_only=False)
    U, means, cnt = m["U"], m["means"], m["counts"].float()
    cfg = m["config"]
    K, D, d = U.shape
    affine = not cfg.get("linear", False)
    src = args.src or cfg.get("src", "latents_2")
    trained = set(int(f) for f in m.get("sampled_files", []))
    print(f"Model: K={K}, d={d}, affine={affine}, trained on {len(trained)} files", flush=True)

    # ---- choose held-out files (disjoint from training) and load them --------
    pool = [f for f in range(N_FILES_TOTAL) if f not in trained]
    if not pool:
        raise SystemExit("Every file was used in training; no held-out set possible.")
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(pool), generator=g)[:min(args.num_files, len(pool))]
    holdout = sorted(pool[i] for i in perm.tolist())
    print(f"Evaluating on {len(holdout)} held-out files (of {len(pool)} unused).", flush=True)

    # Reuse the production loader by handing it an explicit file list. We point its
    # sample.json at a holdout/ subdir so the run's own manifest is untouched, and
    # take all 12288 tokens per file (no subsampling -> seed is irrelevant here).
    hp = os.path.join(args.dir, "holdout")
    os.makedirs(hp, exist_ok=True)
    flist = os.path.join(hp, "files.json")
    with open(flist, "w") as f:
        json.dump({"files": holdout}, f)
    load_args = types.SimpleNamespace(
        src=src, out=hp, num_files=len(holdout), files_from=flist,
        tokens_per_file=N_CELLS, seed=args.seed,
        load_workers=args.load_workers, max_ram_gb=args.max_ram_gb)
    data, _, _, _ = load_tokens(load_args)
    M = data.shape[0]

    # ---- global (trained) mean: denominator matches the in-sample report ------
    w = cnt / cnt.sum()
    mu_g = (w[:, None] * means).sum(0).to(device)            # [D]

    # ---- frozen-model assignment + residual accumulation (single GPU) ---------
    U = U.to(device)
    U_cat = U.permute(1, 0, 2).reshape(D, K * d).contiguous()  # [D, K*d]
    if affine:
        mu = means.to(device)                                # [K, D]
        c = torch.bmm(U.transpose(1, 2), mu.unsqueeze(-1)).squeeze(-1)  # [K, d]
        mnorm = (mu * mu).sum(1)
        cnorm = (c * c).sum(1)

    resid_sum = 0.0      # Sum residual_a(x)            -> objective/token
    within_sum = 0.0     # Sum ||x - mu_a||^2           -> distance to assigned centroid
    total_sum = 0.0      # Sum ||x - mu_global||^2      -> denominator
    cap_sum = 0.0        # Sum ||U_a^T (x - mu_a)||^2   -> captured by the subspace
    for i0 in range(0, M, args.chunk_size):
        X = data[i0:i0 + args.chunk_size].to(device, non_blocking=True).float()
        xnorm = (X * X).sum(1, keepdim=True)                 # [B,1]
        P = (X @ U_cat).view(X.shape[0], K, d)
        pe = (P * P).sum(-1)                                 # [B,K]
        if affine:
            xm = X @ mu.T                                    # [B,K]
            pc = torch.einsum("bkd,kd->bk", P, c)
            dist2 = xnorm - 2 * xm + mnorm                   # ||x-mu_j||^2
            proj = pe - 2 * pc + cnorm                       # ||U_j^T(x-mu_j)||^2
        else:
            # Linear subspaces through the origin (mu_j = 0): dist2 = ||x||^2 and the
            # captured energy is the projection onto U_j. (This previously set
            # proj = 0, which dropped U entirely -- wrong for --linear d>0: every
            # cluster tied at ||x||^2 and the argmin collapsed to cluster 0.)
            dist2 = xnorm - 2 * (X @ means.to(device).T) + (means.to(device) ** 2).sum(1)
            proj = pe
        R = dist2 - proj                                     # residual to each subspace
        # argmin on the raw R; only the *sums* below are clamped at 0 (the
        # orthogonal residual is >= 0 mathematically; TF32 rounding can make it
        # tiny-negative). Clamp after min(1), not before.
        vals, a = R.min(1)                                   # assign by min residual
        idx = torch.arange(a.shape[0], device=device)
        resid_sum += vals.clamp_min(0).double().sum().item()
        within_sum += dist2[idx, a].clamp_min(0).double().sum().item()
        cap_sum += proj[idx, a].clamp_min(0).double().sum().item()
        total_sum += ((X - mu_g) ** 2).sum(1).double().sum().item()

    total = total_sum / M
    resid = resid_sum / M
    within = within_sum / M
    captured = cap_sum / M

    # ---- in-sample reference (recomputed from the model, same as the report) --
    in_between = float((w * ((means - mu_g.cpu()) ** 2).sum(1)).sum())
    in_within = float((w * m["trace"]).sum())
    in_total = in_between + in_within
    in_captured = float((w * m["eigvals"].sum(1)).sum()) if d > 0 else 0.0
    in_resid = in_within - in_captured
    # True objective of the saved model (its final relabel sweep) when present;
    # fall back to the last training iteration's objective for older runs.
    in_obj = (m["final_obj_per_token"] if m.get("final_obj_per_token") is not None
              else (m["history"][-1]["obj_per_token"] if m["history"] else float("nan")))

    result = {
        "num_holdout_files": len(holdout),
        "num_holdout_tokens": M,
        "holdout": {"total": total, "residual": resid, "within": within,
                    "captured": captured,
                    "residual_frac": resid / total, "modeled_frac": 1 - resid / total},
        "in_sample": {"total": in_total, "residual": in_resid,
                      "residual_frac": in_resid / in_total,
                      "final_obj_per_token": in_obj},
    }
    with open(os.path.join(args.dir, "holdout.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n================ Held-out generalization ================")
    print(f"  held-out tokens             : {M:,} from {len(holdout)} unseen files")
    print(f"  residual / total  held-out  : {resid / total:7.1%}")
    print(f"  residual / total  in-sample : {in_resid / in_total:7.1%}")
    print(f"  objective/token   held-out  : {resid:10.2f}")
    print(f"  objective/token   in-sample : {in_obj:10.2f}  (final training iter)")
    gap = (resid / total) - (in_resid / in_total)
    print(f"  generalization gap          : {gap:+.1%}  "
          f"({'overfitting' if gap > OVERFIT_GAP_THRESHOLD else 'generalises well'})")
    print(f"\nWrote {os.path.join(args.dir, 'holdout.json')}")


if __name__ == "__main__":
    main()
