#!/usr/bin/env python3
"""K-subspaces clustering (k-means for affine/linear subspaces) on weather-encoder tokens.

Per iteration, one streaming sweep over the sampled tokens, split across both GPUs:
  1. Assignment: token x goes to cluster j with the smallest orthogonal residual
         r_j(x) = ||x - mu_j||^2 - ||U_j^T (x - mu_j)||^2.
  2. Update: per-cluster second moments are accumulated on-GPU during the same
     sweep; the new basis U_j is the top-d eigenvectors of the cluster covariance.
     This is exact PCA (batched torch.linalg.eigh on the 2048x2048 covariances) --
     no randomized SVD / power iterations needed.

Token selection: --num-files latent files are drawn uniformly at random, and
--tokens-per-file random HEALPix cells are kept from each (default: all 12288).
Each file spans the full globe, so subsampling files only thins the time axis.

Soft mode (--soft) swaps the hard argmin for EM on the matching MPPCA mixture, so each
token carries a distribution over its --soft-topm best clusters instead of one label.
Motivation: ~31% of tokens sit within a 10% residual margin of a second cluster, so the
hard label both discards real information and injects seed-dependent noise there. The
partition itself barely moves (the likelihood rule flips ~3.7% of assignments); the point
is the per-token uncertainty. See --soft-temp: at the literal likelihood (T=1) the DIM=2048
score is so peaked that responsibilities come out one-hot anyway.

Outputs in --out:
  model.pt        U [K, 2048, d], means [K, 2048], eigvals [K, d], trace [K],
                  counts [K], config, per-iteration history, sampled file list;
                  --soft adds sigma2 [K], mixing [K], soft_counts [K]
  assignments.pt  file_id / cell_id / label (int32) for every sampled token;
                  --soft adds resp_idx [T, m] int16 + resp_w [T, m] float16
  sample.json     reproducible manifest: fingerprint, seed, tokens_per_file, and
                  the sorted list of sampled file ids -- written before clustering

Reproducibility / direct comparison: to cluster a *different* configuration on the
exact same tokens (e.g. vary K or d but hold the data fixed), pass the previous run's
sample with --files-from <dir>/sample.json (or <dir>/model.pt) plus the same --seed and
--tokens-per-file. Runs sharing a sample fingerprint are directly comparable.

Full run:  nohup python3 src/clustering/subspace_kmeans.py > logs/subspace_run.log 2>&1 &
"""

import argparse
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.cluster_io import (DIM, N_CELLS, N_FILES_TOTAL, load_tokens,
                               sample_fingerprint, save_assignments, save_model)

# Lower bound on the MPPCA noise variance sigma2 (--soft only). Observed sigma2 on the
# v6 model is 0.29..1.54, so 1e-3 is ~2.5 orders below anything real: it never binds on a
# healthy cluster, and it stops the EM degeneracy where a component drives sigma2 -> 0
# (and its log-det term to -inf) by collapsing onto a handful of points.
SIGMA2_FLOOR = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="latents_2", help="directory with latent_{i}.pt files")
    p.add_argument("--out", default="runs/clustering/new_run", help="output directory")
    p.add_argument("--num-files", type=int, default=1500, help="random latent files to sample")
    p.add_argument("--files-from", default=None,
                   help="reuse the exact file list from a previous run's sample.json or "
                        "model.pt (for direct comparison; overrides --num-files). Combine "
                        "with the same --seed and --tokens-per-file to reproduce the token set.")
    p.add_argument("--tokens-per-file", type=int, default=N_CELLS, help="random cells kept per file")
    p.add_argument("--clusters", "-K", type=int, default=64, help="number of subspace clusters")
    p.add_argument("--dim", "-d", type=int, default=16, help="subspace dimension per cluster")
    p.add_argument("--iters", type=int, default=25, help="max assignment/update iterations")
    p.add_argument("--tol", type=float, default=1e-3, help="stop when fraction of changed labels < tol")
    p.add_argument("--linear", action="store_true", help="subspaces through the origin (no per-cluster mean)")
    p.add_argument("--soft", action="store_true",
                   help="EM with truncated responsibilities (MPPCA) instead of hard argmin: "
                        "each token is claimed fractionally by its --soft-topm best clusters. "
                        "Gives per-token uncertainty; the partition itself barely moves.")
    p.add_argument("--soft-topm", type=int, default=4,
                   help="responsibility slots kept per token under --soft (M-step cost is "
                        "~m x the hard cost; 4 covers virtually all the mass)")
    p.add_argument("--soft-temp", type=float, default=1.0,
                   help="temperature on the MPPCA score (--soft). 1.0 is the literal "
                        "likelihood, which in DIM=2048 is so peaked that responsibilities "
                        "collapse back to one-hot (measured: mean top-1 weight 0.99). The "
                        "likelihood treats all 2048 dims as independent evidence though the "
                        "data has ~121 effective dims, so T ~ 2048/121 ~ 17 de-overcounts it "
                        "and recovers graded responsibilities (deterministic-annealing EM).")
    p.add_argument("--chunk-size", type=int, default=262144, help="tokens per GPU chunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpus", type=int, default=min(2, torch.cuda.device_count()))
    p.add_argument("--load-workers", type=int, default=16, help="parallel file-loading threads")
    p.add_argument("--max-ram-gb", type=float, default=350.0, help="abort if sampled tokens exceed this")
    return p.parse_args()


def sweep_worker(rank, device, data, labels, chunks, model, K, d, affine,
                 accumulate, results, soft_m=0, resp_idx=None, resp_w=None,
                 soft_temp=1.0):
    """Process this GPU's chunks: assign tokens, optionally accumulate moments.

    Hard mode (soft_m == 0) is the validated K-subspaces path: argmin of the orthogonal
    residual, one cluster per token.

    Soft mode (soft_m > 0) fits the matching MPPCA mixture instead. Its covariance is
        C_j = U_j diag(lam_j) U_j^T + sigma2_j (I - U_j U_j^T),
    which makes the Mahalanobis distance and log-determinant separable into terms this
    kernel already computes (z_j = U_j^T (x - mu_j)):
        (x-mu)^T C^-1 (x-mu) = R_j / sigma2_j + sum_i z_ji^2 / lam_ji
        log det C            = sum_i log lam_ji + (DIM - d) log sigma2_j
    so the *only* extra cost per token is two [B,K,d] einsums; the assignment scan is
    unchanged. Responsibilities are truncated to the best `soft_m` clusters and
    renormalised -- with ~69% of tokens carrying a >10% margin, virtually all the mass
    lives in the first few slots, and truncation caps the M-step at ~m x the hard cost
    instead of K x.
    """
    torch.cuda.set_device(device)
    U = model["U"].to(device)                                       # [K, DIM, d]
    U_cat = U.permute(1, 0, 2).reshape(DIM, K * d).contiguous()     # [DIM, K*d]
    if affine:
        means = model["means"].to(device)                           # [K, DIM]
        c = torch.bmm(U.transpose(1, 2), means.unsqueeze(-1)).squeeze(-1)  # [K, d]
        mnorm = (means * means).sum(1)                              # [K]
        cnorm = (c * c).sum(1)
    soft = soft_m > 0
    if soft:
        # lam is floored at sigma2: PPCA requires the modelled directions to carry at
        # least the noise variance, and it keeps log(lam) finite for a just-reseeded
        # cluster whose trailing eigenvalues are still 0.
        sig2 = model["sigma2"].to(device).clamp_min(SIGMA2_FLOOR)   # [K]
        pi = model["mixing"].to(device).clamp_min(1e-12)            # [K]
        lam = model["eigvals"].to(device).clamp_min(sig2.unsqueeze(1))   # [K, d]
        inv_lam = 1.0 / lam
        # const absorbs everything that does not depend on x, incl. the mixing weights.
        const = lam.log().sum(1) + (DIM - d) * sig2.log() - 2.0 * pi.log()   # [K]
        if affine:
            c_lam = c * inv_lam                                     # [K, d]
            cnorm_lam = (c * c * inv_lam).sum(1)                     # [K]
    if accumulate:
        S = torch.zeros(K, DIM, DIM, device=device)
        msum = torch.zeros(K, DIM, device=device)
        cnt = torch.zeros(K, device=device)
    obj, obj_nll, changed = 0.0, 0.0, 0

    for i0, i1 in chunks:
        X = data[i0:i1].to(device, non_blocking=True).float()
        B = X.shape[0]
        xnorm = (X * X).sum(1, keepdim=True)                        # [B, 1]
        P = (X @ U_cat).view(B, K, d)
        if affine:
            xm = X @ means.T                                        # [B, K]
            pc = torch.einsum("bkd,kd->bk", P, c)
        if soft:
            # Consume P before squaring it in place -- P is [B, K, d] (8.6 GB at the
            # default chunk size), so materialising a second copy would risk OOM.
            if affine:
                pcl = torch.einsum("bkd,kd->bk", P, c_lam)
            P.pow_(2)                                               # P := P^2
            pe = P.sum(-1)                                          # [B, K]
            t1 = torch.einsum("bkd,kd->bk", P, inv_lam)             # sum_i z_i^2/lam (x-part)
        else:
            pe = (P * P).sum(-1)                                    # [B, K]
        if affine:
            R = xnorm - 2 * xm + mnorm - (pe - 2 * pc + cnorm)
        else:
            R = xnorm - pe

        if soft:
            maha_in = (t1 - 2 * pcl + cnorm_lam) if affine else t1
            score = R.clamp_min(0) / sig2 + maha_in + const         # [B, K] = -2 log p + C
            sv, si = score.topk(soft_m, dim=1, largest=False)       # ascending -> best first
            # Temperature divides the log-likelihood gap. At T=1 (the literal likelihood)
            # this softmax saturates in DIM=2048 -- the gap between two clusters is
            # hundreds of nats -- so responsibilities come out one-hot and the run is just
            # a slower hard k-subspaces. T>1 recovers graded weights; the *objective*
            # (obj_nll below) is always the untempered likelihood, so runs stay comparable.
            r = torch.softmax(-0.5 * (sv - sv[:, :1]) / soft_temp, dim=1)   # [B, m], sums to 1
            a = si[:, 0].contiguous()                               # argmax responsibility
            vals = R.gather(1, a.unsqueeze(1)).squeeze(1)           # hard residual, comparable
            # Truncated mixture NLL (drops the constant DIM/2 * log 2pi).
            obj_nll += (0.5 * sv[:, 0]
                        - torch.logsumexp(-0.5 * (sv - sv[:, :1]), dim=1)).double().sum().item()
            resp_idx[i0:i1] = si.to("cpu", torch.int16)
            resp_w[i0:i1] = r.to("cpu", torch.float16)
        else:
            vals, a = R.min(1)
        # Assignment (argmin) uses the raw residual R; only the objective *sum* is
        # clamped at 0 to absorb TF32-induced tiny negatives (the orthogonal
        # residual is >= 0 mathematically). Keep the clamp AFTER min(1) -- clamping
        # R first would distort the argmin between near-tied subspaces.
        obj += vals.clamp_min(0).double().sum().item()
        a_cpu = a.to("cpu", torch.int32)
        changed += (a_cpu != labels[i0:i1]).sum().item()
        labels[i0:i1] = a_cpu
        if accumulate and soft:
            # Weighted moments. Flatten the [B, m] slots to (token, cluster, weight)
            # triples and group by cluster, so each cluster does exactly one addmm over
            # its own rows -- total work sum_j n_j = B*m, i.e. m x the hard path.
            fj = si.reshape(-1)
            order = torch.argsort(fj)
            fj, fw = fj[order], r.reshape(-1)[order]
            tok = torch.div(order, soft_m, rounding_mode="floor")   # flat = token*m + slot
            uniq, ucnt = torch.unique_consecutive(fj, return_counts=True)
            bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                ucnt.cumsum(0)]).tolist()
            for jj, j in enumerate(uniq.tolist()):
                s, e = bounds[jj], bounds[jj + 1]
                Xj, wj = X[tok[s:e]], fw[s:e]
                Xw = Xj * wj.sqrt().unsqueeze(1)                    # sqrt on both sides
                S[j].addmm_(Xw.T, Xw)                               # -> sum_n w_n x x^T
                msum[j] += (Xj * wj.unsqueeze(1)).sum(0)
                cnt[j] += wj.sum()
        elif accumulate:
            for j in torch.unique(a).tolist():
                Xj = X[a == j]
                S[j].addmm_(Xj.T, Xj)
                msum[j] += Xj.sum(0)
                cnt[j] += Xj.shape[0]

    # Direct GPU<->GPU copies are silently broken on this machine (P2P returns
    # zeros/garbage), so accumulators must leave the GPU via D2H here, in the
    # thread that owns the stream. The merge then happens on CPU.
    results[rank] = {
        "S": S.cpu() if accumulate else None,
        "msum": msum.cpu() if accumulate else None,
        "cnt": cnt.cpu() if accumulate else None,
        "obj": obj, "obj_nll": obj_nll, "changed": changed,
    }


def run_sweep(devices, data, labels, model, K, d, affine, accumulate, chunk_size,
              soft_m=0, resp_idx=None, resp_w=None, soft_temp=1.0):
    T = data.shape[0]
    chunks = [(i0, min(i0 + chunk_size, T)) for i0 in range(0, T, chunk_size)]
    results = {}
    threads = [
        threading.Thread(target=sweep_worker,
                         args=(r, dev, data, labels, chunks[r::len(devices)], model,
                               K, d, affine, accumulate, results, soft_m, resp_idx,
                               resp_w, soft_temp))
        for r, dev in enumerate(devices)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    obj = sum(r["obj"] for r in results.values())
    nll = sum(r["obj_nll"] for r in results.values())
    changed = sum(r["changed"] for r in results.values())
    if not accumulate:
        return None, obj, nll, changed
    S = sum(r["S"] for r in results.values())
    msum = sum(r["msum"] for r in results.values())
    cnt = sum(r["cnt"] for r in results.values())
    return (S, msum, cnt), obj, nll, changed


def update_model(moments, data, K, d, affine, seed_gen, device, soft=False):
    """New bases = top-d eigenvectors of each cluster covariance (exact PCA)."""
    S, msum, cnt = (m.to(device) for m in moments)  # CPU-merged -> H2D (P2P unusable)
    n = cnt.clamp(min=1.0)
    means = msum / n[:, None] if affine else torch.zeros(K, DIM, device=device)
    C = S / n[:, None, None]
    if affine:
        C = C - means.unsqueeze(-1) * means.unsqueeze(1)
    trace = C.diagonal(dim1=-2, dim2=-1).sum(-1)
    evals, evecs = torch.linalg.eigh(C)                  # ascending
    # NB: slice by DIM-d rather than -d, since Python's `-0` means 0 (the whole
    # array), not "nothing" -- which would silently break the d=0 (point-cluster /
    # k-means) case.
    U = evecs[..., DIM - d:].flip(-1).contiguous()       # [K, DIM, d], descending
    eigvals = evals[..., DIM - d:].flip(-1).clamp_min(0)

    # Re-seed clusters too small to support a d-dim basis. A fresh point-seed
    # (U=0) cannot out-compete the other clusters' d-dim subspaces -- its residual
    # is the full ||x-mu||^2 while theirs is tiny -- so at large d a collapsed
    # cluster never recovers (observed: d=64 strands 1 cluster, d=128 strands 2 at
    # size 1 every iteration). Instead we *split the largest healthy cluster* along
    # its top principal axis: both halves inherit that cluster's subspace and are
    # offset by +/-1 std along PC1, so the next assignment partitions the parent's
    # tokens between them. This keeps all K subspaces populated even at large d.
    # d=0 (plain k-means) has no subspace to split, so it keeps the random-token
    # re-seed (the validated path).
    min_count = max(2 * d, 64)
    tiny = (cnt < min_count).nonzero().flatten().tolist()
    if tiny:
        order = torch.argsort(cnt, descending=True).tolist()
        donors = [p for p in order if cnt[p] >= 2 * min_count]   # large, splittable
        for i, j in enumerate(tiny):
            print(f"  re-seeding cluster {j} (size {int(cnt[j])})", flush=True)
            if affine and d > 0 and i < len(donors):
                p = donors[i]
                delta = U[p][:, 0] * eigvals[p][0].clamp_min(0).sqrt()  # 1 std along PC1
                means[j] = means[p] - delta
                means[p] = means[p] + delta
                U[j] = U[p].clone()
                eigvals[j] = eigvals[p].clone()
            else:                                                  # k-means / no donor
                t = int(torch.randint(data.shape[0], (1,), generator=seed_gen))
                if affine:
                    means[j] = data[t].to(device).float()
                    U[j] = 0
                else:
                    U[j] = torch.linalg.qr(torch.randn(DIM, d, generator=seed_gen).to(device))[0]
                eigvals[j] = 0
    out = {"U": U.cpu(), "means": means.cpu(), "eigvals": eigvals.cpu(),
           "trace": trace.cpu(), "counts": cnt.to("cpu", torch.int64)}
    if soft:
        # PPCA scale parameters for the MPPCA score. sigma2 is the *average* leftover
        # variance per unmodelled dimension (the ML PPCA noise estimate) and `mixing` the
        # mixture weight pi_j. Both are floored: an unfloored sigma2 -> 0 is the classic
        # EM collapse (a component wins by shrinking onto a handful of points) and would
        # drag the score's log-det term to -inf with it. Emitted only under --soft, so a
        # hard run's model.pt keeps exactly the fields it always had.
        out["sigma2"] = ((trace - eigvals.sum(1)).clamp_min(0) / max(DIM - d, 1)
                         ).clamp_min(SIGMA2_FLOOR).cpu()
        out["mixing"] = (cnt / cnt.sum().clamp_min(1e-12)).clamp_min(1e-12).cpu()
        out["soft_counts"] = cnt.cpu()                   # fractional responsibility mass
    return out


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.gpus < 1:
        raise SystemExit("Needs at least one CUDA GPU.")
    devices = [f"cuda:{i}" for i in range(args.gpus)]
    K, d, affine = args.clusters, args.dim, not args.linear
    os.makedirs(args.out, exist_ok=True)

    soft_m = args.soft_topm if args.soft else 0
    if soft_m and d == 0:
        raise SystemExit("--soft needs --dim > 0: with no subspace there is no PPCA "
                         "covariance to build a likelihood from (plain k-means has only "
                         "a centroid). Use --dim 16 or higher.")
    if soft_m and not 1 <= soft_m <= K:
        raise SystemExit(f"--soft-topm must be in 1..K ({K}); got {soft_m}.")

    data, file_ids, cell_ids, sampled = load_tokens(args)
    T = data.shape[0]
    g = torch.Generator().manual_seed(args.seed + 1)
    labels = torch.full((T,), -1, dtype=torch.int32)
    # Truncated responsibilities, filled by every soft sweep and saved from the last one.
    resp_idx = torch.zeros(T, soft_m, dtype=torch.int16) if soft_m else None
    resp_w = torch.zeros(T, soft_m, dtype=torch.float16) if soft_m else None
    if soft_m:
        print(f"Soft (MPPCA EM) mode: top-{soft_m} responsibilities per token "
              f"(+{T * soft_m * 4 / 2**30:.2f} GiB)", flush=True)

    # Init: K random tokens as seeds with zero bases -> first sweep assigns
    # each token to its nearest seed (a plain k-means step), which breaks the
    # symmetry that a random partition cannot (all its PCA bases coincide).
    seeds = torch.randperm(T, generator=g)[:K]
    if affine:
        model = {"U": torch.zeros(K, DIM, d), "means": data[seeds].float()}
    else:
        model = {"U": torch.linalg.qr(torch.randn(K, DIM, d, generator=g))[0],
                 "means": torch.zeros(K, DIM)}

    history = []
    t_start = time.time()
    for it in range(1, args.iters + 1):
        t0 = time.time()
        # Iteration 1 is always hard: the init has U=0 / no eigvals, so there is no
        # covariance to score a likelihood with. Bases (and sigma2/mixing) exist from
        # the first update_model onwards, and the sweep switches to soft then.
        use_soft = soft_m if "sigma2" in model else 0
        moments, obj, nll, changed = run_sweep(
            devices, data, labels, model, K, d, affine,
            accumulate=True, chunk_size=args.chunk_size,
            soft_m=use_soft, resp_idx=resp_idx, resp_w=resp_w, soft_temp=args.soft_temp)
        model = update_model(moments, data, K, d, affine, g, devices[0], soft=bool(soft_m))
        dt = time.time() - t0
        frac = changed / T
        sizes = model["counts"]
        h = {"iter": it, "obj_per_token": obj / T, "frac_changed": frac,
             "size_min": int(sizes.min()), "size_max": int(sizes.max())}
        if use_soft:
            h["nll_per_token"] = nll / T
        history.append(h)
        print(f"iter {it:2d}: obj/token={obj / T:.4f}  "
              + (f"nll/token={nll / T:.4f}  " if use_soft else "")
              + f"changed={frac:.4%}  "
              f"sizes[min/med/max]={int(sizes.min())}/{int(sizes.median())}/{int(sizes.max())}  "
              f"({dt:.1f}s)", flush=True)
        if frac < args.tol:
            print("Converged.", flush=True)
            break

    # Final labeling under the final bases. This relabels every token under the
    # model we are about to save, so the assignment it produces is the one stored
    # in assignments.pt. Reconcile counts to these final labels (update_model fit
    # from the *previous* assignment, so model["counts"] would otherwise lag by one
    # relabelling step), and record this sweep's objective as the true objective of
    # the saved model (history[-1] is the previous model's objective).
    use_soft = soft_m if "sigma2" in model else 0
    _, obj, nll, changed = run_sweep(devices, data, labels, model, K, d, affine,
                                     accumulate=False, chunk_size=args.chunk_size,
                                     soft_m=use_soft, resp_idx=resp_idx, resp_w=resp_w,
                                     soft_temp=args.soft_temp)
    # `counts` stays the integer bincount of the saved hard labels even under --soft, so
    # the cross-algorithm invariant counts == bincount(assignments['label']) holds for
    # every reader; the fractional masses live in `soft_counts`.
    model["counts"] = torch.bincount(labels.long(), minlength=K)
    final_obj_per_token = obj / T
    print(f"final: obj/token={final_obj_per_token:.4f}  "
          + (f"nll/token={nll / T:.4f}  " if use_soft else "")
          + f"changed={changed / T:.4%}", flush=True)

    fp = sample_fingerprint(sampled, min(args.tokens_per_file, N_CELLS), args.seed)
    config = {**vars(args), "method": "subspace_em" if soft_m else "subspace_kmeans"}
    save_model(args.out, **model, config=config, history=history,
               sampled_files=sampled, sample_fingerprint=fp,
               final_obj_per_token=final_obj_per_token)
    save_assignments(args.out, file_id=file_ids, cell_id=cell_ids, label=labels,
                     resp_idx=resp_idx, resp_w=resp_w)
    if use_soft:
        top1 = resp_w[:, 0].float()
        print(f"Responsibilities: mean top-1 weight {top1.mean():.3f}; "
              f"tokens with top-1 < 0.5 (genuinely ambiguous): "
              f"{(top1 < 0.5).float().mean():.1%}; "
              f"< 0.9: {(top1 < 0.9).float().mean():.1%}")
    evr = model["eigvals"].sum(1) / model["trace"].clamp(min=1e-12) if d > 0 else torch.zeros(K)
    print(f"Saved model.pt + assignments.pt to {args.out}/")
    print(f"Explained variance ratio (top-{d}): min={evr.min():.3f} "
          f"median={evr.median():.3f} max={evr.max():.3f}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
