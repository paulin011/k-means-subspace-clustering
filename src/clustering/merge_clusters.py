#!/usr/bin/env python3
"""Agglomerative merging of K-subspace clusters: fit at large K, merge down to the K you want.

Motivation (docs/ideas/overcluster_merge.md): fitting K-subspaces with more clusters than
you need and then merging is a standard cure for k-means-style local minima -- the
over-parameterised seeding explores partitions a direct fit at the target K cannot reach,
and the merge step repairs the resulting over-segmentation.

Criterion (--criterion residual, the default) is Ward's linkage with *this project's own
objective* substituted for Ward's ESS: merge the pair whose union increases the total
orthogonal residual least,

    dR(a,b) = R_{a+b} - R_a - R_b >= 0,   R_j = n_j (tr(C_j) - sum_{i<=d} lam_ji)

so a merge's price is denominated in the units the run already reports (obj_per_token).
The --criterion affinity alternative (mean squared principal-angle cosine, as in
worldmap.build_affinity_matrix) is offered for comparison but is NOT the default: affinity
sees subspace *orientation* only and is blind to mean placement and density -- two clusters
can share an orientation and sit far apart along it (README/CLAUDE.md: c123 maxAff 0.697 /
near-tie 1.3% vs c122 maxAff 0.668 / near-tie 45.7%).

Scoring is EXACT and needs no data pass beyond the moments the fit already builds. R_j is
already in model.pt (counts, trace, eigvals), and a merged cluster's moments follow in
closed form from the parents' -- second moments are simply additive:

    n  = n_a + n_b,  delta = mu_a - mu_b,  mu = (n_a msum_a + n_b msum_b)/n
    C  = (S_a + S_b)/n - mu mu^T                                      (exact)

The one non-trivial term, sum_{i<=d} lam_i(C), comes from subspace iteration warm-started at
V = orth([U_a, U_b, delta]) -- already close to the merged top-d, so a few steps converge it
(measured on real tokens: median error in dR of 4.5% / 0.50% / 0.099% / 0.023% / 0.0016% at
0..5 steps; --power-iters defaults to 5). C is never materialised -- C@V = (S_a@V + S_b@V)/n
- mu(mu^T V) uses the stored S directly -- which keeps the whole thing a bmm at 10.3 TFLOPS
instead of 46.6 ms/pair of full eigh. Feed it moments.pt from subspace_kmeans --save-moments.

Without moments.pt the script falls back to a PPCA surrogate that models each parent's
unmodelled tail as isotropic (C_j ~= U_j diag(lam_j) U_j^T + s_j (I - U_j U_j^T), the same
covariance model --soft commits to), reducing the spectrum to a q x q eigh with q <= 2d+1.
That assumption is weak on real tokens -- ~121 effective dims against d=64 -- and was
MEASURED at ~35% error in dR, enough to pick the wrong cheapest pair. It still ranks pairs
usefully (Spearman 0.94), so it remains a usable fallback for runs predating --save-moments
(e.g. v6), but a --merge-threshold denominated in objective percent is only meaningful on
the exact path.

Because the cascade is that cheap it is ALWAYS run to completion (down to --min-k), and the
threshold is applied afterwards as a *cut* through the recorded dendrogram. Re-tuning the
threshold therefore costs nothing and needs no refit; merge_log.json holds every step.

Counts, means and traces are exact in both modes; each accepted merge is additionally
re-fitted with a full eigh of the merged covariance, and --refit-iters > 0 re-derives every
surviving basis from real data. d=0 (plain k-means) reduces to Ward's classic exact formula
dR = (n_a n_b/n)|delta|^2, which is the identity that pins down the delta term's factor.

Outputs in --out: model.pt + assignments.pt (the standard cluster_io schema, so
analyze_clusters.py / holdout_eval.py / temporal_spatial.py / file_signature.py all read a
merged run unchanged), sample.json, and merge_log.json (the full dendrogram).

Typical use -- merge a K=256 fit down to 128 for a like-for-like comparison against v6:

  python3 src/clustering/merge_clusters.py --dir runs/clustering/v11_k256_d64 \
      --out runs/clustering/v12_k256to128_d64 --target-k 128 --refit-iters 3

NOTE --refit-iters > 0 re-loads the parent's full token sample (~352 GB RAM for a 7000-file
run), so it must not overlap with the fit that produced it.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.cluster_io import (DIM, N_CELLS, load_tokens, sample_fingerprint,
                               save_assignments, save_model)
import subspace_kmeans as sk  # same directory; reuse the validated sweep/update kernels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="input run directory (with model.pt)")
    p.add_argument("--out", required=True, help="output run directory")
    p.add_argument("--criterion", choices=["residual", "affinity"], default="residual",
                   help="residual: Ward linkage on the K-subspaces objective (default). "
                        "affinity: mean squared principal-angle cosine -- orientation only, "
                        "blind to mean placement; for comparison")
    p.add_argument("--merge-threshold", type=float, default=0.002,
                   help="PER-MERGE budget: stop when the single cheapest remaining merge "
                        "costs more than this, as a fraction of the objective (dR/R_total). "
                        "Note individual merges are cheap -- MEASURED at 0.0002..0.0004 on a "
                        "K=256 cascade -- so this is a guard against one catastrophic merge, "
                        "NOT the knob that picks K; use --max-total-cost or --target-k for "
                        "that. Under --criterion affinity it is instead a principal-angle "
                        "affinity in [0,1] (merge while the best pair exceeds it; try 0.9)")
    p.add_argument("--max-total-cost", type=float, default=0.0024,
                   help="CUMULATIVE budget and the main knob (--criterion residual only): "
                        "stop once the merges given up this much of the objective in total. "
                        "Default 0.0024 = the measured v6/v7/v8 seed spread of 0.24%%, i.e. "
                        "'merge until the damage equals run-to-run seed noise'. Pass a huge "
                        "value to disable. Ignored when --target-k is given")
    p.add_argument("--target-k", type=int, default=None,
                   help="merge down to exactly this many clusters. TAKES PRECEDENCE over "
                        "both budgets -- an exact K is exactly what they would otherwise "
                        "choose for you (the reported cost then tells you what it cost)")
    p.add_argument("--refit-iters", type=int, default=3,
                   help="assignment/update sweeps on the original token sample after the "
                        "cut, to replace surrogate bases with exact PCA ones (0 = no data "
                        "pass at all: labels are remapped through the merge tree instead)")
    p.add_argument("--min-k", type=int, default=2, help="run the dendrogram down to this K")
    p.add_argument("--pair-batch", type=int, default=None,
                   help="pairs scored per batch (default 32 exact / 256 surrogate; exact "
                        "mode gathers 2 x [DIM,DIM] per pair, ~34 MB each)")
    p.add_argument("--moments", default=None,
                   help="path to moments.pt for EXACT merge scoring (default: <dir>/moments.pt "
                        "if present). Without it the PPCA surrogate is used, which on real "
                        "tokens misestimates dR by ~35%% -- fine for ranking, not for a "
                        "threshold in objective percent. Produce one with "
                        "subspace_kmeans.py --save-moments")
    p.add_argument("--surrogate", action="store_true",
                   help="force the PPCA surrogate even when moments.pt exists")
    p.add_argument("--power-iters", type=int, default=5,
                   help="subspace-iteration steps for the exact top-d eigenvalues. "
                        "Convergence is geometric and spectrum-dependent: measured on real "
                        "tokens the median error in dR falls 4.5%% -> 0.50%% -> 0.099%% -> "
                        "0.023%% -> 0.0016%% for 0..5 steps, so 5 is comfortably converged. "
                        "A near-degenerate spectrum at the d-boundary converges slower "
                        "(synthetic worst case: 0.33%% at 5); the merge ORDER is stable long "
                        "before the value is (argmin agrees even at 0 steps)")
    p.add_argument("--src", default=None, help="latent dir for --refit-iters (default: parent run's)")
    p.add_argument("--chunk-size", type=int, default=131072, help="tokens per GPU chunk (refit)")
    p.add_argument("--tol", type=float, default=1e-3, help="refit early-stop on frac changed")
    p.add_argument("--gpus", type=int, default=min(2, torch.cuda.device_count()))
    p.add_argument("--load-workers", type=int, default=16)
    p.add_argument("--max-ram-gb", type=float, default=420.0)
    p.add_argument("--seed", type=int, default=None, help="override the parent run's seed")
    return p.parse_args()


# ---------------------------------------------------------------- state


def load_state(model_path, device):
    """Per-cluster sufficient statistics for merging, straight out of model.pt.

    Linear algebra stays float32 (that is what the bases are); the residual arithmetic is
    float64 because dR is a difference of ~1e9-sized terms and float32 would leave only
    ~2 significant digits in it.
    """
    m = torch.load(model_path, map_location="cpu", weights_only=False)
    K, D, d = m["U"].shape
    if D != DIM:
        raise SystemExit(f"model.pt has DIM={D}, expected {DIM}")
    trace = m["trace"].double().to(device)
    eigvals = m["eigvals"].float().to(device)
    counts = m["counts"].double().to(device)
    # s_j = average leftover variance per unmodelled dimension (the ML PPCA noise estimate,
    # identical to subspace_kmeans's `sigma2`). Floored for the same reason it is there.
    resid_var = (trace - eigvals.double().sum(1)).clamp_min(0)
    sigma2 = (resid_var / max(D - d, 1)).clamp_min(sk.SIGMA2_FLOOR)
    state = {"U": m["U"].float().to(device), "means": m["means"].float().to(device),
             "eigvals": eigvals, "trace": trace, "counts": counts, "sigma2": sigma2}
    return state, m, K, d


def load_moments(path, K, d, device, chunk=16):
    """Read moments.pt and re-derive the per-cluster state from it by exact PCA.

    Deriving the state here rather than reusing model.pt's fields is deliberate: moments.pt
    is accumulated during the *final relabel sweep*, so it belongs to the labels in
    assignments.pt, whereas model.pt's bases were fitted one sweep earlier. Re-fitting makes
    state, moments and labels mutually consistent -- which is what lets the merged residual
    be exact rather than merely close.
    """
    m = torch.load(path, map_location="cpu", weights_only=False)
    if m["K"] != K or m["dim"] != d:
        raise SystemExit(f"moments.pt is K={m['K']} d={m['dim']}, model.pt is K={K} d={d}")
    S = m["S"].to(device).float()
    msum = m["msum"].to(device).float()
    cnt = m["cnt"].to(device).float()
    print(f"  moments.pt: S {tuple(S.shape)} ({S.numel() * 4 / 2**30:.2f} GiB on {device})",
          flush=True)

    U = torch.zeros(K, DIM, d, device=device)
    means = torch.zeros(K, DIM, device=device)
    eigvals = torch.zeros(K, d, device=device)
    trace = torch.zeros(K, dtype=torch.float64, device=device)
    for lo in range(0, K, chunk):
        hi = min(lo + chunk, K)
        n = cnt[lo:hi].clamp_min(1.0).view(-1, 1, 1)
        mu = msum[lo:hi] / n.view(-1, 1)
        C = S[lo:hi] / n - mu.unsqueeze(2) @ mu.unsqueeze(1)
        C = 0.5 * (C + C.transpose(1, 2))
        evals, evecs = torch.linalg.eigh(C)
        means[lo:hi] = mu
        trace[lo:hi] = C.diagonal(dim1=-2, dim2=-1).sum(-1).double()
        if d > 0:
            U[lo:hi] = evecs[..., DIM - d:].flip(-1)
            eigvals[lo:hi] = evals[..., DIM - d:].flip(-1).clamp_min(0)
    counts = cnt.double()
    sigma2 = ((trace - eigvals.double().sum(1)).clamp_min(0)
              / max(DIM - d, 1)).clamp_min(sk.SIGMA2_FLOOR)
    state = {"U": U, "means": means, "eigvals": eigvals, "trace": trace,
             "counts": counts, "sigma2": sigma2}
    return state, {"S": S, "msum": msum, "cnt": cnt}


def residual(state, idx=None):
    """R_j = n_j (tr(C_j) - sum_{i<=d} lam_ji), the cluster's contribution to the objective."""
    tr, ev, cnt = state["trace"], state["eigvals"].double(), state["counts"]
    if idx is not None:
        tr, ev, cnt = tr[idx], ev[idx], cnt[idx]
    return cnt * (tr - ev.sum(-1)).clamp_min(0)


# ---------------------------------------------------------------- pair scoring


def score_pairs(state, pairs, d, device, batch=256, want_basis=False, mom=None,
                power_iters=5):
    """dR (float64) -- and, optionally, the merged bases -- for a batch of candidate pairs.

    Both scoring modes assemble dR in the same algebraically reduced form

        dR = (n_a n_b/n)|delta|^2 + n_a sum(lam_a) + n_b sum(lam_b) - n sum_{i<=d} lam_i(C)

    rather than as R_ab - R_a - R_b. The identity behind it is
    n_a|mu_a|^2 + n_b|mu_b|^2 - n|mu|^2 = (n_a n_b/n)|delta|^2, and it matters numerically:
    the n_j tr_j terms are the largest quantities in play and identical on both sides, so
    here they cancel *exactly* instead of leaving dR as the low-order residue of a
    ~1e9-sized subtraction. Only the merged sum_{i<=d} lam_i(C) differs between modes.

    mom=None -- PPCA surrogate (see the module docstring): C ~= W + sI with W of rank
    <= 2d+1, so the spectrum comes from a q x q eigh with q <= 129. Needs nothing but
    model.pt, but assumes the unmodelled tail is isotropic. That assumption is weak on real
    tokens (~121 effective dims vs d=64), measured at ~35% error on dR -- fine for ranking
    (Spearman 0.94) but not for a threshold denominated in objective percent.

    mom given -- EXACT, from the stored second moments. C = (S_a+S_b)/n - mu mu^T is exact
    because S is additive over a merge, and its top-d eigenvalues come from subspace
    iteration warm-started at V = orth([U_a, U_b, delta]) -- already close to the answer, so
    a few `power_iters` steps converge it (see the flag's help for the measured curve). C is
    never materialised: C@V = (S_a@V + S_b@V)/n - mu(mu^T V) reuses the stored S directly,
    which keeps this a bmm (10.3 TFLOPS measured) rather than 46.6 ms/pair of full eigh.
    Verified against fp64 brute force: the fp32 moments themselves cost only ~2e-8 relative,
    so the iteration is the only error term.
    """
    U, means, eigvals = state["U"], state["means"], state["eigvals"]
    trace, counts, sigma2 = state["trace"], state["counts"], state["sigma2"]
    P = pairs.shape[0]
    out = torch.empty(P, dtype=torch.float64, device=device)
    aff = torch.empty(P, dtype=torch.float64, device=device)
    U_new = torch.empty(P, DIM, d, device=device) if want_basis else None
    ev_new = torch.empty(P, d, dtype=torch.float64, device=device) if want_basis else None
    tr_new = torch.empty(P, dtype=torch.float64, device=device) if want_basis else None

    for lo in range(0, P, batch):
        pa, pb = pairs[lo:lo + batch, 0], pairs[lo:lo + batch, 1]
        B = pa.shape[0]
        na, nb = counts[pa], counts[pb]
        n = na + nb
        wa, wb = (na / n), (nb / n)                             # [B]
        wd = na * nb / (n * n)                                  # [B]
        delta = means[pa] - means[pb]                           # [B, DIM] float32
        dn2 = delta.double().pow(2).sum(1)                       # |delta|^2, float64
        dnorm = dn2.sqrt().clamp_min(1e-12).float()
        dhat = delta / dnorm.unsqueeze(1)                        # unit, for conditioning

        # V = orth([U_a, U_b, delta]) -- the reduced basis in surrogate mode, the subspace
        # iteration's warm start in exact mode. Rank deficiency is harmless either way: QR
        # still returns orthonormal columns, and directions carrying no variance score 0.
        A = torch.cat([U[pa], U[pb], dhat.unsqueeze(2)], dim=2)  # [B, DIM, 2d+1]
        V = torch.linalg.qr(A, mode="reduced")[0]                # [B, DIM, q] float32
        q = V.shape[2]
        take = min(d, q)

        if mom is None:
            Vd = V.double()
            Ga = Vd.transpose(1, 2) @ U[pa].double()             # [B, q, d]
            Gb = Vd.transpose(1, 2) @ U[pb].double()
            gd = (Vd.transpose(1, 2) @ dhat.double().unsqueeze(2)).squeeze(2)   # [B, q]
            # lam - s >= 0 (d-th largest eigenvalue >= mean of the tail), so M is PSD.
            ca = (eigvals[pa].double() - sigma2[pa].unsqueeze(1)).clamp_min(0) * wa.unsqueeze(1)
            cb = (eigvals[pb].double() - sigma2[pb].unsqueeze(1)).clamp_min(0) * wb.unsqueeze(1)
            M = (Ga * ca.unsqueeze(1)) @ Ga.transpose(1, 2)
            M += (Gb * cb.unsqueeze(1)) @ Gb.transpose(1, 2)
            M += (wd * dn2).view(B, 1, 1) * (gd.unsqueeze(2) @ gd.unsqueeze(1))
            s = (na * sigma2[pa] + nb * sigma2[pb]) / n           # merged noise level
            V_out = Vd
        else:
            Sa, Sb = mom["S"][pa], mom["S"][pb]                   # [B, DIM, DIM] float32
            mu = ((mom["msum"][pa] + mom["msum"][pb]) / n.float().unsqueeze(1))

            def CV(Vx):
                Z = torch.baddbmm(torch.bmm(Sa, Vx), Sb, Vx)      # S_a@V + S_b@V
                Z /= n.float().view(B, 1, 1)
                return Z - mu.unsqueeze(2) @ (mu.unsqueeze(1) @ Vx)

            for _ in range(power_iters):
                V = torch.linalg.qr(CV(V), mode="reduced")[0]
            M = (V.transpose(1, 2) @ CV(V)).double()              # Rayleigh-Ritz
            s = torch.zeros_like(n)                               # no isotropic tail term
            V_out = V.double()

        M = 0.5 * (M + M.transpose(1, 2))                         # symmetrise fp noise
        evals, evecs = torch.linalg.eigh(M)                       # ascending
        top = evals[:, q - take:].flip(-1).clamp_min(0)           # [B, take] descending

        # NB the delta term is n*wd*|delta|^2 = (n_a n_b/n)|delta|^2 -- it enters dR through
        # n*tr(C), not tr(C). At d=0 this collapses to exactly that term, i.e. Ward's
        # classic linkage, which is the check that pins the factor down.
        dR = (na * nb / n * dn2
              + na * eigvals[pa].double().sum(1) + nb * eigvals[pb].double().sum(1)
              - n * (top.sum(1) + d * s))
        out[lo:lo + B] = dR.clamp_min(0)

        if d > 0:
            Uab = torch.einsum("bik,bkj->bij", U[pa].transpose(1, 2), U[pb])   # [B, d, d]
            aff[lo:lo + B] = Uab.double().pow(2).sum((1, 2)) / d
        else:
            mu_a = means[pa] / means[pa].norm(dim=1, keepdim=True).clamp_min(1e-12)
            mu_b = means[pb] / means[pb].norm(dim=1, keepdim=True).clamp_min(1e-12)
            aff[lo:lo + B] = (1.0 + (mu_a * mu_b).sum(1).double()) / 2.0

        if want_basis:
            Vt = evecs[:, :, q - take:].flip(-1)                  # [B, q, take]
            U_new[lo:lo + B, :, :take] = (V_out @ Vt).float()
            ev_new[lo:lo + B, :take] = top + s.unsqueeze(1)
            if take < d:                                          # q < d: pad at tail level
                U_new[lo:lo + B, :, take:] = 0
                ev_new[lo:lo + B, take:] = s.unsqueeze(1)
            tr_new[lo:lo + B] = (na * trace[pa] + nb * trace[pb]) / n + wd * dn2

    return out, aff, (U_new, ev_new, tr_new) if want_basis else None


def apply_merge(state, a, b, d, device, mom=None):
    """Fold cluster b into slot a; slot b is left for the caller to deactivate.

    With moments the merged cluster is re-fitted *exactly* (one full eigh of the merged
    covariance, 46.6 ms) rather than through the iteration used for scoring -- the accepted
    merge is kept as the model, so it is worth the extra accuracy.
    """
    na, nb = state["counts"][a].clone(), state["counts"][b].clone()
    n = na + nb
    if mom is not None:
        mom["S"][a] += mom["S"][b]
        mom["msum"][a] += mom["msum"][b]
        mom["cnt"][a] = n.float()
        mu = mom["msum"][a] / n.float()
        C = mom["S"][a] / n.float() - torch.outer(mu, mu)
        C = 0.5 * (C + C.T)
        evals, evecs = torch.linalg.eigh(C)
        state["means"][a] = mu
        state["U"][a] = evecs[:, DIM - d:].flip(-1) if d > 0 else state["U"][a]
        state["eigvals"][a] = evals[DIM - d:].flip(-1).clamp_min(0) if d > 0 \
            else state["eigvals"][a]
        state["trace"][a] = C.diagonal().sum().double()
    else:
        pair = torch.tensor([[a, b]], device=device)
        _, _, basis = score_pairs(state, pair, d, device, batch=1, want_basis=True)
        U_new, ev_new, tr_new = basis
        state["means"][a] = (na.float() * state["means"][a]
                             + nb.float() * state["means"][b]) / n.float()
        state["U"][a] = U_new[0]
        state["eigvals"][a] = ev_new[0].float()
        state["trace"][a] = tr_new[0]
    state["counts"][a] = n
    state["sigma2"][a] = ((state["trace"][a] - state["eigvals"][a].double().sum()).clamp_min(0)
                          / max(DIM - d, 1)).clamp_min(sk.SIGMA2_FLOOR)


# ---------------------------------------------------------------- cascade


def run_cascade(state, K, d, device, criterion, pair_batch, min_k, r_total, mom=None,
                power_iters=5):
    """Greedy agglomerative merge down to `min_k`, recording every step.

    Runs to completion regardless of the threshold: the dendrogram is minutes of work at
    most, so the cut becomes free post-processing and re-tuning the threshold needs no
    recompute.
    """
    active = torch.ones(K, dtype=torch.bool)
    key = torch.full((K, K), float("inf"), dtype=torch.float64, device=device)
    cost = torch.full((K, K), float("inf"), dtype=torch.float64, device=device)
    affm = torch.full((K, K), -1.0, dtype=torch.float64, device=device)

    def fill(pairs):
        if pairs.numel() == 0:
            return
        dR, aff, _ = score_pairs(state, pairs, d, device, batch=pair_batch, mom=mom,
                                 power_iters=power_iters)
        k = dR / r_total if criterion == "residual" else (1.0 - aff)
        for src, dst in ((k, key), (dR, cost), (aff, affm)):
            dst[pairs[:, 0], pairs[:, 1]] = src
            dst[pairs[:, 1], pairs[:, 0]] = src

    t0 = time.time()
    fill(torch.combinations(torch.arange(K, device=device), 2))
    print(f"  scored {K * (K - 1) // 2:,} initial pairs in {time.time() - t0:.1f}s", flush=True)

    log, n_active = [], K
    while n_active > min_k:
        flat = int(torch.argmin(key))
        a, b = flat // K, flat % K
        if not torch.isfinite(key[a, b]):
            break
        a, b = (a, b) if state["counts"][a] >= state["counts"][b] else (b, a)
        log.append({"step": len(log) + 1, "k_before": n_active, "k_after": n_active - 1,
                    "into": int(a), "from": int(b),
                    "n_into": int(state["counts"][a]), "n_from": int(state["counts"][b]),
                    "delta_r": float(cost[a, b]), "rel_cost": float(cost[a, b] / r_total),
                    "affinity": float(affm[a, b]), "key": float(key[a, b])})
        apply_merge(state, a, b, d, device, mom=mom)
        active[b] = False
        key[b, :] = key[:, b] = float("inf")
        cost[b, :] = cost[:, b] = float("inf")
        n_active -= 1
        others = active.nonzero().flatten().to(device)
        others = others[others != a]
        if others.numel():
            fill(torch.stack([torch.full_like(others, a), others], dim=1))
        if len(log) % 32 == 0:
            print(f"    {n_active} clusters left ({time.time() - t0:.0f}s)", flush=True)
    print(f"  cascade {K} -> {n_active} in {time.time() - t0:.1f}s", flush=True)
    return log


def cut_index(log, criterion, threshold, target_k, K, max_total=None):
    """Number of merges to keep.

    --target-k, when given, WINS OUTRIGHT: asking for an exact K and silently getting a
    different one because a threshold tripped is the worst possible surprise, and the
    thresholds' job (choosing K for you) is precisely the one --target-k takes over.

    Otherwise the cut is the first step violating either budget. Two are offered because
    they answer different questions, and the per-merge one alone is a trap:

      per-merge (`threshold`)  guards against one catastrophic merge. MEASURED on a K=256
          cascade, individual merges cost ~0.0002-0.0004 of the objective and climb very
          slowly, so any per-merge bound near the run-to-run seed spread (0.24%) never fires
          and means "merge everything".
      cumulative (`max_total`) is the one denominated like the numbers this project quotes:
          total objective given up so far. On that same cascade the cumulative cost reached
          1.67% over 56 merges and crossed the 0.24% seed spread after 14 -- i.e. it is the
          budget that actually discriminates, hence the default.

    The FIRST violating step ends the cut: greedy Ward on a cascade is not guaranteed
    inversion-free, so a later cheap step must not resurrect it past an expensive one.
    """
    if target_k is not None:
        return max(K - target_k, 0)
    n, total = len(log), 0.0
    for i, e in enumerate(log):
        total += e["rel_cost"]
        bad = (e["rel_cost"] > threshold) if criterion == "residual" \
            else (e["affinity"] < threshold)
        if criterion == "residual" and max_total is not None and total > max_total:
            bad = True
        if bad:
            n = i
            break
    return n


def label_map(log, n_merges, K):
    """Original cluster id -> compacted merged id, from the first `n_merges` steps."""
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in log[:n_merges]:
        parent[find(e["from"])] = find(e["into"])
    roots = sorted({find(i) for i in range(K)})
    compact = {r: i for i, r in enumerate(roots)}
    return torch.tensor([compact[find(i)] for i in range(K)], dtype=torch.int64), len(roots)


# ---------------------------------------------------------------- refit


def refit(args, parent_cfg, state_sel, K_new, d, devices):
    """Re-derive the merged bases from real data on the parent run's exact token sample."""
    ns = argparse.Namespace(
        src=args.src or parent_cfg.get("src", "latents_2"), out=args.out,
        num_files=parent_cfg.get("num_files", 1500),
        files_from=os.path.join(args.dir, "sample.json"),
        tokens_per_file=parent_cfg.get("tokens_per_file", N_CELLS),
        seed=args.seed if args.seed is not None else parent_cfg.get("seed", 0),
        load_workers=args.load_workers, max_ram_gb=args.max_ram_gb)
    data, file_ids, cell_ids, sampled = load_tokens(ns)
    T = data.shape[0]
    labels = torch.full((T,), -1, dtype=torch.int32)
    g = torch.Generator().manual_seed(ns.seed + 1)
    affine = not parent_cfg.get("linear", False)
    model = {"U": state_sel["U"].cpu(), "means": state_sel["means"].cpu()}

    history = []
    for it in range(1, args.refit_iters + 1):
        t0 = time.time()
        moments, obj, _, changed = sk.run_sweep(devices, data, labels, model, K_new, d,
                                                affine, accumulate=True,
                                                chunk_size=args.chunk_size)
        model = sk.update_model(moments, data, K_new, d, affine, g, devices[0])
        sizes, frac = model["counts"], changed / T
        history.append({"iter": it, "obj_per_token": obj / T, "frac_changed": frac,
                        "size_min": int(sizes.min()), "size_max": int(sizes.max())})
        print(f"refit {it:2d}: obj/token={obj / T:.4f}  changed={frac:.4%}  "
              f"sizes[min/med/max]={int(sizes.min())}/{int(sizes.median())}/"
              f"{int(sizes.max())}  ({time.time() - t0:.1f}s)", flush=True)
        if frac < args.tol:
            print("Converged.", flush=True)
            break

    _, obj, _, changed = sk.run_sweep(devices, data, labels, model, K_new, d, affine,
                                      accumulate=False, chunk_size=args.chunk_size)
    model["counts"] = torch.bincount(labels.long(), minlength=K_new)
    print(f"final: obj/token={obj / T:.4f}  changed={changed / T:.4%}", flush=True)
    return model, history, obj / T, (file_ids, cell_ids, labels), sampled, ns


# ---------------------------------------------------------------- main


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    state, parent, K, d = load_state(os.path.join(args.dir, "model.pt"), device)
    cfg = parent.get("config", {})
    print(f"Loaded {args.dir}: K={K} d={d} tokens={float(state['counts'].sum()):,.0f} "
          f"method={cfg.get('method')}", flush=True)

    mom_path = args.moments or os.path.join(args.dir, "moments.pt")
    exact = os.path.exists(mom_path) and not args.surrogate
    if exact:
        state, mom = load_moments(mom_path, K, d, device)
    else:
        mom = None
        print("  WARNING: no moments.pt -- falling back to the PPCA surrogate. dR is then "
              "accurate to ~35% on real tokens, so --merge-threshold is only approximately "
              "a percentage of the objective. Re-run the fit with --save-moments for an "
              "exact threshold.", flush=True)
    pair_batch = args.pair_batch or (32 if exact else 256)
    T_parent = float(state["counts"].sum())
    r_total = float(residual(state).sum())

    # Invariant: the objective reassembled from the per-cluster fields must reproduce the
    # objective the parent run reported. Catches a stale or mismatched model.pt/moments.pt.
    ref = parent.get("final_obj_per_token") or parent["history"][-1]["obj_per_token"]
    got = r_total / T_parent
    # In exact mode a small NEGATIVE gap here is expected, not a fault: moments.pt is
    # accumulated during the final relabel sweep, so re-fitting the bases to those final
    # labels (load_moments) legitimately lowers the residual by one half-iteration's worth.
    print(f"  objective check: from {'moments' if exact else 'model fields'} {got:.4f} "
          f"vs reported {ref:.4f} ({got / ref - 1:+.2e}"
          + (", refit to final labels" if exact and got <= ref else "") + ")", flush=True)

    print(f"Cascade ({args.criterion} criterion, "
          f"{'exact' if exact else 'surrogate'} dR) ...", flush=True)
    # tf32 would cost ~10 mantissa bits in the C@V products that dR is differenced from.
    torch.backends.cuda.matmul.allow_tf32 = False
    log = run_cascade(state, K, d, device, args.criterion, pair_batch,
                      args.min_k, r_total, mom=mom, power_iters=args.power_iters)
    torch.backends.cuda.matmul.allow_tf32 = True
    inversions = sum(1 for i in range(1, len(log)) if log[i]["key"] < log[i - 1]["key"])

    n_merges = cut_index(log, args.criterion, args.merge_threshold, args.target_k, K,
                         max_total=args.max_total_cost)
    lmap, K_new = label_map(log, n_merges, K)
    cum = sum(e["rel_cost"] for e in log[:n_merges])
    print(f"Cut after {n_merges} merges: K {K} -> {K_new}  "
          f"(cumulative objective cost {cum:+.4%}; {inversions} cascade inversions)", flush=True)
    if args.target_k is not None:
        would = cut_index(log, args.criterion, args.merge_threshold, None, K,
                          max_total=args.max_total_cost)
        print(f"  --target-k took precedence; the budgets alone would have cut after "
              f"{would} merges (K -> {K - would})", flush=True)
    elif n_merges < len(log):
        print(f"  next merge would cost {log[n_merges]['rel_cost']:.4%} "
              f"(per-merge cap {args.merge_threshold}, cumulative cap "
              f"{args.max_total_cost})", flush=True)

    # Replay to the cut. Re-running the accepted merges from the pristine state is cheaper
    # than snapshotting the [K, 2048, d] bases at every step of the full cascade (the
    # cascade mutates S/state in place, so the pristine state has to be re-read).
    torch.backends.cuda.matmul.allow_tf32 = False
    if exact:
        del mom, state
        torch.cuda.empty_cache()
        state, mom = load_moments(mom_path, K, d, device)
    else:
        state, _, _, _ = load_state(os.path.join(args.dir, "model.pt"), device)
    alive = list(range(K))
    for e in log[:n_merges]:
        apply_merge(state, e["into"], e["from"], d, device, mom=mom)
        alive.remove(e["from"])
    torch.backends.cuda.matmul.allow_tf32 = True
    sel = torch.tensor(alive, device=device)
    state_sel = {k: v[sel] for k, v in state.items()}
    order = torch.argsort(lmap[torch.tensor(alive)])       # match label_map's compaction
    state_sel = {k: v[order] for k, v in state_sel.items()}
    pred_obj = float(residual(state_sel).sum()) / T_parent
    print(f"Merged model ({'exact' if exact else 'surrogate'}, pre-refit) "
          f"obj/token={pred_obj:.4f} vs parent {got:.4f} "
          f"({pred_obj / got - 1:+.4%})", flush=True)
    # The 4.3 GiB of moments must go before the refit sweeps claim the GPU.
    state_sel = {k: v.clone() for k, v in state_sel.items()}
    del mom, state
    torch.cuda.empty_cache()

    with open(os.path.join(args.out, "merge_log.json"), "w") as f:
        json.dump({"parent": args.dir, "parent_k": K, "dim": d,
                   "criterion": args.criterion, "threshold": args.merge_threshold,
                   "target_k": args.target_k, "n_merges": n_merges, "k_final": K_new,
                   "r_total_parent": r_total, "parent_obj_per_token": got,
                   "surrogate_merged_obj_per_token": pred_obj,
                   "cumulative_rel_cost": cum, "inversions": inversions,
                   "label_map": lmap.tolist(), "steps": log}, f, indent=1)

    method = f"subspace_kmeans_merged_{args.criterion}"
    merge_meta = {"parent_dir": args.dir, "parent_k": K,
                  "parent_fingerprint": parent.get("sample_fingerprint"),
                  "criterion": args.criterion, "threshold": args.merge_threshold,
                  "target_k": args.target_k, "n_merges": n_merges,
                  "cumulative_rel_cost": cum, "refit_iters": args.refit_iters}

    if args.refit_iters > 0:
        if args.gpus < 1:
            raise SystemExit("--refit-iters > 0 needs a CUDA GPU (use --refit-iters 0)")
        devices = [f"cuda:{i}" for i in range(args.gpus)]
        model, history, final_obj, assign, sampled, ns = refit(args, cfg, state_sel,
                                                               K_new, d, devices)
        fp = sample_fingerprint(sampled, min(ns.tokens_per_file, N_CELLS), ns.seed)
        file_ids, cell_ids, labels = assign
    else:
        # No data pass: carry the parent's tokens over through the merge tree. Exact, since
        # counts == bincount(label) holds for the parent, so it holds for the mapped labels.
        a = torch.load(os.path.join(args.dir, "assignments.pt"), map_location="cpu",
                       weights_only=False)
        file_ids, cell_ids = a["file_id"], a["cell_id"]
        labels = lmap[a["label"].long()].to(torch.int32)
        model = {k: state_sel[k].cpu() for k in ("U", "means", "eigvals", "trace")}
        model["eigvals"] = model["eigvals"].float()
        model["trace"] = model["trace"].float()
        model["counts"] = torch.bincount(labels.long(), minlength=K_new)
        history, final_obj = [], pred_obj
        fp = parent.get("sample_fingerprint")
        sampled = parent.get("sampled_files")
        if not os.path.exists(os.path.join(args.out, "sample.json")):
            with open(os.path.join(args.dir, "sample.json")) as f_in, \
                 open(os.path.join(args.out, "sample.json"), "w") as f_out:
                f_out.write(f_in.read())

    # Parent config first, then this run's flags -- but never let an unset flag (src=None,
    # target_k=None, ...) overwrite the value the parent actually ran with.
    config = {**cfg, **{k: v for k, v in vars(args).items() if v is not None},
              "clusters": K_new, "dim": d, "method": method, "merge": merge_meta}
    save_model(args.out, U=model["U"], means=model["means"], eigvals=model["eigvals"],
               trace=model["trace"], counts=model["counts"], config=config,
               history=history, sampled_files=sampled, sample_fingerprint=fp,
               final_obj_per_token=final_obj)
    save_assignments(args.out, file_id=file_ids, cell_id=cell_ids, label=labels)

    assert int(model["counts"].sum()) == labels.numel(), "counts must sum to token count"
    assert torch.equal(model["counts"].long(), torch.bincount(labels.long(), minlength=K_new)), \
        "counts must equal bincount(label)"
    evr = (model["eigvals"].sum(1) / model["trace"].clamp(min=1e-12)) if d > 0 \
        else torch.zeros(K_new)
    print(f"Saved model.pt + assignments.pt to {args.out}/  (K={K_new})")
    print(f"Explained variance ratio (top-{d}): min={evr.min():.3f} "
          f"median={evr.median():.3f} max={evr.max():.3f}")
    print(f"Cluster sizes: min={int(model['counts'].min()):,} "
          f"median={int(model['counts'].median()):,} max={int(model['counts'].max()):,}")
    print(f"final_obj_per_token={final_obj:.4f} (parent {got:.4f}, "
          f"{final_obj / got - 1:+.4%})")


if __name__ == "__main__":
    main()
