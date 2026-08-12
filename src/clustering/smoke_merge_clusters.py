#!/usr/bin/env python3
"""Smoke tests for merge_clusters.py (docs/ideas/overcluster_merge.md §4).

S1 synthetic split/merge  an artificially split cluster must be the cascade's FIRST merge,
                          at a cost orders of magnitude below any genuine pair
S2 algebra                merged n / mu / tr(C) from the update formulas vs brute force
S3 dR accuracy            both scoring modes vs dR computed brute-force from the member
                          tokens: the exact (moments) path to a tight tolerance, the PPCA
                          surrogate fallback on pair RANKING only
S4 schema                 merged output satisfies the cluster_io model.pt contract
S5 degenerate cuts        --target-k K is a no-op; the cascade runs down to min-k
S6 end-to-end             (separate, real data) merge_clusters.py on a mini run

S1/S2/S4/S5 are data-free. S3 uses real tokens (--src/--files, a few latent files) and falls
back to synthetic data with --synthetic-only.

  python3 src/clustering/smoke_merge_clusters.py
  python3 src/clustering/smoke_merge_clusters.py --synthetic-only
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.cluster_io import DIM, N_CELLS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_clusters as mc
from merge_clusters import (cut_index, label_map, residual, run_cascade, score_pairs)

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- helpers


def fit_state(X, labels, K, d, device):
    """Exact per-cluster PCA -> the same fields subspace_kmeans writes to model.pt."""
    U = torch.zeros(K, DIM, d, device=device)
    means = torch.zeros(K, DIM, device=device)
    eigvals = torch.zeros(K, d, device=device)
    trace = torch.zeros(K, dtype=torch.float64, device=device)
    counts = torch.zeros(K, dtype=torch.float64, device=device)
    for j in range(K):
        Xj = X[labels == j].double()
        n = Xj.shape[0]
        mu = Xj.mean(0)
        Xc = Xj - mu
        C = (Xc.T @ Xc) / n
        ev, evec = torch.linalg.eigh(C)
        means[j] = mu.float()
        trace[j] = C.diagonal().sum()
        counts[j] = n
        if d > 0:
            U[j] = evec[:, DIM - d:].flip(-1).float()
            eigvals[j] = ev[DIM - d:].flip(-1).clamp_min(0).float()
    sigma2 = ((trace - eigvals.double().sum(1)).clamp_min(0)
              / max(DIM - d, 1)).clamp_min(mc.sk.SIGMA2_FLOOR)
    return {"U": U, "means": means, "eigvals": eigvals, "trace": trace,
            "counts": counts, "sigma2": sigma2}


def make_moments(X, labels, K, device):
    """The per-cluster second moments subspace_kmeans --save-moments writes."""
    S = torch.zeros(K, DIM, DIM, device=device)
    msum = torch.zeros(K, DIM, device=device)
    cnt = torch.zeros(K, device=device)
    for j in range(K):
        Xj = X[labels == j]
        S[j] = Xj.T @ Xj
        msum[j] = Xj.sum(0)
        cnt[j] = Xj.shape[0]
    return {"S": S, "msum": msum, "cnt": cnt}


def exact_residual(X, d):
    """R = n (tr(C) - sum_{i<=d} lam_i) computed from the tokens themselves."""
    Xd = X.double()
    n = Xd.shape[0]
    Xc = Xd - Xd.mean(0)
    C = (Xc.T @ Xc) / n
    tr = C.diagonal().sum()
    if d == 0:
        return float(n * tr)
    ev = torch.linalg.eigvalsh(C)
    return float(n * (tr - ev[DIM - d:].sum()))


def make_subspace_data(K, n_per, d_true, noise, device, gen, spread=6.0):
    """K clusters, each on its own random d_true-dim affine subspace."""
    X, y = [], []
    for j in range(K):
        B = torch.linalg.qr(torch.randn(DIM, d_true, generator=gen, device=device))[0]
        centre = torch.randn(DIM, generator=gen, device=device) * spread
        coef = torch.randn(n_per, d_true, generator=gen, device=device) * 3.0
        X.append(coef @ B.T + centre + torch.randn(n_per, DIM, generator=gen,
                                                   device=device) * noise)
        y.append(torch.full((n_per,), j, dtype=torch.int64, device=device))
    return torch.cat(X), torch.cat(y)


# ---------------------------------------------------------------- tests


def s1_split_merge(device):
    print("S1 synthetic split/merge")
    g = torch.Generator(device=device).manual_seed(0)
    d = 4
    X, y = make_subspace_data(4, 800, d, 0.6, device, g)
    # Artificially split cluster 0 into two halves -> ids 0 and 4.
    half = (y == 0).nonzero().flatten()[:400]
    y[half] = 4
    K = 5
    state = fit_state(X, y, K, d, device)
    mom = make_moments(X, y, K, device)
    r_total = float(residual(state).sum())
    log = run_cascade(dict(state), K, d, device, "residual", 8, 1, r_total, mom=mom)

    first = log[0]
    pair = {first["into"], first["from"]}
    check("first merge is the artificial split", pair == {0, 4}, f"merged {pair}")
    # A free split still costs a little: each half fits its own d-dim PCA to half the
    # points, so it overfits its own noise, and merging gives that advantage back. The
    # signal is the ~10x gap to the first genuine merge, not an absolute zero.
    check("split costs far less than the next merge",
          log[1]["rel_cost"] > 5 * max(first["rel_cost"], 1e-12),
          f"{first['rel_cost']:.3e} vs {log[1]['rel_cost']:.3e}")
    check("costs are non-negative", all(e["delta_r"] >= 0 for e in log))

    # The recovered 4-cluster partition must match the generative labels exactly.
    lmap, K_new = label_map(log, 1, K)
    y_true = y.cpu().clone()
    y_true[y_true == 4] = 0
    check("recovers the generative partition",
          K_new == 4 and len(set(zip(lmap[y.cpu()].tolist(), y_true.tolist()))) == 4)


def s2_algebra(device):
    print("S2 merged-moment algebra")
    g = torch.Generator(device=device).manual_seed(1)
    X, y = make_subspace_data(2, 500, 3, 1.0, device, g)
    d = 3
    state = fit_state(X, y, 2, d, device)
    na, nb = state["counts"][0], state["counts"][1]
    n = na + nb
    mu = (na.float() * state["means"][0] + nb.float() * state["means"][1]) / n.float()
    delta = (state["means"][0] - state["means"][1]).double()
    tr = ((na * state["trace"][0] + nb * state["trace"][1]) / n
          + (na * nb / (n * n)) * delta.pow(2).sum())

    Xd = X.double()
    check("merged n", int(n) == Xd.shape[0])
    check("merged mean", torch.allclose(mu.double(), Xd.mean(0), atol=1e-6),
          f"max|d|={float((mu.double() - Xd.mean(0)).abs().max()):.2e}")
    Xc = Xd - Xd.mean(0)
    tr_brute = ((Xc * Xc).sum() / Xd.shape[0])
    check("merged trace", abs(float(tr) - float(tr_brute)) / float(tr_brute) < 1e-9,
          f"{float(tr):.6f} vs {float(tr_brute):.6f}")


def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return float(((ra - ra.mean()) * (rb - rb.mean())).sum()
                 / (ra.std(unbiased=False) * rb.std(unbiased=False) * len(ra)))


def s3_accuracy(X, y, K, d, device, label, rel_tol=0.01, rho_tol=0.98):
    """Both scoring modes against dR computed brute-force from the member tokens."""
    print(f"S3 dR accuracy ({label}, K={K}, d={d})")
    state = fit_state(X, y, K, d, device)
    mom = make_moments(X, y, K, device)
    pairs = torch.combinations(torch.arange(K, device=device), 2)

    truth = []
    for a, b in pairs.tolist():
        Xa, Xb = X[y == a], X[y == b]
        truth.append(exact_residual(torch.cat([Xa, Xb]), d)
                     - exact_residual(Xa, d) - exact_residual(Xb, d))
    truth = torch.tensor(truth, dtype=torch.float64, device=device)
    check("brute-force dR >= 0", bool((truth >= -1e-6).all()))

    # --- the primary path: exact, from second moments
    ex, _, _ = score_pairs(state, pairs, d, device, batch=8, mom=mom)
    rel = (ex - truth).abs() / truth.abs().clamp_min(1e-9)
    check(f"EXACT: median relative error < {rel_tol:.2%}", float(rel.median()) < rel_tol,
          f"median {float(rel.median()):.4%}, max {float(rel.max()):.4%}")
    check("EXACT: cheapest pair agrees", int(ex.argmin()) == int(truth.argmin()),
          f"got {pairs[int(ex.argmin())].tolist()} want {pairs[int(truth.argmin())].tolist()}")
    check(f"EXACT: ranking preserved (Spearman > {rho_tol})", spearman(ex, truth) > rho_tol,
          f"rho={spearman(ex, truth):.5f}")

    # More power iterations must not move the answer -- that is the convergence check.
    ex8, _, _ = score_pairs(state, pairs, d, device, batch=8, mom=mom, power_iters=8)
    drift = float(((ex8 - ex).abs() / ex8.abs().clamp_min(1e-9)).max())
    check("EXACT: converged at the default power-iters", drift < 0.01,
          f"max drift vs 8 iters {drift:.4%}")

    # --- the fallback path: PPCA surrogate. Ranking is all it is trusted for.
    ap, _, _ = score_pairs(state, pairs, d, device, batch=64)
    rel_s = (ap - truth).abs() / truth.abs().clamp_min(1e-9)
    rho_s = spearman(ap, truth)
    check("SURROGATE: ranking usable (Spearman > 0.80)", rho_s > 0.80, f"rho={rho_s:.4f}")
    print(f"    (surrogate median rel err {float(rel_s.median()):.1%} -- informational; "
          f"this is why moments.pt is the default path)")


def s4_schema(device):
    print("S4 model.pt schema of a merged model")
    from common.cluster_io import save_model
    import tempfile
    g = torch.Generator(device=device).manual_seed(3)
    d, K = 4, 6
    X, y = make_subspace_data(K, 400, d, 0.8, device, g)
    state = fit_state(X, y, K, d, device)
    r_total = float(residual(state).sum())
    log = run_cascade(dict(state), K, d, device, "residual", 64, 1, r_total)
    n = cut_index(log, "residual", 1e9, 3, K)
    lmap, K_new = label_map(log, n, K)

    st = dict(fit_state(X, y, K, d, device))
    alive = list(range(K))
    for e in log[:n]:
        mc.apply_merge(st, e["into"], e["from"], d, device)
        alive.remove(e["from"])
    sel = torch.tensor(alive, device=device)
    st = {k: v[sel] for k, v in st.items()}
    check("cut reaches target K", K_new == 3, f"K_new={K_new}")
    U = st["U"].cpu()
    gram = U.transpose(1, 2) @ U
    eye = torch.eye(d).expand(K_new, d, d)
    check("merged bases orthonormal", float((gram - eye).abs().max()) < 1e-4,
          f"max|UtU-I|={float((gram - eye).abs().max()):.2e}")
    evr = st["eigvals"].double().sum(1) / st["trace"].clamp_min(1e-12)
    check("explained variance ratio in [0,1]",
          bool(((evr >= 0) & (evr <= 1)).all()), f"range {float(evr.min()):.3f}..{float(evr.max()):.3f}")
    with tempfile.TemporaryDirectory() as tmp:
        save_model(tmp, U=U, means=st["means"].cpu(), eigvals=st["eigvals"].cpu().float(),
                   trace=st["trace"].cpu().float(),
                   counts=st["counts"].cpu().long(), config={"method": "subspace_kmeans_merged"},
                   history=[], sampled_files=[0], sample_fingerprint="test",
                   final_obj_per_token=1.0)
        check("save_model accepts the merged model",
              os.path.exists(os.path.join(tmp, "model.pt")))


def s5_degenerate(device):
    print("S5 degenerate cuts")
    g = torch.Generator(device=device).manual_seed(4)
    d, K = 4, 6
    X, y = make_subspace_data(K, 300, d, 0.8, device, g)
    state = fit_state(X, y, K, d, device)
    r_total = float(residual(state).sum())
    log = run_cascade(dict(state), K, d, device, "residual", 64, 2, r_total)
    check("cascade runs down to min-k", len(log) == K - 2, f"{len(log)} merges")

    n0 = cut_index(log, "residual", 1e9, K, K)
    lmap0, k0 = label_map(log, n0, K)
    check("--target-k K is a no-op", n0 == 0 and k0 == K and lmap0.tolist() == list(range(K)))
    n1 = cut_index(log, "residual", 0.0, None, K)
    check("threshold 0 merges nothing", n1 == 0, f"{n1} merges")
    nz = cut_index(log, "residual", 1e9, 2, K)
    _, kz = label_map(log, nz, K)
    check("--target-k 2 terminates", kz == 2, f"K_new={kz}")
    # Inversion guard: the cut must stop at the FIRST violation, never past it.
    tau = log[0]["rel_cost"] * 1.000001
    nn = cut_index(log, "residual", tau, None, K)
    check("cut stops at first violation",
          all(e["rel_cost"] <= tau for e in log[:nn]), f"{nn} merges at tau={tau:.3e}")

    # --target-k must win over a threshold that would otherwise cut earlier: asking for an
    # exact K and silently getting another is the footgun this precedence removes.
    n_t = cut_index(log, "residual", 0.0, 3, K, max_total=0.0)
    _, k_t = label_map(log, n_t, K)
    check("--target-k overrides both budgets", k_t == 3, f"K_new={k_t} (want 3)")

    # Cumulative cap fires where the per-merge cap never would.
    cum = sum(e["rel_cost"] for e in log)
    n_c = cut_index(log, "residual", 1e9, None, K, max_total=cum / 2)
    tot = sum(e["rel_cost"] for e in log[:n_c])
    check("cumulative cap bounds total cost", n_c < len(log) and tot <= cum / 2,
          f"{n_c}/{len(log)} merges, total {tot:.4f} <= {cum / 2:.4f}")
    check("per-merge cap alone would not have cut",
          cut_index(log, "residual", 1e9, None, K, max_total=None) == len(log))


def load_real(src, n_files, d_tokens, device, gen):
    """A few real latent files, subsampled -- for S3 on genuine token geometry."""
    import glob
    paths = sorted(glob.glob(os.path.join(src, "latent_*.pt")))[:n_files]
    if not paths:
        return None, None
    X = []
    for p in paths:
        lat = torch.load(p, map_location="cpu", weights_only=False)["latent"]
        if lat.dim() == 3:
            lat = lat.squeeze(0)
        rows = torch.randperm(N_CELLS, generator=gen)[:d_tokens]
        X.append(lat[rows].float())
    return torch.cat(X).to(device), paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="latents_2")
    p.add_argument("--files", type=int, default=4)
    p.add_argument("--tokens-per-file", type=int, default=3000)
    p.add_argument("--synthetic-only", action="store_true")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device} DIM={DIM}\n")

    s1_split_merge(device)
    s2_algebra(device)

    g = torch.Generator(device=device).manual_seed(2)
    X, y = make_subspace_data(6, 600, 6, 1.5, device, g)
    s3_accuracy(X, y, 6, 6, device, "synthetic (near-degenerate worst case)",
                rel_tol=0.01, rho_tol=0.98)

    if not args.synthetic_only:
        gen = torch.Generator().manual_seed(7)
        X, paths = load_real(args.src, args.files, args.tokens_per_file, device, gen)
        if X is None:
            print(f"  (skipped real-token S3: no latent_*.pt in {args.src})")
        else:
            # Cheap k-means-ish labels: assign to the nearest of K random tokens, so the
            # clusters are real token groups rather than an arbitrary partition.
            K, d = 8, 16
            gg = torch.Generator(device=device).manual_seed(11)
            c = X[torch.randperm(X.shape[0], generator=gg, device=device)[:K]]
            for _ in range(8):
                lab = torch.cdist(X, c).argmin(1)
                for j in range(K):
                    if (lab == j).any():
                        c[j] = X[lab == j].mean(0)
            print(f"  ({X.shape[0]:,} real tokens from {len(paths)} files, "
                  f"sizes {torch.bincount(lab, minlength=K).tolist()})")
            s3_accuracy(X, lab, K, d, device, "real tokens", rel_tol=0.001, rho_tol=0.999)

    s4_schema(device)
    s5_degenerate(device)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
