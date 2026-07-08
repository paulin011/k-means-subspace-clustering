"""k-center clustering (Gonzalez farthest-point greedy) on weather-encoder tokens.

Point-cluster (d=0) counterpart to subspace_kmeans.py: no per-cluster subspace, just K
centroids chosen to minimise the *covering radius* (the max distance from any token to its
nearest centre) -- k-center's native minimax objective, as opposed to the mean squared
distance k-means/subspace_kmeans minimise.

Algorithm (Gonzalez, 2-approximation for the minimax objective):
  1. Pick a random token as the first centre.
  2. Maintain min_sq[t] = squared distance from token t to its nearest chosen centre.
  3. Repeat K-1 times: add the token with the largest min_sq (the current worst-covered
     point) as the next centre, then update min_sq.
  4. Final pass: assign every token to its nearest centre; compute per-cluster counts,
     trace (mean squared distance), and radius (max Euclidean distance = covering radius).

Sampling and all IO go through cluster_io.py so runs are directly comparable to
subspace_kmeans (same sample.json / fingerprint) and readable by analyze_clusters.py.

Full run:  nohup python3 kcenter.py --num-files 7000 -K 128 > kcenter_run.log 2>&1 &
"""

import argparse
import time

import torch

import cluster_io
from cluster_io import DIM


def _iter_chunks(T, chunk):
    for lo in range(0, T, chunk):
        yield lo, min(lo + chunk, T)


def kcenter(data, K, device, chunk, seed):
    """Gonzalez farthest-point greedy. `data` is CPU fp16 [T, DIM]; returns
    (centres [K, DIM] float32 on `device`, history list, chosen index list)."""
    T = data.shape[0]
    centres = torch.empty(K, DIM, dtype=torch.float32, device=device)
    min_sq = torch.full((T,), float("inf"), device=device)      # nearest-centre sq dist
    history, chosen = [], []

    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(T, (1,), generator=g).item())

    def add_centre(k, idx):
        c = data[idx].to(device).float()                        # [DIM]
        centres[k] = c
        chosen.append(idx)
        for lo, hi in _iter_chunks(T, chunk):
            x = data[lo:hi].to(device).float()                  # [chunk, DIM]
            d = torch.cdist(x, c[None]).squeeze(1) ** 2          # squared dist to new centre
            min_sq[lo:hi] = torch.minimum(min_sq[lo:hi], d)

    t0 = time.time()
    add_centre(0, first)
    for k in range(1, K):
        nxt = int(min_sq.argmax().item())                       # worst-covered token
        add_centre(k, nxt)
        cov_radius = float(min_sq.max().sqrt().item())          # current covering radius
        history.append({"centre": k, "covering_radius": cov_radius,
                        "obj_per_token": float(min_sq.mean().item())})
        if k % 16 == 0 or k == K - 1:
            print(f"  centre {k + 1}/{K} | covering radius {cov_radius:.2f} "
                  f"| {(time.time() - t0):.1f}s", flush=True)
    return centres, history, chosen


def assign(data, centres, device, chunk):
    """Nearest-centre assignment pass. Returns labels [T] int32 (CPU), counts [K],
    trace [K] (mean squared dist), radius [K] (max Euclidean dist), obj_per_token."""
    T, K = data.shape[0], centres.shape[0]
    labels = torch.empty(T, dtype=torch.int32)
    counts = torch.zeros(K, device=device)
    sum_sq = torch.zeros(K, device=device)                      # for trace
    max_d = torch.zeros(K, device=device)                       # for radius (Euclidean)

    for lo, hi in _iter_chunks(T, chunk):
        x = data[lo:hi].to(device).float()                      # [chunk, DIM]
        d = torch.cdist(x, centres)                             # [chunk, K] Euclidean
        dmin, lab = d.min(dim=1)                                # nearest centre
        labels[lo:hi] = lab.to(torch.int32).cpu()
        counts.scatter_add_(0, lab, torch.ones_like(dmin))
        sum_sq.scatter_add_(0, lab, dmin ** 2)
        max_d.scatter_reduce_(0, lab, dmin, reduce="amax", include_self=True)

    trace = sum_sq / counts.clamp(min=1)
    obj_per_token = float((sum_sq.sum() / T).item())
    return labels, counts.cpu(), trace.cpu(), max_d.cpu(), obj_per_token


def main():
    ap = argparse.ArgumentParser(description="k-center (Gonzalez) clustering")
    # --- sampling / IO args: must match what cluster_io.load_tokens() reads ---
    ap.add_argument("--src", default="/home/mremane/embeddings/latents_2",
                    help="directory with latent_{i}.pt files")
    ap.add_argument("--out", default="kcenter_out", help="output directory")
    ap.add_argument("--num-files", type=int, default=1500)
    ap.add_argument("--files-from", default=None,
                    help="reuse a previous run's sample.json / model.pt (for direct comparison)")
    ap.add_argument("--tokens-per-file", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-workers", type=int, default=16)
    ap.add_argument("--max-ram-gb", type=float, default=350.0)
    # --- algorithm args ---
    ap.add_argument("--clusters", "-K", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=262144, help="tokens per GPU chunk")
    ap.add_argument("--gpu", type=int, default=0, help="single GPU index to run on")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    data, file_ids, cell_ids, sampled = cluster_io.load_tokens(args)
    T = data.shape[0]
    fp = cluster_io.sample_fingerprint(sampled, min(args.tokens_per_file, cluster_io.N_CELLS),
                                       args.seed)
    K = args.clusters

    print(f"k-center: {T:,} tokens -> K={K} centres on {device}", flush=True)
    centres, history, _ = kcenter(data, K, device, args.chunk_size, args.seed)
    labels, counts, trace, radius, obj = assign(data, centres, device, args.chunk_size)
    print(f"done | covering radius {float(radius.max()):.2f} | obj/token {obj:.2f}", flush=True)

    # d=0 point clusters: empty subspace basis + eigvals (save_model sets evr=0 for d=0)
    U = torch.empty(K, DIM, 0)
    eigvals = torch.empty(K, 0)
    config = {**vars(args), "method": "kcenter"}

    cluster_io.save_model(args.out, U=U, means=centres.cpu(), eigvals=eigvals,
                          trace=trace, counts=counts, config=config, history=history,
                          sampled_files=sampled, sample_fingerprint=fp,
                          radius=radius, final_obj_per_token=obj)
    cluster_io.save_assignments(args.out, file_ids, cell_ids, labels)
    print(f"wrote {args.out}/model.pt + assignments.pt", flush=True)


if __name__ == "__main__":
    main()