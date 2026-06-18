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

Outputs in --out:
  model.pt        U [K, 2048, d], means [K, 2048], eigvals [K, d], trace [K],
                  counts [K], config, per-iteration history, sampled file list
  assignments.pt  file_id / cell_id / label (int32) for every sampled token
  sample.json     reproducible manifest: fingerprint, seed, tokens_per_file, and
                  the sorted list of sampled file ids -- written before clustering

Reproducibility / direct comparison: to cluster a *different* configuration on the
exact same tokens (e.g. vary K or d but hold the data fixed), pass the previous run's
sample with --files-from <dir>/sample.json (or <dir>/model.pt) plus the same --seed and
--tokens-per-file. Runs sharing a sample fingerprint are directly comparable.

Full run:  nohup python3 subspace_kmeans.py > subspace_run.log 2>&1 &
"""

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

N_FILES_TOTAL = 13021
N_CELLS = 12288
DIM = 2048


def sample_fingerprint(files, tokens_per_file, seed):
    """Stable short hash identifying the exact token sample (for run comparison)."""
    h = hashlib.sha1()
    h.update(f"{tokens_per_file}|{seed}|".encode())
    h.update(",".join(map(str, sorted(files))).encode())
    return h.hexdigest()[:12]


def load_file_list(path):
    """Read a sampled-file list from a previous run's sample.json or model.pt."""
    if path.endswith(".json"):
        with open(path) as f:
            return list(json.load(f)["files"])
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return list(obj["sampled_files"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="latents_2", help="directory with latent_{i}.pt files")
    p.add_argument("--out", default="subspace_out", help="output directory")
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
    p.add_argument("--chunk-size", type=int, default=262144, help="tokens per GPU chunk")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpus", type=int, default=min(2, torch.cuda.device_count()))
    p.add_argument("--load-workers", type=int, default=16, help="parallel file-loading threads")
    p.add_argument("--max-ram-gb", type=float, default=350.0, help="abort if sampled tokens exceed this")
    return p.parse_args()


def load_tokens(args):
    """Sample files, load them in parallel, return (data fp16 [T, DIM], file_ids, cell_ids)."""
    tpf = min(args.tokens_per_file, N_CELLS)
    if args.files_from:
        sampled = load_file_list(args.files_from)
        print(f"Reusing {len(sampled)} files from {args.files_from} "
              f"(--num-files ignored)", flush=True)
    else:
        g = torch.Generator().manual_seed(args.seed)
        n_files = min(args.num_files, N_FILES_TOTAL)
        sampled = torch.randperm(N_FILES_TOTAL, generator=g)[:n_files].tolist()
    n_files = len(sampled)

    fp = sample_fingerprint(sampled, tpf, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sample.json"), "w") as f:
        # files kept in load order so --files-from reproduces the run exactly
        # (init seeds depend on order); the fingerprint sorts internally.
        json.dump({"fingerprint": fp, "num_files": n_files, "tokens_per_file": tpf,
                   "seed": args.seed, "src": args.src, "files": list(sampled)}, f)

    T = n_files * tpf
    ram_gb = T * DIM * 2 / 1e9
    print(f"Sampling {n_files} files x {tpf} tokens = {T:,} tokens "
          f"({ram_gb:.1f} GB fp16 in RAM) | sample fingerprint {fp}", flush=True)
    if ram_gb > args.max_ram_gb:
        raise SystemExit(f"Would need {ram_gb:.0f} GB > --max-ram-gb={args.max_ram_gb}; "
                         f"reduce --num-files or --tokens-per-file.")

    data = torch.empty(T, DIM, dtype=torch.float16)
    file_ids = torch.empty(T, dtype=torch.int32)
    cell_ids = torch.empty(T, dtype=torch.int32)

    def load_one(pos, fid):
        d = torch.load(os.path.join(args.src, f"latent_{fid}.pt"),
                       map_location="cpu", weights_only=False)
        lat = d["latent"]
        if lat.dim() == 3:
            lat = lat.squeeze(0)
        if lat.shape != (N_CELLS, DIM):
            raise ValueError(f"latent_{fid}.pt has shape {tuple(lat.shape)}")
        if tpf < N_CELLS:
            gg = torch.Generator().manual_seed(args.seed * 1_000_003 + fid)
            rows = torch.randperm(N_CELLS, generator=gg)[:tpf]
            lat = lat[rows]
        else:
            rows = torch.arange(N_CELLS)
        o = pos * tpf
        data[o:o + tpf] = lat.to(torch.float16)
        cell_ids[o:o + tpf] = rows.to(torch.int32)
        file_ids[o:o + tpf] = fid

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.load_workers) as ex:
        futs = [ex.submit(load_one, pos, fid) for pos, fid in enumerate(sampled)]
        for n, f in enumerate(as_completed(futs), 1):
            f.result()
            if n % 100 == 0 or n == n_files:
                el = time.time() - t0
                print(f"  loaded {n}/{n_files} files ({n / el:.1f} files/s)", flush=True)
    return data, file_ids, cell_ids, sampled


def sweep_worker(rank, device, data, labels, chunks, model, K, d, affine,
                 accumulate, results):
    """Process this GPU's chunks: assign tokens, optionally accumulate moments."""
    torch.cuda.set_device(device)
    U = model["U"].to(device)                                       # [K, DIM, d]
    U_cat = U.permute(1, 0, 2).reshape(DIM, K * d).contiguous()     # [DIM, K*d]
    if affine:
        means = model["means"].to(device)                           # [K, DIM]
        c = torch.bmm(U.transpose(1, 2), means.unsqueeze(-1)).squeeze(-1)  # [K, d]
        mnorm = (means * means).sum(1)                              # [K]
        cnorm = (c * c).sum(1)
    if accumulate:
        S = torch.zeros(K, DIM, DIM, device=device)
        msum = torch.zeros(K, DIM, device=device)
        cnt = torch.zeros(K, device=device)
    obj, changed = 0.0, 0

    for i0, i1 in chunks:
        X = data[i0:i1].to(device, non_blocking=True).float()
        B = X.shape[0]
        xnorm = (X * X).sum(1, keepdim=True)                        # [B, 1]
        P = (X @ U_cat).view(B, K, d)
        pe = (P * P).sum(-1)                                        # [B, K]
        if affine:
            xm = X @ means.T                                        # [B, K]
            pc = torch.einsum("bkd,kd->bk", P, c)
            R = xnorm - 2 * xm + mnorm - (pe - 2 * pc + cnorm)
        else:
            R = xnorm - pe
        vals, a = R.min(1)
        obj += vals.clamp_min(0).double().sum().item()
        a_cpu = a.to("cpu", torch.int32)
        changed += (a_cpu != labels[i0:i1]).sum().item()
        labels[i0:i1] = a_cpu
        if accumulate:
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
        "obj": obj, "changed": changed,
    }


def run_sweep(devices, data, labels, model, K, d, affine, accumulate, chunk_size):
    T = data.shape[0]
    chunks = [(i0, min(i0 + chunk_size, T)) for i0 in range(0, T, chunk_size)]
    results = {}
    threads = [
        threading.Thread(target=sweep_worker,
                         args=(r, dev, data, labels, chunks[r::len(devices)], model,
                               K, d, affine, accumulate, results))
        for r, dev in enumerate(devices)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    obj = sum(r["obj"] for r in results.values())
    changed = sum(r["changed"] for r in results.values())
    if not accumulate:
        return None, obj, changed
    S = sum(r["S"] for r in results.values())
    msum = sum(r["msum"] for r in results.values())
    cnt = sum(r["cnt"] for r in results.values())
    return (S, msum, cnt), obj, changed


def update_model(moments, data, K, d, affine, seed_gen, device):
    """New bases = top-d eigenvectors of each cluster covariance (exact PCA)."""
    S, msum, cnt = (m.to(device) for m in moments)  # CPU-merged -> H2D (P2P unusable)
    n = cnt.clamp(min=1.0)
    means = msum / n[:, None] if affine else torch.zeros(K, DIM, device=device)
    C = S / n[:, None, None]
    if affine:
        C = C - means.unsqueeze(-1) * means.unsqueeze(1)
    trace = C.diagonal(dim1=-2, dim2=-1).sum(-1)
    evals, evecs = torch.linalg.eigh(C)                  # ascending
    U = evecs[..., -d:].flip(-1).contiguous()            # [K, DIM, d], descending
    eigvals = evals[..., -d:].flip(-1).clamp_min(0)

    # Re-seed clusters too small to support a d-dim basis: mean = random token,
    # zero basis, so the next assignment captures that token's neighbourhood.
    min_count = max(2 * d, 64)
    for j in (cnt < min_count).nonzero().flatten().tolist():
        print(f"  re-seeding cluster {j} (size {int(cnt[j])})", flush=True)
        t = int(torch.randint(data.shape[0], (1,), generator=seed_gen))
        if affine:
            means[j] = data[t].to(device).float()
            U[j] = 0
        else:
            U[j] = torch.linalg.qr(torch.randn(DIM, d, generator=seed_gen).to(device))[0]
        eigvals[j] = 0
    return {"U": U.cpu(), "means": means.cpu(), "eigvals": eigvals.cpu(),
            "trace": trace.cpu(), "counts": cnt.to("cpu", torch.int64)}


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

    data, file_ids, cell_ids, sampled = load_tokens(args)
    T = data.shape[0]
    g = torch.Generator().manual_seed(args.seed + 1)
    labels = torch.full((T,), -1, dtype=torch.int32)

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
        moments, obj, changed = run_sweep(
            devices, data, labels, model, K, d, affine,
            accumulate=True, chunk_size=args.chunk_size)
        model = update_model(moments, data, K, d, affine, g, devices[0])
        dt = time.time() - t0
        frac = changed / T
        sizes = model["counts"]
        history.append({"iter": it, "obj_per_token": obj / T, "frac_changed": frac,
                        "size_min": int(sizes.min()), "size_max": int(sizes.max())})
        print(f"iter {it:2d}: obj/token={obj / T:.4f}  changed={frac:.4%}  "
              f"sizes[min/med/max]={int(sizes.min())}/{int(sizes.median())}/{int(sizes.max())}  "
              f"({dt:.1f}s)", flush=True)
        if frac < args.tol:
            print("Converged.", flush=True)
            break

    # Final labeling under the final bases.
    _, obj, changed = run_sweep(devices, data, labels, model, K, d, affine,
                                accumulate=False, chunk_size=args.chunk_size)
    print(f"final: obj/token={obj / T:.4f}  changed={changed / T:.4%}", flush=True)

    evr = model["eigvals"].sum(1) / model["trace"].clamp(min=1e-12)
    fp = sample_fingerprint(sampled, min(args.tokens_per_file, N_CELLS), args.seed)
    torch.save({**model, "explained_var_ratio": evr, "config": vars(args),
                "history": history, "sampled_files": sampled, "sample_fingerprint": fp},
               os.path.join(args.out, "model.pt"))
    torch.save({"file_id": file_ids, "cell_id": cell_ids, "label": labels},
               os.path.join(args.out, "assignments.pt"))
    print(f"Saved model.pt + assignments.pt to {args.out}/")
    print(f"Explained variance ratio (top-{d}): min={evr.min():.3f} "
          f"median={evr.median():.3f} max={evr.max():.3f}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
