"""Shared sampling/IO layer for all clustering algorithms (K-subspaces, k-means, k-center).

Every clustering script should sample tokens via `load_tokens()` and save its result via
`save_model()` / `save_assignments()`, so that:
  - runs from different algorithms on the same `sample.json` (`--files-from`) are directly
    comparable (same `sample_fingerprint`), and
  - `analyze_clusters.py` can read any algorithm's `model.pt` / `assignments.pt` uniformly.

model.pt contract (required keys, regardless of algorithm):
  U                  [K, DIM, d] orthonormal per-cluster basis (d=0 for point clusters,
                     i.e. plain k-means/k-center -- no subspace, just a centroid)
  means              [K, DIM]   per-cluster centroid
  eigvals            [K, d]     per-cluster basis eigenvalues, descending
  trace              [K]        per-cluster total variance E||x-mu||^2 (mean squared
                     distance to centroid -- what k-means/subspace_kmeans optimize)
  counts             [K]        cluster sizes
  explained_var_ratio [K]       eigvals.sum(1) / trace, 0 when d==0
  config             dict       run config; must include config["method"] in
                     {"kmeans", "kcenter", "subspace_kmeans"}
  history            list[dict] per-iteration convergence stats
  sampled_files      list[int]  latent file ids used
  sample_fingerprint str        see sample_fingerprint() below

Optional keys:
  radius             [K] max distance from centroid to any member -- k-center's native
                     (minimax) objective; absent for algorithms that don't optimize it.

assignments.pt contract (all algorithms):
  file_id, cell_id, label   each [T] int32 -- which sampled token, which cluster.
"""

import hashlib
import json
import os
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


def load_tokens(args):
    """Sample files, load them in parallel, return (data fp16 [T, DIM], file_ids, cell_ids, sampled).

    Expects `args` to have: src, out, num_files, files_from, tokens_per_file, seed,
    load_workers, max_ram_gb. Writes <out>/sample.json (reproducible manifest).
    """
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


def save_model(out_dir, *, U, means, eigvals, trace, counts, config, history,
               sampled_files, sample_fingerprint, radius=None):
    """Validate shapes and write model.pt under the shared cross-algorithm schema."""
    K, D, d = U.shape
    assert means.shape == (K, D), f"means {tuple(means.shape)} != ({K}, {D})"
    assert eigvals.shape == (K, d), f"eigvals {tuple(eigvals.shape)} != ({K}, {d})"
    assert trace.shape == (K,), f"trace {tuple(trace.shape)} != ({K},)"
    assert counts.shape == (K,), f"counts {tuple(counts.shape)} != ({K},)"
    assert "method" in config, "config must include config['method']"
    if radius is not None:
        assert radius.shape == (K,), f"radius {tuple(radius.shape)} != ({K},)"

    evr = eigvals.sum(1) / trace.clamp(min=1e-12) if d > 0 else torch.zeros(K)
    obj = {"U": U, "means": means, "eigvals": eigvals, "trace": trace, "counts": counts,
           "explained_var_ratio": evr, "config": config, "history": history,
           "sampled_files": sampled_files, "sample_fingerprint": sample_fingerprint}
    if radius is not None:
        obj["radius"] = radius
    os.makedirs(out_dir, exist_ok=True)
    torch.save(obj, os.path.join(out_dir, "model.pt"))


def save_assignments(out_dir, file_id, cell_id, label):
    """Write assignments.pt under the shared cross-algorithm schema."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"file_id": file_id, "cell_id": cell_id, "label": label},
               os.path.join(out_dir, "assignments.pt"))
