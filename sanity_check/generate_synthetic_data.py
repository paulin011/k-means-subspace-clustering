#!/usr/bin/env python3
"""Generate synthetic latent files with a known ground-truth cluster structure.

Unlike the real weather-encoder latents, here we know the true answer: every token
is drawn from one of `--k-true` well-separated Gaussian blobs in the 2048-dim token
space. That makes it possible to check kmeans/kcenter for *correctness* (do they
recover the planted clusters?), not just that they run without crashing (a smoke test).

Writes files in the exact schema `cluster_io.load_tokens()` expects, so the existing
`subspace_kmeans.py --dim 0` / `kcenter.py` can cluster this data unmodified via
`--files-from <out>/sample.json --src <out>`:
  - `latent_{i}.pt`: `{"idx": i, "latent": float32 [1, 12288, 2048]}`
  - `sample.json`: `{"files": [...]}` manifest for `--files-from`
  - `ground_truth.pt`: `file_id` / `cell_id` / `true_label` (int32) for every token
    written, in the same (file_id, cell_id) scheme `assignments.pt` uses -- so
    `check_recovery.py` can join a run's recovered labels back to the truth.
  - `true_centers.pt`: the planted [k_true, 2048] centers.

Usage:
  python3 sanity_check/generate_synthetic_data.py --out sanity_check/data
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from cluster_io import DIM, N_CELLS, sample_fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sanity_check/data", help="output directory")
    ap.add_argument("--num-files", type=int, default=4,
                    help="synthetic latent_{i}.pt files to write (each holds all 12288 cells)")
    ap.add_argument("--tokens-per-file", type=int, default=N_CELLS,
                    help="passed through into sample.json; the algorithms subsample at load time")
    ap.add_argument("--k-true", type=int, default=6, help="number of planted ground-truth clusters")
    ap.add_argument("--center-scale", type=float, default=5.0,
                    help="stddev of the planted cluster centers (bigger => more separated)")
    ap.add_argument("--noise-std", type=float, default=1.0,
                    help="per-token Gaussian noise stddev around its cluster center")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    centers = torch.randn(args.k_true, DIM, generator=g) * args.center_scale

    os.makedirs(args.out, exist_ok=True)
    all_file_id, all_cell_id, all_true_label = [], [], []
    for fid in range(args.num_files):
        labels = torch.randint(args.k_true, (N_CELLS,), generator=g)
        noise = torch.randn(N_CELLS, DIM, generator=g) * args.noise_std
        latent = (centers[labels] + noise).unsqueeze(0)          # [1, N_CELLS, DIM]
        torch.save({"idx": torch.tensor(fid), "latent": latent},
                   os.path.join(args.out, f"latent_{fid}.pt"))
        all_file_id.append(torch.full((N_CELLS,), fid, dtype=torch.int32))
        all_cell_id.append(torch.arange(N_CELLS, dtype=torch.int32))
        all_true_label.append(labels.to(torch.int32))
        print(f"  wrote latent_{fid}.pt ({N_CELLS} cells, true labels "
              f"{torch.bincount(labels, minlength=args.k_true).tolist()})", flush=True)

    torch.save({"file_id": torch.cat(all_file_id), "cell_id": torch.cat(all_cell_id),
               "true_label": torch.cat(all_true_label)},
               os.path.join(args.out, "ground_truth.pt"))
    torch.save({"centers": centers}, os.path.join(args.out, "true_centers.pt"))

    files = list(range(args.num_files))
    fp = sample_fingerprint(files, args.tokens_per_file, args.seed)
    with open(os.path.join(args.out, "sample.json"), "w") as f:
        json.dump({"fingerprint": fp, "num_files": args.num_files,
                   "tokens_per_file": args.tokens_per_file, "seed": args.seed,
                   "src": args.out, "files": files}, f)

    iu = torch.triu_indices(args.k_true, args.k_true, offset=1)
    center_dists = (centers[iu[0]] - centers[iu[1]]).norm(dim=1)
    print(f"\n{args.num_files * N_CELLS:,} tokens, K_true={args.k_true}, "
          f"center-to-center dist min/mean {float(center_dists.min()):.1f}/"
          f"{float(center_dists.mean()):.1f} vs. per-token noise scale "
          f"~{args.noise_std * DIM ** 0.5:.1f} -- the bigger that ratio, the more a "
          f"correct clustering run should recover these exactly (ARI/NMI near 1).")
    print(f"Wrote {args.out}/ (latent_{{0..{args.num_files - 1}}}.pt, sample.json, "
          f"ground_truth.pt, true_centers.pt)")

    print(f"\nCluster it, e.g.:\n"
          f"  python3 subspace_kmeans.py --files-from {args.out}/sample.json --dim 0 "
          f"--clusters {args.k_true} --iters 20 --src {args.out} --out sanity_check/kmeans_out\n"
          f"  python3 kcenter.py --files-from {args.out}/sample.json "
          f"--clusters {args.k_true} --src {args.out} --out sanity_check/kcenter_out\n"
          f"then check recovery:\n"
          f"  python3 sanity_check/check_recovery.py --dir sanity_check/kmeans_out\n"
          f"  python3 sanity_check/check_recovery.py --dir sanity_check/kcenter_out")


if __name__ == "__main__":
    main()
