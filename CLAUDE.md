# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Analysis of token embeddings produced by a weather encoder trained on the ERA5 dataset. Current goal: implement a **k-means-based subspace clustering** algorithm to understand the structure of the tokens. Requirements:

- Must finish in **≤ ~6 hours** wall time on this machine.
- Must output the **subspace bases** (per-cluster basis vectors), not just cluster assignments.
- Token selection: planned approach is a parameter `k` that samples tokens uniformly at random across **all** latent files (not per-file).
- Open design decisions: how many SVD/power iterations to use for the per-cluster subspace estimation, or whether full PCA per cluster is preferable (e.g. `torch.pca_lowrank` / `torch.svd_lowrank` on GPU).

## Data

`latents_2/` — 1.2 TB, **13,021 files** `latent_{i}.pt` for `i = 0 … 13020` (complete, no gaps). Ignore `latents_2/scp_transfer.log`.

Each file is a pickled dict (load with `torch.load(path, map_location="cpu", weights_only=False)`):

| key | content |
|---|---|
| `idx` | scalar int64 tensor (sample index) |
| `latent` | float32 tensor of shape **`[1, 12288, 2048]`** |

Important details:

- The latent has a **leading batch dim of 1** — squeeze it; the real shape is `[12288, 2048]`.
- `12288` = HEALPix cells (nside=32 grid, 12·32²), one token per cell. `2048` = embedding dim per token.
- The cell index uses **NESTED HEALPix ordering** (confirmed 2026-06-12: the dominant-cluster world map is geographically coherent under NESTED, noise under RING — see `subspace_out/dominant_cluster_map.png`). `analyze_subspaces.py` contains pure-numpy `healpix_nest2ring` / `healpix_ring_lonlat` helpers for cell→lat/lon.
- One file ≈ 100 MB float32. The full dataset (1.2 TB) does **not** fit in RAM (512 GB), so any full-data pass must stream files. Total token count is 13,021 × 12,288 ≈ 160 M tokens.
- Sampling tokens "randomly across all latents" still requires opening every touched file (~100 MB read per file), so I/O dominates — group sampled token indices by file and read each file at most once.

## Hardware / Environment

- 48-core CPU, 512 GB RAM, 2× NVIDIA A100 40 GB.
- Python: `/usr/bin/python3` (no venv), PyTorch 2.6.0 with CUDA, both GPUs visible.
- No build/lint/test tooling exists; these are standalone scripts run directly: `python3 script.py` (long runs via `nohup`).
- **GPU peer-to-peer is silently broken on this machine**: `tensor.to()` directly between `cuda:0` and `cuda:1` returns zeros/garbage without any error (`can_device_access_peer` still reports True). Always route inter-GPU transfers through CPU (`.cpu()` then `.to(other_gpu)`), and do D2H copies from the thread that launched the producing kernels.

## Code

See **README.md** for full documentation. Always keep README.md and this section up to date when changing code (user requirement: every script must be documented).

- `subspace_kmeans.py` — K-subspaces clustering (assignment by orthogonal residual to affine subspaces; per-cluster bases from exact PCA via batched `eigh` of streaming-accumulated 2048×2048 covariances; both GPUs used via one thread per GPU, accumulators merged on CPU). Defaults: 1500 random files × all 12288 tokens (~75 GB RAM), K=64 clusters, d=16, ~7 min wall time. Outputs `model.pt` (bases `U [K, 2048, d]`, means, eigvals, counts, history, `sample_fingerprint`), `assignments.pt` (file_id / cell_id / label per sampled token), and `sample.json` (reproducible manifest). Use `--files-from <dir>/sample.json` to cluster a new config on the **identical token sample** for direct comparison (runs sharing a fingerprint are comparable; same seed+tokens-per-file reproduces exactly).
- `analyze_subspaces.py` — generates a self-describing Markdown report from a result dir (convergence, variance decomposition, per-cluster spatial/temporal stats, subspace affinity, token-sample fingerprint + reproduce command) plus `dominant_cluster_map.png` (NESTED world map, cells colored by spectral seriation of the subspace-affinity matrix so similar clusters share hues). Backfills `sample.json` for pre-manifest runs. `python3 analyze_subspaces.py --dir <out_dir> --out <report.md>`.
- Completed runs: `subspace_out/` (1500 files, K=64, d=16); `subspace_big/` (7000 files, K=128, d=32, fingerprint `82ca602ed7e7`); `subspace_big_i100/` (same sample, 100 iters) queued 2026-06-12. Each has a `report.md`. Note: at large K the dominant-cluster map looks "scattered" only due to colormap aliasing — neighbor-agreement metrics confirm structure improves with K (see report's World map / README).

## Legacy — ignore

- `JL-Downscaling/` — earlier, unrelated Johnson–Lindenstrauss random-projection experiments (per-sample flattening to 2028 dims, and per-token 2048→256 projection). Not part of the current task.
- `latents_downscaled/` — output of those legacy scripts.
- `nohup.out`, `output.log`, `verify_script.*` — logs from the legacy runs.
