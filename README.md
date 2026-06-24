# Weather-Encoder Latent Analysis

Structure analysis of token embeddings produced by a weather encoder trained on ERA5.
The dataset (`latents_2/`, 1.2 TB) holds 13,021 files `latent_{i}.pt`, one per sample
(time step), each a dict `{idx: int64 scalar, latent: float32 [1, 12288, 2048]}` —
12,288 HEALPix cells (nside=32, **NESTED ordering**) × 2048-dim token per cell, ≈160 M tokens total.

## Scripts

### `cluster_io.py` — shared sampling/IO for all clustering algorithms

Every clustering script (`subspace_kmeans.py`, and any k-means / k-center scripts) is
expected to import this module rather than reimplement sampling or output, so that runs
from different algorithms can be compared directly:

- `load_tokens(args)` — samples `--num-files` latent files uniformly at random (or
  reuses an exact file list via `--files-from`), loads them in parallel, and writes
  `sample.json` (the reproducible manifest: fingerprint, seed, tokens-per-file, file
  list in load order).
- `sample_fingerprint(files, tokens_per_file, seed)` / `load_file_list(path)` — used to
  identify/reuse a token sample.
- `save_model(out_dir, **fields)` — validates shapes and writes `model.pt` under the
  shared schema below.
- `save_assignments(out_dir, file_id, cell_id, label)` — writes `assignments.pt`.

**`model.pt` contract** (required regardless of algorithm): `U [K, 2048, d]` (`d=0` for
plain point clusters — k-means/k-center have no subspace, just a centroid), `means
[K, 2048]`, `eigvals [K, d]`, `trace [K]` (mean squared distance to centroid — what
k-means/subspace_kmeans optimize), `counts [K]`, `explained_var_ratio [K]` (0 when
`d=0`), `config` (must include `config["method"]` ∈ `{"kmeans", "kcenter",
"subspace_kmeans"}`), per-iteration `history`, `sampled_files`, `sample_fingerprint`.
Optional: `radius [K]` — k-center's native minimax objective (max distance from
centroid to any member), populated only by k-center.

K-means is literally the `d=0` case of K-subspaces clustering: the orthogonal-residual
assignment formula collapses to plain squared distance to centroid when there's no
basis. `subspace_kmeans.py --dim 0` exercises this path directly.

`assignments.pt` contract (all algorithms): `file_id` / `cell_id` / `label` (int32) for
every sampled token.

### `subspace_kmeans.py` — K-subspaces clustering

K-means generalized to affine subspaces: each cluster is a mean μⱼ plus an orthonormal
basis `Uⱼ [2048, d]`, and each token is assigned to the cluster with the smallest
orthogonal residual ‖x−μⱼ‖² − ‖Uⱼᵀ(x−μⱼ)‖².

Algorithm details:

- **One streaming sweep per iteration** over the sampled tokens (held in RAM as fp16),
  split across both A100s (one worker thread per GPU). Assignment and the per-cluster
  second-moment accumulation happen in the same sweep.
- **Exact per-cluster PCA**: bases come from batched `torch.linalg.eigh` of the
  2048×2048 cluster covariances — no randomized SVD or power iterations needed,
  because the covariances are cheap to accumulate on GPU during the sweep.
- **Init**: K random tokens as seeds with zero bases (first pass = nearest-seed k-means
  step). A random partition does *not* work — all K bases collapse onto the global PCA.
- Tiny/empty clusters are re-seeded from random tokens automatically.

Token selection: `--num-files` files drawn uniformly at random, `--tokens-per-file`
random cells from each (default: all 12,288). Every file covers the whole globe, so
file subsampling only thins the time axis. RAM = `num_files × tokens_per_file × 4096` bytes.

```bash
# defaults: 1500 files (~75 GB RAM), K=64, d=16, ≤25 iterations  — ~7 min wall time
nohup python3 subspace_kmeans.py --out subspace_kmeans_runs/v1_subspace_out \
    > subspace_kmeans_runs/v1_subspace_out/run.log 2>&1 &

# large run: 7000 files (~352 GB RAM), K=128, d=32  — ~1-1.5 h wall time
nohup python3 subspace_kmeans.py --num-files 7000 --clusters 128 --dim 32 \
    --iters 40 --max-ram-gb 420 --out subspace_kmeans_runs/v2_subspace_big \
    > subspace_kmeans_runs/v2_subspace_big/run.log 2>&1 &
```

**Reproducible sampling & cross-run comparison.** Every run writes `sample.json`
(a manifest with seed, tokens-per-file, the file-id list in load order, and a short
`fingerprint`). To cluster a *different* configuration on the **exact same tokens** —
the only fair way to compare, e.g. K or `d` or iteration count — reuse that sample:

```bash
python3 subspace_kmeans.py --files-from subspace_kmeans_runs/v2_subspace_big/sample.json \
    --clusters 128 --dim 32 --iters 100 --out subspace_kmeans_runs/v3_subspace_big_i100
```

`--files-from` accepts a `sample.json` or a `model.pt`; it overrides `--num-files`.
Runs that share a fingerprint were clustered on identical tokens and are directly
comparable (the report prints the fingerprint), even across algorithms. The token set is
order-independent, but load order — and therefore the random init — is preserved, so
reusing a sample with the same `--seed`/`--tokens-per-file` reproduces a run *exactly*.
`analyze_clusters.py` backfills `sample.json` for older runs that predate the manifest.

Outputs (in `--out`), via `cluster_io.py`'s `save_model`/`save_assignments` — see the
shared contract above:

| file | contents |
|---|---|
| `model.pt` | `U [K, 2048, d]` orthonormal bases (descending eigenvalue order), `means [K, 2048]`, `eigvals [K, d]`, `trace [K]`, `counts [K]`, `explained_var_ratio [K]`, `config`, per-iteration `history`, `sampled_files`, `sample_fingerprint` |
| `assignments.pt` | `file_id` / `cell_id` / `label` (int32) for every sampled token — maps each assignment back to its HEALPix cell and time step |
| `sample.json` | reproducible manifest (fingerprint, seed, tokens-per-file, file-id list in load order) — feed to `--files-from` |

Project a token onto its cluster subspace with `(x - means[j]) @ U[j]`.

### `analyze_clusters.py` — Markdown report generator

Algorithm-agnostic: reads any `model.pt`/`assignments.pt` following the `cluster_io.py`
schema, from `subspace_kmeans.py` or from k-means/k-center (the `d=0` point-cluster
case). Turns a result directory into a self-describing report (`report.md`):
configuration, convergence table, global variance decomposition (between-cluster /
subspace-captured / residual), a per-cluster table (size, spatial concentration over
HEALPix cells, temporal coverage and variation, plus EVR/effective-dimensionality when
`d>0`, plus k-center's `radius` when present), pairwise subspace affinity (`d>0` only —
mean squared principal-angle cosines, flags merge candidates), and temporal enrichment
profiles of the most time-varying clusters.

```bash
python3 analyze_clusters.py --dir subspace_kmeans_runs/v1_subspace_out \
    --out subspace_kmeans_runs/v1_subspace_out/report.md
```

It also renders `dominant_cluster_map.png`: a Mollweide world map of the dominant
cluster per cell under NESTED ordering (pure-numpy HEALPix geometry, no healpy needed).
The encoder is confirmed to use NESTED cell indexing — geographically coherent continent-
scale regions appear under NESTED, incoherent stripes under RING. Cells are colored by
**spectral seriation of the subspace-affinity matrix** (or, for `d=0` point clusters,
centroid-cosine affinity) via Fiedler-vector order through the `turbo` colormap, so
similar clusters get similar colors: real structure reads as smooth gradients while
genuine noise stays speckled — this avoids the false "scatter" a random hue shuffle
produces at large K. Use the script's `healpix_nest2ring` + `healpix_ring_lonlat` helpers
to map any cell id to lat/lon.

### Chained run (fire and forget)

```bash
out=subspace_kmeans_runs/v2_subspace_big; mkdir -p "$out"
nohup bash -c "python3 subspace_kmeans.py --num-files 7000 --clusters 128 --dim 32 \
  --iters 40 --max-ram-gb 420 --out $out && \
  python3 analyze_clusters.py --dir $out --out $out/report.md" \
  > $out/run.log 2>&1 &
```

## Results so far

Runs live under `subspace_kmeans_runs/`, each in a `vI_<name>` directory numbered in
chronological order (v1 → v5 below).

- `v1_subspace_out/` — first full run (1500 files, K=64, d=16,
  6.7 min). Key findings: clusters are **spatially localized but temporally universal**
  (geographic regimes — each present in ~100% of time steps but concentrated in a few
  hundred of the 12,288 cells); cluster identity alone explains 8.5% of token variance,
  the top-16 subspaces a further 33.5%; within-cluster spectra are fairly flat
  (d80 ≈ 10–12 of 16), motivating larger `--dim`.
- **The encoder's HEALPix cell indexing is NESTED** (established from
  `v1_subspace_out/dominant_cluster_map.png`: coherent continent-scale regions under the
  NESTED interpretation, incoherent stripes under RING).
- `v2_subspace_big/` — larger run (7000 files ≈ 86 M tokens, K=128, d=32, 65 min,
  fingerprint `82ca602ed7e7`). Within-cluster EVR rose to 0.49 (d=32 captures more);
  geographic structure sharpened into clear latitude bands + continents.
- `v3_subspace_big_i100/` — same sample as v2 (fingerprint `82ca602ed7e7`),
  K=128 d=32 but **100 iterations**, to see how much further the objective settles past
  iter 40 (it was still at ~0.9% labels-changed). Queued 2026-06-12.
- **The K=128 map looked "more scattered" than K=64 — but that was a visualization
  artifact, not worse clustering.** Measured neighbor-agreement (share of adjacent cells
  with the same dominant cluster) was 0.72 at K=64 vs 0.60 at K=128, yet *relative to
  chance* the big run is more structured (46× vs 77× the 1/K baseline), and its per-cell
  mode-purity is higher (0.31 → 0.35). Doubling K just doubles region boundaries and a
  random hue shuffle aliased the finer sub-regions; the affinity-ordered colormap (above)
  restores smooth gradients.
- `v4_subspace_big_d64/` — same sample as v2 (fingerprint `82ca602ed7e7`), K=128 but
  d=64. Count-weighted within-cluster EVR(top-64) rose to 0.664 and residual variance
  dropped to 31.5%; d80 (min/median/max 28/37/65) still close to d, so the spectrum stays
  fairly flat — larger `--dim` keeps paying off.
- `v5_subspace_big_d128/` — same sample as v2 (fingerprint `82ca602ed7e7`), K=128, d=128.
  EVR(top-128) reached 0.816, residual down to 17.8%; d80 (min/median/max 47/63/129) is
  again close to d, so within-cluster structure is still not fully captured even at
  d=128.

## Hardware notes (this machine)

- 48-core CPU, 512 GB RAM, 2× A100 40 GB, `/usr/bin/python3` + PyTorch 2.6.0 (no venv).
- **GPU↔GPU peer copies are silently broken**: `tensor.to()` between `cuda:0` and
  `cuda:1` returns zeros/garbage with no error, although `can_device_access_peer`
  reports True. Route all inter-GPU transfers through CPU, and do device-to-host copies
  from the thread that launched the producing kernels. Both scripts follow this rule.

## Legacy

`JL-Downscaling/` (Johnson–Lindenstrauss projection experiments) and
`latents_downscaled/` (its output) predate this work and are unrelated.
