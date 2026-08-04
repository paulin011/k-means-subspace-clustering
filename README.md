# Weather-Encoder Latent Analysis

Structure analysis of token embeddings produced by a weather encoder trained on ERA5.
The dataset (`latents_2/`, 1.2 TB) holds 13,021 files `latent_{i}.pt`, one per sample
(time step), each a dict `{idx: int64 scalar, latent: float32 [1, 12288, 2048]}` —
12,288 HEALPix cells (nside=32, **NESTED ordering**) × 2048-dim token per cell, ≈160 M tokens total.

**Time axis:** the files are ERA5 states at **6-hourly cadence (4/day: 00/06/12/18 UTC)**.
Each file stores only an integer `idx` (its sample index), **not a timestamp**, so the
calendar date is reconstructed positionally as `datetime = 2014-01-01 00:00 UTC + idx×6h`
(file 0 = 2014-01-01 00:00; file 13020 = 2022-11-30 00:00 — a full 9-year 2014→2022 ERA5
range would run to 2022-12-31 18:00, ~127 more steps, so the last ~month of 2022 is not
fully covered). `temporal_spatial.py` groups files into calendar months this way.

**Dependencies:** `requirements.txt` (`numpy`, `torch`, `matplotlib`, `scipy`). On this
machine they are already present system-wide under `/usr/bin/python3` — there is no venv and
nothing to install; the file exists so the scripts can be run elsewhere. The scripts
deliberately avoid `healpy`/`cartopy`/`shapely` (`worldmap.py` reimplements the geometry and
parses coastlines with stdlib `json`).

## Layout

```
src/              scripts, grouped by role:
  common/           cluster_io.py, worldmap.py   — shared by every other directory
  clustering/       subspace_kmeans.py, holdout_eval.py, run_seed_sweep.sh
  analysis/         analyze_clusters.py, temporal_spatial.py, file_signature.py,
                    analyze_forecast_error.py   — everything that reads a finished run
  forecast/         persistence_error.py        — per-cell error extraction (runs here)
  supercomputer/    extract_forecast_error.py   — runs only where the WGen checkpoint lives
docs/     METRICS.md, wgen_architecture.md, INTERPRETATION_i100.md, ideas/
runs/     clustering/  signatures/  persistence/  forecast_error/   (all generated results)
logs/     run logs
assets/   ne_110m_coastline.geojson (coastline cache)
latents_2/            the 1.2 TB dataset (gitignored)
latents_downscaled/, JL-Downscaling/   legacy, see below
```

The split is by *role*, and the boundaries are meaningful: `common/` is the only
directory anything else imports; `clustering/` produces `model.pt`/`assignments.pt`;
`analysis/` only ever consumes them; `forecast/` writes the `[13020, 12288]` per-cell
error arrays; `supercomputer/` is quarantined because it imports `weathergen` and
therefore **cannot run on this box** (`ModuleNotFoundError` is the expected outcome here).

> **Run every script from the repository root** — `python3 src/clustering/subspace_kmeans.py`,
> never `cd src/clustering`. Each script's data-path defaults (`latents_2`, `runs/…`,
> `assets/…`) are root-relative, so a different working directory breaks them.
>
> Python only puts the invoked script's *own* directory on `sys.path`, so every script that
> needs `cluster_io`/`worldmap` adds `src/` itself, right above the import:
>
> ```python
> sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
> from common.cluster_io import load_tokens
> ```
>
> That two-line bootstrap (not a package install, not `PYTHONPATH`) is what keeps plain
> `python3 src/<role>/<script>.py` working. Copy it into any new script that imports from
> `common/`; scripts with no cross-import (`persistence_error.py`,
> `extract_forecast_error.py`) don't need it.

`README.md` and `CLAUDE.md` stay at the repository root by design: GitHub renders the former
as the landing page, and Claude Code only auto-loads the latter from the root.

## Documentation

- **`docs/PIPELINE.md`** — how the scripts chain together: what produces what, why each
  stage exists, and the cross-stage invariants. Start here if you want the architecture
  rather than a single script.
- **`docs/KERNEL.md`** — the matmul trick behind the assignment kernel (the **E-step**),
  explained from first principles. Read it if `subspace_kmeans.py:148-166` looks like plain
  k-means to you (it is the exact orthogonal residual to each subspace; the resemblance is
  the point of confusion the file clears up).
- **`docs/PCA.md`** — the companion: the **update step**, from the ground up. Opens with a
  worked-numbers primer (why the anchor is the average, the 3-4-5 triangle showing "capture
  most" and "leave least" are one goal, what an eigenvalue *is*), then covers why the top-d
  eigenvectors are provably optimal rather than a heuristic, why the moments are streamed,
  and why the loop converges (to a *local* minimum — hence the seed sweep). **§8b answers
  "is this really optimising subspaces, or just k-means with extra steps?"** with a
  measurement: the two assignment rules agree on only **43%** of tokens and the centre-only
  proxy leaves **20.1%** more residual.
- **`docs/METRICS.md`** — definition and interpretation of every number in `report.md`.
- **`docs/wgen_architecture.md`** — the upstream WeatherGenerator model, for the
  forecast-error work.

## Scripts

### `src/common/cluster_io.py` — shared sampling/IO for all clustering algorithms

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

### `src/clustering/subspace_kmeans.py` — K-subspaces clustering

K-means generalized to affine subspaces: each cluster is a mean μⱼ plus an orthonormal
basis `Uⱼ [2048, d]`, and each token is assigned to the cluster with the smallest
orthogonal residual ‖x−μⱼ‖² − ‖Uⱼᵀ(x−μⱼ)‖².

That residual is the **full distance to the affine subspace**, not to its centre: it is
algebraically identical to ‖(I − UⱼUⱼᵀ)(x−μⱼ)‖² (verified to 6e-06 relative), expanded into
norms and one `[B,2048]×[2048,K·d]` matmul so the projection costs one matmul instead of K
dense 2048×2048 projectors. Assigning by centroid distance instead would move 54.4% of
tokens; the d=64 subspaces absorb 65.6% of the distance-to-centre.

#### Soft assignment (`--soft`)

`--soft` replaces the hard argmin with EM on the matching **MPPCA** mixture
(`Cⱼ = Uⱼ diag(λⱼ) Uⱼᵀ + σ²ⱼ(I − UⱼUⱼᵀ)`), so a token carries a distribution over clusters
instead of one label. The Mahalanobis and log-det terms decompose into quantities the
kernel already computes, `R_j/σ²ⱼ + Σᵢzᵢ²/λᵢ` and `Σᵢlog λᵢ + (D−d)log σ²ⱼ`, so the extra
cost is two `[B,K,d]` einsums.

Why bother, given the partition barely moves (the likelihood rule flips only ~3.7% of
assignments)? Because **~31% of tokens sit within a 10% residual margin of a second
cluster** — for those the hard label discards real information and injects seed-dependent
noise. Soft assignment is uncertainty quantification, not a better partition; do not expect
the residual to drop.

- `--soft-topm` (default 4) truncates responsibilities to the best m clusters, capping the
  M-step at ~m× the hard cost instead of K×. Iteration 1 is always hard (no bases yet).
- **`--soft-temp` matters more than it looks.** At T=1 (the literal likelihood) the D=2048
  score is so sharply peaked that responsibilities collapse to one-hot — measured mean
  top-1 weight **0.993**, i.e. a slower hard k-subspaces. The likelihood treats all 2048
  dimensions as independent evidence while the data has only ~121 effective dimensions, so
  **T ≈ 2048/121 ≈ 17** removes that overcounting (deterministic-annealing EM).
  **Its empirical validation was retracted on 2026-08-04.** An early smoke test reported
  mean top-1 0.877 with 37.2% below 0.9 and claimed a match with the independently measured
  ~31% near-tie fraction. The production run (v10, 7000 files × 25 iterations) measures
  **mean top-1 0.950, 15.1% below 0.9, 0.9% below 0.5** — much more peaked. The smoke test
  ran 300 files for 3 iterations with dozens of re-seeded clusters, so its tokens really
  were ambiguous; and the two numbers never measured the same thing anyway (the margin uses
  only the residual, while the likelihood also uses within-subspace position, `σ²`, the
  log-det volume term and the mixing weights — strictly more information, so legitimately
  more decisive). T=17 still follows from the effective-dimension argument, which is a
  property of the data rather than of that run, but treat it as an **unvalidated prior**.
  Matching the measured ~33% ambiguity would need T ≈ 30–40; testable cheaply by rescoring
  the saved v10 model at several T **without refitting**.
- Outputs are **purely additive**: `model.pt` gains `sigma2`/`mixing`/`soft_counts`,
  `assignments.pt` gains `resp_idx [T,m]` int16 + `resp_w [T,m]` float16. `label` remains
  the argmax and `counts` the integer bincount of it, so every existing reader
  (`analyze_clusters.py`, `temporal_spatial.py`, `holdout_eval.py`, `file_signature.py`)
  works on a soft run unchanged.

```bash
python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v6_subspace_big_d64/sample.json \
    -K 128 -d 64 --soft --soft-temp 17 --max-ram-gb 420 --out runs/clustering/v10_soft_d64
```

Algorithm details:

- **One streaming sweep per iteration** over the sampled tokens (held in RAM as fp16),
  split across both A100s (one worker thread per GPU). Assignment and the per-cluster
  second-moment accumulation happen in the same sweep.
- **Exact per-cluster PCA**: bases come from batched `torch.linalg.eigh` of the
  2048×2048 cluster covariances — no randomized SVD or power iterations needed,
  because the covariances are cheap to accumulate on GPU during the sweep.
- **Init**: K random tokens as seeds with zero bases (first pass = nearest-seed k-means
  step). A random partition does *not* work — all K bases collapse onto the global PCA.
- **Tiny/empty clusters are re-seeded by splitting the largest healthy cluster** along
  its top principal axis (both halves inherit that cluster's subspace, offset ±1 std
  along PC1). A plain random-token re-seed (`U=0`) cannot recover at large `d` — a fresh
  point-seed has residual `‖x−μ‖²` while every rival owns a `d`-dim subspace with near-zero
  residual, so it never wins more than its one starter token (observed: `d=64` strands 1
  cluster, `d=128` strands 2 at size 1 forever). The split keeps all K subspaces populated.
  The `d=0` (plain k-means) path has no subspace to split, so it keeps the random-token re-seed.

Token selection: `--num-files` files drawn uniformly at random, `--tokens-per-file`
random cells from each (default: all 12,288). Every file covers the whole globe, so
file subsampling only thins the time axis. RAM = `num_files × tokens_per_file × 4096` bytes
(the fp16 `data` buffer; `--max-ram-gb` guards against this). True peak is a little higher —
the int32 `file_id`/`cell_id` arrays (~8 bytes/token) plus transient float32 decode buffers
(~`--load-workers` × 100 MB) — so leave headroom.

```bash
# defaults: 1500 files (~75 GB RAM), K=64, d=16, ≤25 iterations  — ~7 min wall time
nohup python3 src/clustering/subspace_kmeans.py --out runs/clustering/v1_subspace_out \
    > runs/clustering/v1_subspace_out/run.log 2>&1 &

# large run: 7000 files (~352 GB RAM), K=128, d=32  — ~1-1.5 h wall time
nohup python3 src/clustering/subspace_kmeans.py --num-files 7000 --clusters 128 --dim 32 \
    --iters 40 --max-ram-gb 420 --out runs/clustering/v2_subspace_big \
    > runs/clustering/v2_subspace_big/run.log 2>&1 &
```

**Reproducible sampling & cross-run comparison.** Every run writes `sample.json`
(a manifest with seed, tokens-per-file, the file-id list in load order, and a short
`fingerprint`). To cluster a *different* configuration on the **exact same tokens** —
the only fair way to compare, e.g. K or `d` or iteration count — reuse that sample:

```bash
python3 src/clustering/subspace_kmeans.py --files-from runs/clustering/v2_subspace_big/sample.json \
    --clusters 128 --dim 32 --iters 100 --out runs/clustering/v3_subspace_big_i100
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

### `src/analysis/analyze_clusters.py` — Markdown report generator

Algorithm-agnostic: reads any `model.pt`/`assignments.pt` following the `cluster_io.py`
schema, from `subspace_kmeans.py` or from k-means/k-center (the `d=0` point-cluster
case). Turns a result directory into a self-describing report (`report.md`):
configuration, convergence table, global variance decomposition (between-cluster /
subspace-captured / residual), a per-cluster table (size, spatial concentration over
HEALPix cells, temporal coverage and variation, plus EVR/effective-dimensionality when
`d>0`, plus k-center's `radius` when present), and pairwise subspace affinity (`d>0`
only — mean squared principal-angle cosines, flags merge candidates).

```bash
python3 src/analysis/analyze_clusters.py --dir runs/clustering/v1_subspace_out \
    --out runs/clustering/v1_subspace_out/report.md
```

The report opens with an **Overview** that defines the core quantities once (μⱼ, Uⱼ, the
orthogonal residual, `eigvals`, `trace`, `counts`), and each table is preceded by a *"How to
read this"* note naming the `model.pt` fields it comes from — so `report.md` is readable
standalone, without the reader having to hold the algorithm in their head.

The **world map and the temporal/seasonal analysis live in a separate report** — see
`temporal_spatial.py` below; `analyze_clusters.py` keeps the compact per-cluster
spatial/temporal columns and a pointer to `temporal_report.md`. Every metric in
`report.md` is defined and interpreted in **[METRICS.md](docs/METRICS.md)**.

**A metric only earns its column if it separates clusters.** Two changes came out of
auditing the v6 table for exactly that:

- `files` ("share of time steps where the cluster appears at all") was **vacuous** — with
  12,288 cells over K=128 clusters every cluster shows up somewhere in nearly every
  snapshot, so it read exactly 100% for 112 of 128 clusters and took 12 distinct values
  overall. Replaced by **`files@50%`**, the time-axis twin of `cells@50%`: the share of
  time steps holding half the cluster's tokens. Threshold-free, separates 126/128
  clusters, correlates only 0.13 with cluster size, and has a fixed reference point —
  **50% = perfectly uniform in time, lower = bursty/seasonal** (v6 spans 17%…50%).
- Added **`maxAff`**, each cluster's affinity to its single nearest neighbour. The affinity
  table only lists the top pairs globally, so a near-duplicate cluster was invisible there
  unless its pair happened to rank; this column always surfaces it.

Two formatting fixes in the same pass: `share` and `tCV` were quantised by their print
precision (14 and 19 distinct values across 128 clusters), now 72 and 85.

A **cluster health flags** line under the table calls out degenerate rows that are easy to
miss in a 128-row table: `owned == 0` (never any cell's majority ⇒ not a spatial regime),
size under ¼ of the mean, and `maxAff > 0.9` (near-duplicate ⇒ K too large). Validation
that it works: run without being told, it independently rediscovers the known stranded
clusters — v4's singleton (65) and v5's two (8, 65).

**Assignment margin (`--margins`, optional).** Every other column here is derived from the
saved model and the sampled assignments, so it is free; the one genuinely missing metric —
*how separated is each cluster in the data*, not just in subspace geometry — needs a data
pass. `file_signature.py` computes it over the whole dataset (see below) and writes
`cluster_margin.csv`; this report merges it in as the **`margin`** / **`near%`** columns and a
`near% > 50` health flag. Auto-detected from any `runs/signatures/*/` whose manifest names
this run as its frozen model, or point at it explicitly with `--margins <csv>`. Note this is
*not* the silhouette coefficient, which assumes Euclidean distance to a centroid and spherical
clusters and so does not apply to subspace clusters; see `docs/METRICS.md`.

### `src/common/worldmap.py` — shared HEALPix geometry, cluster coloring, and map renderer

Imported by both `analyze_clusters.py` and `temporal_spatial.py`, so the two reports
share one geometry, one cluster-color assignment, and one map renderer:

- Pure-numpy `healpix_nest2ring` / `healpix_ring_lonlat` — cell id → lon/lat (no healpy).
  Validated data-free: `nest2ring` is a bijection of `[0, 12288)`, lon/lat are clean
  (lat ∈ [−88.5°, 88.5°]), and the equal-area cell distribution matches analytics (50.0%
  at |lat|<30°; 13.7% at |lat|>60° vs 13.4% theoretical).
- `build_affinity_matrix` (subspace affinity for `d>0`, centroid cosine for `d=0`) and
  `affinity_ordered_colors` — **spectral seriation** of the affinity matrix via its Fiedler
  vector, mapped through the `turbo` colormap, so subspace-similar clusters get similar
  hues. Real structure then reads as smooth gradients, not the false "scatter" a random
  hue shuffle produces at large K.
- `get_coastlines()` — fetches + caches the Natural Earth 110m coastline GeoJSON (parsed
  with stdlib `json`; no cartopy/shapely, which aren't installed on this box).
- `render_world_map()` — a Mollweide map of the dominant cluster per cell with two
  readability features: **continent outlines**, and a smooth **heatmap** (the per-cell RGB
  is interpolated with `scipy.griddata` onto a regular grid, lightly Gaussian-smoothed, and
  painted with `pcolormesh`) that replaces the old speckled 12,288-pixel scatter with a
  continuous, coast-aligned field. Interpolating RGB (not the categorical cluster id) is
  valid precisely because `affinity_ordered_colors` already makes neighbors similar. ~0.8 s/map.
- `render_change_map()` — a Mollweide map of *where the dominant cluster differs* between
  two periods (e.g. Jan vs Jul). Cells that keep their label are drawn muted; cells that
  flip are colored by the transition, and the function returns the legend of the dominant
  source→destination pairs so the report can name them.

### `src/analysis/temporal_spatial.py` — temporal & spatial report

Dedicated report (`temporal_report.md`) for the spatial and temporal structure at calendar
(monthly) resolution:

- an **annual** dominant-cluster world map + **12 monthly** maps (continent outlines +
  smooth heatmap via `worldmap.py`), all on one shared color scale so months are comparable;
- **4 seasonal maps** (DJF/MAM/JJA/SON, Northern-Hemisphere convention — DJF spans the year
  boundary, so December is grouped with the *following* Jan/Feb) plus a season-to-season
  stability row, the coarser and less noisy companion to the monthly view;
- a **monthly enrichment** table flagging the most seasonal clusters (1.0 = year-round,
  ≫1 = concentrated in those months);
- a **month-to-month stability** curve — the share of cells whose dominant cluster flips
  between consecutive months, including the Dec→Jan wrap;
- **Jan → Jul change maps** (`map_change_jan_jul.png`, `map_change_jul_dest.png`) via
  `worldmap.render_change_map` — where the winter/summer dominant cluster flips, and which
  cluster each flipped cell moves *to*, with the top transitions named in a table.

```bash
python3 src/analysis/temporal_spatial.py --dir runs/clustering/v6_subspace_big_d64
```

It reconstructs each file's calendar month from `datetime = 2014-01-01 00:00 + idx×6h` and
reuses the run's **existing** `assignments.pt` (v6 covers all 12,288 cells in every
calendar month, ~528–623 files/month) — so there is **no recomputation**: it's a ~10 s
rendering/reporting pass. For a run that under-samples some month, re-cluster with more
`--num-files` and re-run.

### `src/clustering/holdout_eval.py` — generalization check on unseen files

The variance decomposition in the report is measured on the tokens the model was *fit*
on. With `d`-dim per-cluster bases the model has many free parameters (K·2048·d), so an
in-sample residual could be optimistic. This script freezes the trained means and bases
and replays the **exact** assignment rule (`‖x−μ_j‖² − ‖Uⱼᵀ(x−μ_j)‖²`) on tokens from
latent files the run never saw (sampled disjoint from `sampled_files`), then reports the
held-out residual fraction next to the in-sample one. A small gap ⇒ the subspaces capture
reusable structure; a large positive gap ⇒ overfitting (lower `--dim` or add tokens).

```bash
python3 src/clustering/holdout_eval.py --dir runs/clustering/v6_subspace_big_d64 --num-files 200
```

Writes `<dir>/holdout.json`; the next `analyze_clusters.py` run renders a **Held-out
generalization** section from it automatically. (v6, d=64: held-out 31.5% vs in-sample
31.5% — the subspaces generalise.)

### `src/analysis/file_signature.py` — per-file regime signature + residual (for timestamp selection)

Where the other scripts study the *clusters*, this one turns a frozen clustering run into a
**per-timestep summary of the entire dataset** — the coordinate system for "which timestamps
to train on" (goal 1 of the WeatherGenerator-training analysis). It replays
`holdout_eval.py`'s exact frozen-assignment rule (`‖x−μ_j‖² − ‖Uⱼᵀ(x−μ_j)‖²`) on **all
12,288 cells of every one of the 13,021 files**, but reduces the result *per file* instead of
per held-out set:

- `mix [K]` — fraction of the file's cells in each cluster (its regime-mix fingerprint; a
  low-dim snapshot signature, stable because clusters are geographic & time-stable);
- `mean_residual` — the file's mean orthogonal residual, i.e. the part the regime+subspace
  model leaves unexplained. Its count-weighted global mean reproduces the model's
  `final_obj_per_token` — a built-in correctness check (v6: 1887.16 over the held-out files,
  matching `holdout.json` to 4 sig figs);
- `residual_frac` = `mean_residual / mean_total` (globally ~31.5% for v6; high outliers = the
  anomalous / hard-to-encode timestamps — candidates for hard-example upweighting);
- `rare_exposure` — share of the file's tokens in the globally-rarest clusters (bottom
  quartile by token share; high = rich in rare regimes — candidates for class-balancing);
- the reconstructed calendar date (`2014-01-01 + idx×6h`; `idx` == file id, confirmed).

**Per-cell outlier maps (`residual_map.npy` / `label_map.npy`).** The assignment kernel
computes a residual for every one of the 12,288 cells and the per-file reduction throws that
resolution away. Keeping it costs one extra write per batch and is what turns outlier
detection — the main goal for later fine-tuning — from *per-timestep* into
*per-cell-per-timestep*:

- `residual_map.npy` `[N, 12288]` float32 (0.60 GiB at 13,021 files) — the per-cell residual.
  Averaging a row reproduces that row's `mean_residual` exactly (verified to 6e-08 relative).
- `label_map.npy` `[N, 12288]` int16 — the assigned cluster per cell, over the **whole**
  dataset. This is the dynamic counterpart to `assignments.pt`, which only covers the sampled
  files; a row's histogram reproduces that row's `mix` exactly.
- Rows follow the sorted file list (row *i* ↔ `file_ids[i]`), matching `signatures.npz` /
  `file_summary.csv` row order — **not** file id, so `--limit` / `--files-from` subsets stay
  dense. Written as `.npy` memmaps, so load with `mmap_mode="r"` and never pay 640 MB to read
  one row.
- **Normalise before ranking.** Raw residuals are not comparable across cells — the per-cell
  mean spans **15.7×** (273 … 4276), because a storm-track cell is intrinsically harder than a
  subtropical one. `signatures.npz` therefore also carries `cell_mean_residual[12288]` and
  `cell_std_residual[12288]`, the per-cell temporal climatology:

  ```python
  R = np.load("runs/signatures/v6_d64/residual_map.npy", mmap_mode="r")
  s = np.load("runs/signatures/v6_d64/signatures.npz")
  z = (R - s["cell_mean_residual"]) / np.maximum(s["cell_std_residual"], 1e-6)
  # z[t, cell] > 4  ->  that cell is far outside its own normal range at step t
  ```

Both arrays sit on the same NESTED `[T, 12288]` grid as `runs/persistence/*/err_persist.npy`,
so they gather onto each other with no remapping — intersecting them separates *anomalous
because rapidly changing* (high persistence error) from *anomalous because unmodelled* (high
residual, low persistence error). Pass `--no-cell-maps` to skip writing them.

**Assignment margin (`cluster_margin.csv` / `margin_map.npy`) — the per-cluster separation
metric.** The kernel ranks every token against all K subspaces, so the *runner-up* is free:
`topk(2)` replaces `min(1)` at no measurable cost and yields

```
margin(x) = (R₂(x) − R₁(x)) / R₁(x)        # dimensionless: how much worse the 2nd-best subspace is
```

A token with `margin < 10%` is a **near-tie** — assigned, but barely. This is the
subspace-native analogue of the silhouette coefficient, which does *not* transfer here (it
assumes Euclidean distance to a centroid and spherical clusters — the wrong geometry); the
margin is built from the exact residual the algorithm minimizes. It is reduced three ways:

- **per cluster** → `cluster_margin.csv`: `margin_mean`, `near_frac` (share of the cluster's
  own tokens that are near-ties), and `runner_up` / `runner_up_frac` from the `[K,K]`
  runner-up confusion matrix — each cluster's single closest competitor and how much of its
  contested mass that one rival takes. This decomposes the previously-known *global* ~31%
  near-tie figure **by cluster**, showing where hard labels actually discard information.
  `analyze_clusters.py` merges it into the per-cluster table as the `margin` / `near%`
  columns plus a `near% > 50` health flag (auto-detected from `runs/signatures/*/`, or
  `--margins <csv>`).
- **per file** → `mean_margin` / `near_frac` columns in `file_summary.csv`: a timestep whose
  cells are unusually contested is one the partition describes poorly — a hardness axis
  independent of `residual_frac`.
- **per cell** → `margin_map.npy` `[N, 12288]` float16 (0.30 GiB, clipped at 10), so ambiguity
  is localisable on the globe exactly as `residual_map.npy` localises the residual.

**Relation to `maxAff`.** `maxAff` is a purely geometric angle between two bases that never
touches the data; `near%` measures whether two clusters actually contest the same tokens. On
v6 they **correlate strongly** (Pearson +0.77, Spearman +0.78) — the expected direction, since
overlapping orientation does tend to mean contested tokens. `near%` still adds information
though: 41% of its variance is unexplained by `maxAff`, its range is far wider (0.9%…70.1% vs
0.526…0.805), and it names a different closest rival for 52% of clusters. They diverge because
`maxAff` sees orientation only, blind to where the means sit and where the data is dense — v6
c123 has `maxAff` 0.697 but just 1.3% near-ties, while c122 has a *lower* `maxAff` 0.668 and
45.7%. A high global near-tie rate is not a defect of the fit; it is the measurement that
motivates the soft/MPPCA assignment (`subspace_kmeans.py --soft`).

Measured on v6 (`runs/signatures/v6_d64_margin/`): **33.4%** of all tokens are near-ties,
reproducing the ~31% estimated earlier by an independent route, and `near_frac` takes 118/128
distinct values with correlation +0.04 against cluster size. The global figure hides a wide
spread — c21/c66/c27 sit near 1% (crisp modes) while c106/c45 sit near 70% and are each
other's runner-up (one continuum cut in half); 29 of 128 clusters have most of their own
tokens near-tied.

**Streaming, not load-all:** the full dataset is ~656 GB as fp16 > 512 GB RAM, so files are
loaded in batches (`--batch-files`, ~25 GB each at the default 500), assigned on a single
GPU, and scattered into `[N_FILES]`-wide accumulators. I/O dominates (~17 files/s ⇒ ~13 min
to read all 1.3 TB), so one GPU suffices and the broken inter-GPU P2P never enters.

```bash
# full run over all 13,021 files (~20 min); writes signatures.npz, file_summary.csv,
# cluster_mix.csv, cluster_margin.csv, manifest.json, report.md, residual_map.npy,
# label_map.npy, margin_map.npy under --out
python3 src/analysis/file_signature.py --out runs/signatures/v6_d64
# quick correctness check on 200 files: the printed mean mean_residual must match
# the model's final_obj_per_token (~1888), and mean residual_frac ~31.5%
python3 src/analysis/file_signature.py --limit 200 --out runs/signatures/_smoke
```

`cluster_mix.csv` is the feature matrix for the next step — k-center / farthest-point in
mix-space picks a minimal *covering* subset of timestamps (anti-redundancy: consecutive
snapshots are near-duplicates in mix), and the `month`/`rare_exposure`/`residual_frac`
columns drive seasonal / rare-regime / hard-example upweighting. The most principled variant
re-runs this same script with the model's *forecast errors* as input, so the signature flags
the regimes where the predictor actually fails.

### `docs/wgen_architecture.md` — upstream WeatherGenerator architecture (for the forecast-error step)

To ground the next step (attributing the model's *forecast* error to these clusters), the
upstream ECMWF **WeatherGenerator** model was cloned read-only to `/home/psaher/WeatherGenerator`
and mapped by a 6-reader dynamic workflow (`docs/wgen_architecture.md` in this dir). The findings
(all HIGH confidence) are what make the forecast-error step cheap:

- The `ForecastingEngine` is **latent→latent 2048-d** — it predicts the next-step *latent*
  tokens (`engines.py:564-636`, `model.py:703`, comment "roll-out in latent space"). The
  physical decode is a *separate* branch used only for the loss.
- Therefore `forecast_engine(latents_2[t])` predicts `latents_2[t+1]`, and **both already
  exist on disk** — the forecast-error sweep needs only the `forecast_engine` forward, no
  encoder/decoder/ERA5/Anemoi.
- `latents_2`'s `latent [1,12288,2048]` **is** the encoder output `tokens_global`
  (`encoder.py:140`); NESTED ordering is set in the tokenizer layer (`datasets/utils.py`) —
  the same index order as `assignments.pt`, so per-cell error gathers on with **no remap**.
- The model is **fully dense (zero MoE)**; the natural MoE slot is the per-block FFN
  (`layers.py` MLP), with the 16 `ForecastingEngine` blocks the highest-leverage target.
- Timestep sampling is **uniform random, no weighting** — the goal-1 weighted-sampling hook
  is `multi_stream_data_sampler.py` `reset()`/`_calc_baseperms` and must be inlined
  (`IterableDataset` blocks `WeightedRandomSampler`).

The report's §4 is the concrete supercomputer plan, now concretized in the three scripts
below (`persistence_error.py` → `extract_forecast_error.py` → `analyze_forecast_error.py`):
produce a `[13020,12288]` per-cell latent residual and gather it directly onto
`assignments.pt` (both NESTED — no remap). Open question to verify on the supercomputer:
confirm `latents_2[t]` is the step-0 encoder state feeding `forecast_engine` (decode check)
rather than a forecast-step latent — `extract_forecast_error.py`'s built-in self-check settles
this (it aborts if skill vs persistence is ≤ 0).

### `src/forecast/persistence_error.py` — per-cell latent PERSISTENCE baseline (model-free, local)

The **persistence baseline** the WGen repo lacks (`docs/wgen_architecture.md` decision f): how far
does "predict t+1 = t" land from truth, per cell — computable on the analysis box *right now*
with no checkpoint, because the latents are already on disk. It is the skill-score
denominator for the forecast error and itself measures how *changeable* each cell/timestamp is
between 6h steps (dynamic/stormy vs quiescent).

- For each source `t` (`0..N-2`), `err_persist[t,cell] = ‖x_t[cell] − x_{t+1}[cell]‖²` →
  `[N=13020, 12288]` float32, per-cell squared-L2 over the 2048 latent dim.
- Streams files with a threaded prefetch reader; one file read per transition, all on CPU
  (~15 min for the full 1.3 TB; no GPU needed — it is a pairwise difference).
- **Integrity-verified:** recomputing `‖x_t − x_{t+1}‖²` from raw `latents_2` for a scattered
  12-row sample matches the saved array to 0.00e+00 relative error, and file `idx` keys match
  `source_idx` positionally — so every downstream number rests on this foundation.
- Outputs `<out>/{err_persist.npy, meta.json}`. The full run lives in `runs/persistence/v6/`
  (global mean per-cell 6h error ≈ **3149.6**).

```bash
python3 src/forecast/persistence_error.py --out runs/persistence/v6            # full run (~15 min)
python3 src/forecast/persistence_error.py --limit 200 --out runs/persistence/_smoke   # quick check
```

### `src/supercomputer/extract_forecast_error.py` — per-cell latent FORECAST error (supercomputer only)

Runs **on the supercomputer**, inside the WeatherGenerator env where the trained checkpoint +
production Config + dataset live (not on this box — no checkpoint here). For each source `t`,
`pred = forecast_engine(x_t, step=0, rope_coords)`, `err_forecast[t,cell] = ‖pred − x_{t+1}‖²`
alongside the persistence baseline — same `[13020,12288]` contract, so `analyze_forecast_error.py`
consumes either/both. Multi-lead rollout (`t→t+k`) is intentionally not implemented.

- **Built-in self-check** (answers the blocking open question): on the first batch it prints
  `skill = 1 − mean_forecast/mean_persist`; `skill>0` ⇒ `latents_2[t]` is the correct step-0
  forecast input and the run proceeds, `skill≤0` ⇒ wrong input/rope_coords/config ⇒ it **ABORTS
  before the full sweep**. No ERA5 needed. `--selfcheck-only` runs just this.
- The forecast engine is latent→latent and time-homogeneous (`fstep` unused inside its forward),
  so feeding `latents_2[t]` straight in *is* the model's own forward path; the only op between
  encoder output and `forecast_engine` is an identity-when-`num_input_steps=1` input-step sum.
- Two TODOs marked in-file: `build_dataset()` (construct the production Anemoi dataset as the
  trainer does) and the Config loader (use WGen's training-time merge).

```bash
# 1) cheap verify (loads model, one batch, prints skill vs persistence, exits):
python3 src/supercomputer/extract_forecast_error.py --config <prod.yml> --run-id <id> --selfcheck-only
# 2) full sweep (detached):
setsid nohup python3 src/supercomputer/extract_forecast_error.py --config <prod.yml> --run-id <id> \
    --out err_forecast/run0 > err_forecast.log 2>&1 &
```

### `src/analysis/analyze_forecast_error.py` — attribute forecast/persistence error to clusters + goal-1 weights

The analyzer that turns the per-cell error into the two analysis goals: **(1)** training-
timestamp weights (which timesteps to emphasize) and **(2)** understanding the embeddings
(where the forecast actually fails). Two modes:

- **Bridge** (now — only `err_persist.npy` present): persistence = the 6h-tendency baseline and a
  defensible hardness proxy. Skill/forecast columns masked, awaiting `err_forecast.npy`.
- **Full** (when `err_forecast.npy` lands beside `err_persist.npy`, auto-detected): latent
  **skill = 1 − err_fc/err_ps**, computed as a **ratio-of-sums** (`1 − Σfc/Σps`), never a mean
  of per-cell skill.

**Attribution** uses the static dominant-per-cell map (`mode(label)` per cell over the 7000-file
sample) as the full-coverage default; a **free** static-vs-dynamic gate replays the per-token
labels already in `assignments.pt` on the 7000∩err intersection to test "are the clusters
temporally universal?" Validated v6 result: **NMI 0.76–0.78 every month, none contested** ⇒ the
static map is trustworthy; per-cell purity mean 0.656 (≈34% per-token mis-attribution),
zero-cell clusters `[13,98]` flagged for the optional `--dynamic` full pass.

**Goal-1 weights** (`weights.csv`, one row per source transition) fuse three rank-normalized
percentile axes (hardness, rare-regime, diversity=1/local-density in mix-space) as a weighted
**mean** — not a product, which saturates the clip — through `1+(Wmax−1)·s^α`, renormalized to
mean 1. Validated range `w_final ∈ [0.34, 2.45]`, σ 0.41, physically sensible (spring/autumn
6h-tendencies up-weighted over quiescent winter). Anti-leakage + handoff caveats in the report:
these are *emitted*; WGen's uniform-random `IterableDataset` sampler would need an inlined
weighted/stratified variant to apply them.

```bash
# bridge (now): persistence-only attribution + goal-1 weights
python3 src/analysis/analyze_forecast_error.py --err-dir runs/persistence/v6 --out runs/forecast_error/persist_v6
# full (when err_forecast.npy exists in <err-dir>): skill + full attribution + map_skill.png
python3 src/analysis/analyze_forecast_error.py --err-dir err_forecast/run0 --out forecast_error/full_v6
```

Outputs (`runs/forecast_error/persist_v6/`): `summary.json`, `per_cluster.csv`, `temporal.csv`
(hour/month/year, 2022 flagged partial), `weights.csv`, `static_vs_dynamic.json`, `report.md`,
`maps/{map_persist.png, map_dominant.png}` (+`map_forecast/p95/skill.png` in full mode).
`map_persist.png` is physically coherent — highest 6h tendency in the tropics + mid-latitude
storm tracks, lowest at the poles + subtropical deserts/gyres — confirming the latents encode
meaningful atmospheric structure.

### Chained run (fire and forget)

```bash
out=runs/clustering/v2_subspace_big; mkdir -p "$out"
nohup bash -c "python3 src/clustering/subspace_kmeans.py --num-files 7000 --clusters 128 --dim 32 \
  --iters 40 --max-ram-gb 420 --out $out && \
  python3 src/analysis/analyze_clusters.py --dir $out --out $out/report.md" \
  > $out/run.log 2>&1 &
```

## Results so far

Runs live under `runs/clustering/`, each in a `vI_<name>` directory numbered in
chronological order (v1 → v8 below).

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
  d=128. **Note:** 2 clusters were stranded at size 1 — a degeneracy of large `d` with the
  old random-token re-seed, *not* of large K (d=32 at the same K is singleton-free); see v6.
- `v6_subspace_big_d64/` — **supersedes v4**: identical config (same sample, K=128, d=64)
  but with the split-largest re-seed guard. The one cluster v4 stranded at size 1 is
  rescued: all 128 clusters healthy (min 227,886 tokens), and `max d80` drops 65→**40**
  (every cluster now needs ≤40 of its 64 dims for 80% of captured variance — the spectrum
  is no longer truncated, unlike d=32 where d80≈d). Variance decomposition is otherwise
  identical to v4 (8.4% between / 60.1% within / 31.5% residual), confirming the guard only
  fixed the degenerate cluster. **This is the recommended config**: d=64 captures the
  within-cluster structure without the singleton degeneracy and washed-out cluster identity
  (between-cluster share) that d=128 causes.
- `v7_seed1_d64/`, `v8_seed2_d64/` — **seed-robustness runs**: the v6 config re-run with
  init seeds 1 and 2 on the *same* file sample (only `--seed` changes), to confirm d=64/K=128
  is a stable optimum basin rather than a lucky init. Their convergence history is in
  `report.md`; the full per-run logs are in `logs/seed_sweep.log` (repo root), not a per-dir `run.log`.
- `v9_seed2_d64/` — **not a new clustering run**: the v8 model re-reported with the upgraded
  `analyze_clusters.py` + `temporal_spatial.py` (reader's guide, seasonal maps, Jan→Jul change
  maps). It therefore holds only `report.md`, `temporal_report.md`, `sample.json` and `maps/` —
  the `model.pt`/`assignments.pt` it describes live in `v8_seed2_d64/`, and both reports carry
  that directory in their header. Numbers are v8's; only the presentation is new.
- `v10_soft_d64/` — the first real **`--soft` MPPCA** run (same v2 token sample, K=128, d=64,
  `--soft-topm 4 --soft-temp 17`, 25 iterations, 161.3 min). The verdict: **soft assignment
  does not change the clustering, it annotates it.**

  | comparison | NMI vs v6 |
  |---|---|
  | v7 (different seed, hard) | 0.683 |
  | v8 (different seed, hard) | 0.688 |
  | **v10 (soft)** | **0.737** |

  Switching from hard k-subspaces to MPPCA EM perturbs the partition *less than changing the
  random seed does*. The objective is likewise inside the seed spread: 1892.47 against v6's
  1887.83 (+0.25%; v7 was 1892.30). So per-token responsibilities cost essentially nothing in
  hard-residual terms. `sigma2` lands in [0.30, 1.63] — far above the 1e-3 floor, so no EM
  collapse — and the minimum cluster is 20,866 tokens: no stranding, though ~10× smaller than
  v6's 227,886, so the soft size distribution is more uneven.
  Caveat on the responsibilities themselves: mean top-1 is **0.950** with only 15.1% of tokens
  below 0.9, more peaked than intended — see the `--soft-temp` note above, whose empirical
  validation has been retracted.
- **Seed robustness (the final check).** Across v6 (seed 0), v7 (seed 1), v8 (seed 2):

  | metric | seed 0 (v6) | seed 1 (v7) | seed 2 (v8) |
  |---|---|---|---|
  | final objective/token | 1887.8 | 1892.3 | 1888.2 |
  | variance split (between / top-64 / residual) | 8.4 / 60.1 / 31.5% | 8.5 / 59.9 / 31.6% | 8.5 / 60.1 / 31.5% |
  | within-cluster EVR(top-64) min/med/max | 0.590/0.648/0.782 | 0.586/0.649/0.760 | 0.585/0.648/0.786 |
  | `max d80` (dims for 80% of captured var) | 40 | 40 | 40 |
  | smallest cluster size | 227,886 | 193,240 | 235,389 |
  | tiny/stranded clusters | 0 | 0 | 0 |
  | month-to-month flip min/mean/max | 6.2/13.0/21.5% | 6.9/13.4/22.3% | 5.5/12.7/22.2% |
  | Jan↔Jul (winter↔summer) cell shift | 45.0% | 46.0% | 47.0% |

  The objective, variance decomposition, EVR, d80, cluster health, and the *shape* of the
  seasonal cycle (spring/autumn transition peaks at Apr→May and Oct→Nov, mid-summer minimum
  at Jul→Aug) all reproduce to within ~0.5%. The only thing that changes across seeds is the
  arbitrary cluster *labeling* — e.g. the biggest summer-grower is cluster 82 in v6, 21 in v7,
  102 in v8. On the spatial partition itself (label-invariant), the three runs agree with
  **normalized mutual information 0.72** between every pair (vs 0.13 for a random shuffle;
  adjusted Rand index 0.36 vs ≈0.0). **Verdict: the d=64/K=128 result is seed-stable; v6 is
  the final configuration.** Driver: `src/clustering/run_seed_sweep.sh` (detached `setsid nohup`).
- The **temporal & spatial report** (`temporal_spatial.py` → `temporal_report.md`) breaks
  v6 down by calendar month: clusters 24/82/73/104 peak in NH summer, 7/29/67 in winter;
  month-to-month dominant-cluster flips range 6.2% (Jul→Aug) to 21.5% (Apr→May), and
  Jan↔Jul differ in 45% of cells — a clear hemispheric seasonal cycle. Maps reuse v6's
  existing assignments (no re-clustering).

## Hardware notes (this machine)

- 48-core CPU, 512 GB RAM, 2× A100 40 GB, `/usr/bin/python3` + PyTorch 2.6.0 (no venv).
- **GPU↔GPU peer copies are silently broken**: `tensor.to()` between `cuda:0` and
  `cuda:1` returns zeros/garbage with no error, although `can_device_access_peer`
  reports True. Route all inter-GPU transfers through CPU, and do device-to-host copies
  from the thread that launched the producing kernels. Both scripts follow this rule.

## Legacy

`JL-Downscaling/` (Johnson–Lindenstrauss projection experiments) and
`latents_downscaled/` (its output) predate this work and are unrelated.
