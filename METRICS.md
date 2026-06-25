# Clustering report metrics

Every metric emitted by `analyze_clusters.py` (the report generator for any
`model.pt`/`assignments.pt` following the `cluster_io.py` schema — `subspace_kmeans.py`,
or k-means/k-center as the `d=0` point-cluster case), and how to read it. Sections
below mirror the sections of the generated `report.md`.

## Setup quantities

- `K, D, d = U.shape` — **K** clusters, **D = 2048** embedding dim, **d** = subspace
  dimension. **`d = 0` is the plain k-means / k-center "point cluster" case** (no basis,
  just a centroid). Every subspace-specific metric below appears only when `d > 0`.
- `w = counts / counts.sum()` — **count weights**, each cluster's fraction of all tokens.
  Used as the weighting for every "count-weighted" average, so large clusters dominate
  global figures.

## Time axis

The files are ERA5 states at **6-hourly cadence** (00/06/12/18 UTC — 4/day). Each stores
only an integer sample index (`idx`), not a timestamp, so the calendar date is
**reconstructed positionally**: `datetime = 2014-01-01 00:00 UTC + idx×6h`. The 13,021
files therefore span 2014-01-01 00:00 → 2022-11-30 00:00 (a full 9-yr 2014→2022 range would
reach 2022-12-31 18:00, ~127 more steps; the last ~month of 2022 is not fully covered).
Every calendar-month / seasonal metric below groups files by this reconstructed date.

## Convergence table

Per-iteration optimization history:

- **objective/token** (`obj_per_token`) — mean squared residual per token: squared
  orthogonal distance to the assigned affine subspace (for `d = 0`, squared distance to
  the centroid). Decreases monotonically and flattens at convergence. The orthogonal
  residual is ≥ 0 mathematically, but TF32 matmul rounding can make it tiny-negative,
  so only the objective *sum* is `clamp_min(0)`; the cluster *assignment* (argmin) is
  taken on the raw residual — the clamp stays after `min`, never before (clamping first
  would distort assignments between near-tied subspaces).
- **`final_obj_per_token`** (top-level, not per-iteration) — the objective of the *saved*
  model measured on its final relabel sweep; this is the true minimised residual under
  the saved bases. The per-iteration `history` objectives are one step behind (each is
  the *previous* model's objective), so `holdout_eval.py` compares the held-out
  objective against `final_obj_per_token`, not `history[-1]`. (Absent in older runs.)
- **labels changed** (`frac_changed`) — fraction of tokens that switched cluster this
  iteration; → 0 means converged.
- **min size / max size** — smallest and largest cluster token counts; watch for
  collapsing/degenerate clusters.

## Global variance decomposition

Splits total token variance `E‖x − μ_global‖²` into additive parts. `μ_global` is the
count-weighted average of the cluster means.

- **between** = `Σ wⱼ ‖μⱼ − μ_global‖²` — variance explained by *cluster identity alone*
  (how far apart the centroids are). High ⇒ well-separated clusters.
- **within** = `Σ wⱼ · traceⱼ` — average within-cluster scatter (mean squared distance of
  members to their own centroid).
- **captured** = `Σ wⱼ · Σ eigvalsⱼ` (`d > 0` only) — portion of within-cluster variance
  lying in the top-`d` subspace directions.
- **residual** = `within − captured` — variance the model leaves unexplained (orthogonal
  to the subspace). For `d = 0`, residual = within entirely (a centroid captures nothing
  beyond itself).
- **total** = between + within.

Ideal: most variance is *between* (clean separation) plus *captured* (subspaces fit the
within-cluster spread), with small *residual*.

For `d > 0` it also reports:

- **Count-weighted within-cluster EVR(top-d)** = `Σ wⱼ · evrⱼ` — average fraction of each
  cluster's own variance captured by its `d`-dim subspace. Near 1 ⇒ subspaces fit well.
- **d80** — number of subspace dimensions needed to reach 80% of *captured* variance,
  reported as min/median/max across clusters. **d80 ≈ d ⇒ flat spectrum** (the subspace is
  truncating real structure → increase `--dim`); **d80 ≪ d ⇒ concentrated spectrum** (`d`
  may be larger than needed). d80 is capped at `d+1`: a cluster reporting `d+1` never
  reaches 80% even using all `d` kept directions (its spectrum is essentially flat).

## Held-out generalization (only if `holdout.json` is present)

Written by `holdout_eval.py`. Everything above is **in-sample** — measured on the tokens
the model was fit on. This section re-measures the residual on tokens from latent files
the run never saw, using the frozen trained means/bases and the same assignment rule.

- **in-sample residual %** — the `residual / total` from the section above.
- **held-out residual %** — the same ratio on unseen tokens. `total` uses the *frozen*
  trained `μ_global`, so the denominator is comparable.
- **generalization gap** = held-out − in-sample. Near 0 ⇒ the subspaces capture reusable
  structure. A large positive gap ⇒ overfitting (the bases memorise training tokens →
  lower `--dim` or increase the token count). The objective/token line is the same check
  in raw units (the minimised orthogonal residual), held-out vs the final training iter.

We do **not** re-split held-out variance into between/within: with frozen means the ANOVA
identity (total = between + within) no longer holds exactly, so only `residual / total` is
both well-defined and directly comparable to the in-sample report.

## Per-cluster table (sorted by size)

One row per cluster:

- **tokens** (`counts`) — number of tokens assigned.
- **share** (`w`) — that as a fraction of all tokens.
- **EVR(top-d)** (`evrⱼ`, `d > 0`) — fraction of *this* cluster's variance captured by its
  own `d`-dim subspace (per-cluster version of the global EVR).
- **d80** (`d > 0`) — dimensions needed for 80% of this cluster's captured variance; the
  per-cluster spectrum-flatness gauge.
- **cells@50%** — number of HEALPix cells that together hold half this cluster's tokens.
  **Low ⇒ spatially localized** (a geographic regime); high ⇒ spread across the globe.
  Ordering-independent.
- **owned** — number of cells where this cluster is the *most frequent* (dominant) label —
  how much "territory" it wins on the map.
- **files** — fraction of the sampled time steps (files) in which the cluster appears at
  all. ≈ 100% ⇒ persistent in time; low ⇒ intermittent.
- **tCV** — temporal coefficient of variation: std/mean of the cluster's enrichment across
  the 10 time deciles. **0 ⇒ constant rate over time; high ⇒ concentrated in certain
  periods** (seasonal/trend signature).
- **radius** (k-center only, when `model.pt["radius"]` is present) — the *max* distance
  from centroid to any member, k-center's native minimax objective. Contrast with `trace`
  (mean squared distance, what k-means/subspace_kmeans minimize). A large radius relative
  to trace flags an **outlier-driven cluster**.

How the spatial/temporal stats are built:

- `cell_counts` `[N_CELLS, K]` — token counts per (cell, cluster).
- `dominant` — argmax cluster per cell (drives the `owned` territory count; the full maps
  are in the temporal & spatial report).
- `cells@50%` — from each cluster's sorted-descending cumulative distribution across cells
  (the cells holding the first 50%).
- `enrich` `[10, K]` = `(decK / counts) / (dec_tot / T)` — ratio of a cluster's observed
  share in a time decile to its expected share if temporally flat. **1.0 = flat**, ≫ 1 =
  concentrated there. The decile is derived from the global file index (0…13020). (The
  temporal report uses the 12-month analogue of this for its seasonal profiles.)

## Subspace affinity between clusters (`d > 0` only)

- **Affinity(i, j)** = `‖Uᵢᵀ Uⱼ‖²_F / d ∈ [0, 1]` — mean squared cosine of the principal
  angles between the two subspaces. **1 = identical span, 0 = orthogonal.** Reported as
  median/mean/max over all off-diagonal pairs, plus a table of the most similar pairs.
- **mean-vector cosine** — cosine similarity between the two cluster *centroids* (direction
  only). Shown beside affinity to distinguish "same subspace orientation" from "same
  location in space."
- **Interpretation:** high-affinity pairs are merge candidates (K may be too large);
  uniformly low affinity means genuinely distinct regimes.

For `d = 0` this section is skipped (point clusters have no basis to compare).

## Temporal & spatial report (`temporal_report.md`)

The world map and the temporal/seasonal analysis moved out of the main report into a
dedicated `temporal_spatial.py` report. (The main report keeps the per-cluster `tCV` /
`files` columns — a compact temporal summary — and points here.) All maps share one color
scale from `affinity_ordered_colors` (spectral seriation of the affinity matrix → `turbo`,
so similar clusters share hues); the renderer (`worldmap.py`) adds **continent outlines**
(cached Natural Earth 110m coastlines) and a smooth **heatmap** — the per-cell RGB is
interpolated (`scipy.griddata`, linear + nearest-fill) onto a regular grid and
Gaussian-smoothed, then painted with `pcolormesh`, replacing the old speckled 12,288-pixel
scatter with a continuous, coast-aligned field. Interpolating RGB (not the categorical
cluster id) is valid because the colormap already makes neighbors similar. (The absolute hue
is arbitrary — it reflects only a cluster's rank in the Fiedler similarity order, with no
physical meaning; only the *transitions/groupings* carry information. The generated report's
"How the colors work" section spells out the full pipeline.)

- **Annual map** (`maps/map_annual.png`) — dominant cluster per cell across the full sample
  (NESTED ordering; grey = no data).
- **Monthly maps** (`maps/map_month_01.png` … `maps/map_month_12.png`) — dominant cluster per
  cell for each calendar month (grouped by the reconstructed date). Built from the run's
  existing `assignments.pt`, so no re-clustering.
- **Monthly enrichment** (table) — per cluster, `(share in month) / (average share)` for each
  of the 12 months. **1.0 = year-round**; ≫1 = concentrated in those months (a seasonal
  signature). Supersedes the old 10-decile table; `seasonality` = std/mean across months.
- **Month-to-month flip rate** — share of cells whose dominant cluster changes between
  consecutive months (low = stable geography; peaks = seasonal transitions). Plus a
  **Jan↔Jul** shift (winter vs summer) and the largest owned-cell-count deltas.

## Token sample / Configuration (not metrics)

The report also records the run config and the **sample fingerprint** — a SHA-1 over
tokens-per-file + seed + sorted sampled file ids. Runs sharing a fingerprint were
clustered on the identical token set and are directly comparable, even across algorithms;
the report prints a reproduce command.
