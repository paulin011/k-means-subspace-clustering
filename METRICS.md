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

## Convergence table

Per-iteration optimization history:

- **objective/token** (`obj_per_token`) — mean squared residual per token: squared
  orthogonal distance to the assigned affine subspace (for `d = 0`, squared distance to
  the centroid). Decreases monotonically and flattens at convergence.
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
  may be larger than needed).

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
- `dominant` — argmax cluster per cell (also drives the world map).
- `cells@50%` — from each cluster's sorted-descending cumulative distribution across cells
  (the cells holding the first 50%).
- `enrich` `[10, K]` = `(decK / counts) / (dec_tot / T)` — ratio of a cluster's observed
  share in a time decile to its expected share if temporally flat. **1.0 = flat**, ≫ 1 =
  concentrated there. The decile is derived from the global file index (0…13020).

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

## Most time-varying clusters

The top clusters by `tCV`, each with its full **D0…D9 enrichment profile** across the 10
time deciles (same `enrich` values; 1.00 = the cluster's average rate). A smooth ramp/peak
across deciles indicates seasonal or trend behavior rather than noise.

## World map (`dominant_cluster_map.png`)

A Mollweide projection coloring each of the 12,288 cells by its dominant cluster
(grey = no data), under **NESTED HEALPix ordering** (confirmed: geographically coherent
continent-scale regions under NESTED, incoherent stripes under RING). Colors come from
`affinity_ordered_colors`:

- build the K×K affinity matrix (subspace affinity for `d > 0`, else centroid cosine),
- take the **Fiedler vector** (2nd-smallest eigenvector of the normalized graph Laplacian)
  for 1-D spectral seriation, then map that ordering through the `turbo` colormap.

Effect: subspace-similar clusters get similar hues, so genuine spatial structure reads as
smooth gradients and only true noise stays speckled — legibility no longer rides on a
random hue shuffle (which falsely aliases finer sub-regions as "scatter" at large K).

## Token sample / Configuration (not metrics)

The report also records the run config and the **sample fingerprint** — a SHA-1 over
tokens-per-file + seed + sorted sampled file ids. Runs sharing a fingerprint were
clustered on the identical token set and are directly comparable, even across algorithms;
the report prints a reproduce command.
