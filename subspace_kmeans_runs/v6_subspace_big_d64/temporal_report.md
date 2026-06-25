# Temporal & spatial report — `subspace_kmeans_runs/v6_subspace_big_d64`

*Generated 2026-06-25 17:16 by `temporal_spatial.py`. K=128 affine subspaces (d=64), 86,016,000 tokens from 7000 files. Maps reuse this run's existing assignments (no recomputation).*

## Time axis

The latents are ERA5 atmospheric states at **6-hourly cadence** (00/06/12/18 UTC — 4 per day). Each file stores only an integer sample index (`idx`), **not a timestamp**, so the calendar date is reconstructed positionally:

> `datetime = 2014-01-01 00:00 UTC + idx × 6 h`

The 13,021 files therefore span **file 0 = 2014-01-01 00:00** through **file 13020 = 2022-11-30 00:00**. (A complete 9-year 2014→2022 ERA5 record would run to 2022-12-31 18:00 — about 127 more steps — so the last ~month of 2022 is treated as not fully covered.) Monthly grouping below uses this reconstructed date.

Sampled files per calendar month: Jan 623, Feb 542, Mar 594, Apr 590, May 589, Jun 568, Jul 621, Aug 590, Sep 595, Oct 584, Nov 576, Dec 528.

## Annual map

![Annual dominant cluster](maps/map_annual.png)

Dominant cluster per HEALPix cell across the full sample (NESTED ordering; grey = no data). Continent outlines are Natural Earth 110m; the field is the per-cell RGB (from spectral-ordered cluster colors) interpolated to a smooth heatmap. **Subspace-similar clusters share hues**, so coherent regions read as gradients.

## How the colors work

Cluster colors are **not random** — they are assigned so similar clusters get similar colors, which is what makes the maps readable (and the heatmap interpolation legitimate). The pipeline (`worldmap.py`):

1. **Affinity matrix (K×K)** — for each cluster pair, how similar they are. With subspaces (d>0) it is the mean squared cosine of the principal angles between the two bases, `‖UᵢᵀUⱼ‖²_F / d ∈ [0,1]` (1 = same subspace, 0 = orthogonal); for point clusters (d=0) the cosine of the two centroids. High affinity ⇒ nearby / overlapping structure.
2. **Spectral seriation** — take the *Fiedler vector* (the 2nd-smallest eigenvector of the normalized graph Laplacian of that affinity matrix): the classic 1-D embedding that places similar items next to each other. Ranking clusters by their Fiedler coordinate gives a single similarity-ordered sequence.
3. **Colormap** — that rank (0…K−1) is mapped through the smooth perceptual `turbo` colormap, so the order of colors along the rainbow exactly follows the similarity order: neighboring hues = affinity-similar clusters.
4. **Heatmap** — each cell takes its dominant cluster's RGB, and it is the **RGB** (not the integer cluster id) that is interpolated across the map. Interpolating a categorical id would be meaningless, but because step 3 already gave similar clusters similar RGB, blending two neighbors yields a sensible in-between color.

Reading the maps:
- Genuine structure shows up as **smooth gradients** (a region slowly shading into a neighboring hue); only true salt-and-pepper noise stays speckled. A *random* hue assignment would put a sharp color jump at every boundary and alias fine sub-regions as spurious "scatter," especially at large K.
- The **absolute hue is arbitrary** — blue vs red just reflects a cluster's position in the Fiedler order, which has no physical meaning; only the *transitions* and *groupings* carry information. So two months can look similarly hued where the same cluster family dominates, even if the exact dominant cluster differs.

## Monthly maps

One dominant-cluster map per calendar month (same color scale as the annual map), revealing the seasonal cycle. Each cell is colored by its most frequent cluster among that month's tokens.

**January** (623 files, 7,655,424 tokens)

![January](maps/map_month_01.png)

**February** (542 files, 6,660,096 tokens)

![February](maps/map_month_02.png)

**March** (594 files, 7,299,072 tokens)

![March](maps/map_month_03.png)

**April** (590 files, 7,249,920 tokens)

![April](maps/map_month_04.png)

**May** (589 files, 7,237,632 tokens)

![May](maps/map_month_05.png)

**June** (568 files, 6,979,584 tokens)

![June](maps/map_month_06.png)

**July** (621 files, 7,630,848 tokens)

![July](maps/map_month_07.png)

**August** (590 files, 7,249,920 tokens)

![August](maps/map_month_08.png)

**September** (595 files, 7,311,360 tokens)

![September](maps/map_month_09.png)

**October** (584 files, 7,176,192 tokens)

![October](maps/map_month_10.png)

**November** (576 files, 7,077,888 tokens)

![November](maps/map_month_11.png)

**December** (528 files, 6,488,064 tokens)

![December](maps/map_month_12.png)


## Most seasonal clusters

Enrichment of each cluster per month = `(cluster share in month) / (its average share)`. **1.0 = present year-round**; values ≫ 1 mark the months where the cluster concentrates (a seasonal signature). Sorted by seasonality (std/mean across the 12 months).

| cluster | seasonality | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1.18 | 0.00 | 0.00 | 0.01 | 0.08 | 0.79 | 2.28 | 2.75 | 2.80 | 2.10 | 0.85 | 0.12 | 0.02 |
| 82 | 1.07 | 0.01 | 0.01 | 0.01 | 0.11 | 0.77 | 1.94 | 2.42 | 2.53 | 2.32 | 1.33 | 0.30 | 0.07 |
| 7 | 1.06 | 2.68 | 2.95 | 2.12 | 0.96 | 0.27 | 0.16 | 0.11 | 0.09 | 0.15 | 0.27 | 0.71 | 1.64 |
| 29 | 1.00 | 2.50 | 2.43 | 1.83 | 0.95 | 0.28 | 0.02 | 0.00 | 0.00 | 0.02 | 0.49 | 1.47 | 2.02 |
| 73 | 0.95 | 0.05 | 0.02 | 0.04 | 0.17 | 0.73 | 1.34 | 1.87 | 2.35 | 2.47 | 1.83 | 0.77 | 0.22 |
| 67 | 0.93 | 2.52 | 2.41 | 2.01 | 1.34 | 0.53 | 0.18 | 0.10 | 0.07 | 0.09 | 0.29 | 0.92 | 1.77 |
| 104 | 0.93 | 0.07 | 0.03 | 0.04 | 0.21 | 0.75 | 1.48 | 1.97 | 2.38 | 2.32 | 1.61 | 0.78 | 0.25 |
| 102 | 0.90 | 0.09 | 0.11 | 0.29 | 0.62 | 1.25 | 1.84 | 2.50 | 2.32 | 1.64 | 0.73 | 0.26 | 0.17 |

## Month-to-month stability

Share of cells whose **dominant cluster changes** between consecutive months (low = stable geography; peaks mark the seasonal transitions). Over all 11 month-pairs: min **6.2%**, mean **13.0%**, max **21.5%**.

| transition | cells changing dominant cluster |
|---|---|
| Jan→Feb | 8.2% |
| Feb→Mar | 10.4% |
| Mar→Apr | 13.4% |
| Apr→May | 21.5% |
| May→Jun | 15.3% |
| Jun→Jul | 10.0% |
| Jul→Aug | 6.2% |
| Aug→Sep | 8.8% |
| Sep→Oct | 15.3% |
| Oct→Nov | 18.1% |
| Nov→Dec | 15.1% |

**Jan ↔ Jul** (winter vs summer hemisphere): **45.0%** of cells change dominant cluster. Largest cluster shifts (owned-cell count, Jul − Jan): 82: +353, 88: +305, 6: +278, 95: +248, 24: +235 ….

## Interpretation notes

- **Stable geography + month-to-month flips near the minimum** ⇒ clusters are **geographic regimes** (region/surface type) that hold their territory year-round.
- **Clusters with high seasonality and a single summer/winter peak** ⇒ **seasonal regimes** (e.g. monsoon, sea-ice, snow cover); find them in the table above.
- **Jan↔Jul changes concentrate in one hemisphere** ⇒ a hemispheric seasonal cycle (opposite phases north/south).
- Monthly maps share one color scale, so a hue *appearing* in a region month-to-month is a real shift, not a recoloring.

*See the main clustering report (`subspace_kmeans_runs/v6_subspace_big_d64/report.md`) for convergence, variance decomposition, per-cluster spatial/temporal columns, and subspace affinity.*