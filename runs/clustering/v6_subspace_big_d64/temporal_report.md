# Temporal & spatial report — `runs/clustering/v6_subspace_big_d64`

*Generated 2026-07-31 15:39 by `temporal_spatial.py`. K=128 affine subspaces (d=64), 86,016,000 tokens from 7000 files. Maps reuse this run's existing assignments (no recomputation).*

## Time axis

The latents are ERA5 atmospheric states at **6-hourly cadence** (00/06/12/18 UTC — 4 per day). Each file stores only an integer sample index (`idx`), **not a timestamp**, so the calendar date is reconstructed positionally:

> `datetime = 2014-01-01 00:00 UTC + idx × 6 h`

The 13,021 files therefore span **file 0 = 2014-01-01 00:00** through **file 13020 = 2022-11-30 00:00**. (A complete 9-year 2014→2022 ERA5 record would run to 2022-12-31 18:00 — about 127 more steps — so the last ~month of 2022 is treated as not fully covered.) Monthly and seasonal grouping below uses this reconstructed date.

Sampled files per calendar month: Jan 623, Feb 542, Mar 594, Apr 590, May 589, Jun 568, Jul 621, Aug 590, Sep 595, Oct 584, Nov 576, Dec 528.

Sampled files per season: DJF 1693, MAM 1773, JJA 1779, SON 1755.

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

One dominant-cluster map per calendar month (same color scale as the annual map), revealing the seasonal cycle. Each cell is colored by its most frequent cluster among that month's tokens. Image files are in `runs/clustering/v6_subspace_big_d64maps/`


## Seasonal maps

The same dominant-cluster field aggregated into the four meteorological seasons (Northern Hemisphere convention; DJF groups December with the following January and February). Averaging three months each smooths the month-to-month noise and makes the broad seasonal regimes easier to read than the individual monthly maps. Same color scale throughout.

**Winter (Dec-Jan-Feb)** (1693 files, 20,803,584 tokens)

![Winter (Dec-Jan-Feb)](maps/map_season_DJF.png)

**Spring (Mar-Apr-May)** (1773 files, 21,786,624 tokens)

![Spring (Mar-Apr-May)](maps/map_season_MAM.png)

**Summer (Jun-Jul-Aug)** (1779 files, 21,860,352 tokens)

![Summer (Jun-Jul-Aug)](maps/map_season_JJA.png)

**Autumn (Sep-Oct-Nov)** (1755 files, 21,565,440 tokens)

![Autumn (Sep-Oct-Nov)](maps/map_season_SON.png)


### Season-to-season stability

Share of cells whose **dominant cluster changes** between consecutive seasons (cyclic, so the final row closes the loop SON→DJF). Lower than the monthly flip rate because three-month averaging absorbs short transient regimes.

| transition | cells changing dominant cluster |
|---|---|
| DJF→MAM | 17.4% |
| MAM→JJA | 33.1% |
| JJA→SON | 17.1% |
| SON→DJF | 33.0% |

## Most seasonal clusters

This table identifies which clusters behave like **seasonal regimes** (monsoon, sea-ice, snow cover) versus **year-round geographic regimes**.

**Enrichment** of each cluster per month = `(cluster share in month) / (its average share)`. Read it as a multiplier on the cluster's baseline presence: **1.0 = present at its normal level** that month, **2.0 = twice as concentrated** as usual, **0.5 = half**. A cluster that is genuinely year-round sits near 1.0 across every column; a cluster that erupts in summer and vanishes in winter swings far above and below 1.0. Because the metric is normalized by each cluster's own average, a small cluster and a large cluster are directly comparable: the number is about *timing*, not *size*.

**Seasonality** = `std / mean` of those 12 monthly enrichments (the coefficient of variation). It collapses the whole year into one score for *how peaked* a cluster is: **~0 = flat / aseasonal** (the same every month), **higher = more concentrated** into particular months. Rows are sorted by this score, so the clusters at the top are the most strongly seasonal ones in the run, and scanning their monthly columns tells you *when* each one peaks. An en-dash marks a month with no sampled data, which is excluded from the score rather than counted as zero.

The four right-hand columns repeat the same enrichment aggregated into seasons (DJF/MAM/JJA/SON) for a coarser, lower-noise read of the same signal: a cluster peaking in a single season shows one column well above 1.0 and the rest below.

| cluster | seasonality | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 1.13 | 0.00 | 0.00 | 0.01 | 0.08 | 0.79 | 2.27 | 2.75 | 2.79 | 2.10 | 0.85 | 0.12 | 0.02 | 0.01 | 0.29 | 2.61 | 1.03 |
| 82 | 1.02 | 0.01 | 0.01 | 0.01 | 0.11 | 0.77 | 1.94 | 2.42 | 2.52 | 2.32 | 1.33 | 0.30 | 0.07 | 0.03 | 0.30 | 2.30 | 1.33 |
| 7 | 1.01 | 2.68 | 2.96 | 2.12 | 0.97 | 0.27 | 0.16 | 0.11 | 0.09 | 0.15 | 0.27 | 0.71 | 1.65 | 2.45 | 1.12 | 0.12 | 0.37 |
| 29 | 0.96 | 2.53 | 2.46 | 1.86 | 0.96 | 0.28 | 0.02 | 0.00 | 0.00 | 0.02 | 0.50 | 1.49 | 2.05 | 2.36 | 1.04 | 0.01 | 0.66 |
| 73 | 0.91 | 0.05 | 0.02 | 0.04 | 0.17 | 0.73 | 1.34 | 1.87 | 2.35 | 2.47 | 1.83 | 0.77 | 0.22 | 0.09 | 0.31 | 1.86 | 1.70 |
| 67 | 0.89 | 2.50 | 2.39 | 2.00 | 1.32 | 0.52 | 0.18 | 0.10 | 0.07 | 0.09 | 0.28 | 0.91 | 1.75 | 2.23 | 1.28 | 0.12 | 0.42 |
| 104 | 0.89 | 0.07 | 0.03 | 0.04 | 0.21 | 0.75 | 1.48 | 1.97 | 2.37 | 2.32 | 1.61 | 0.78 | 0.25 | 0.11 | 0.33 | 1.95 | 1.58 |
| 102 | 0.87 | 0.09 | 0.11 | 0.29 | 0.62 | 1.25 | 1.84 | 2.50 | 2.32 | 1.64 | 0.73 | 0.26 | 0.17 | 0.12 | 0.72 | 2.23 | 0.89 |

## Month-to-month stability

Share of cells whose **dominant cluster changes** between consecutive months (low = stable geography; peaks mark the seasonal transitions). Over the 12 comparable month-pair(s): min **6.2%**, mean **12.9%**, max **21.5%**.

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
| Dec→Jan | 12.5% |

**Jan ↔ Jul** (winter vs summer hemisphere): **45.0%** of cells change dominant cluster. Largest cluster shifts (owned-cell count, Jul − Jan): 82: +353, 88: +305, 6: +278, 95: +248, 24: +235 ….

## Jan → Jul change maps


![Jan to Jul changes by July cluster](maps/map_change_jul_dest.png)

**By July destination.** The same flipped cells, now colored by each cell's **July** cluster (grey = held its cluster). Colors match the monthly and seasonal maps, so a changed region's hue tells you which July regime it joined; where a coherent area shares one color, a whole region migrated into the same summer cluster together.
![Jan to Jul transitions](maps/map_change_jan_jul.png)

**By transition.** The 8 most common dominant-cluster flows between January and July. Each color is one **A -> B flow** (cluster A in Jan becomes cluster B in Jul); grey cells either held their cluster or flipped via a rarer transition outside the top 8. Cells sharing a color all made the *same* move, so a coherent colored region is a single seasonal regime shift acting on a whole area. Note these rows are single A -> B flows, so they will not line up with the net per-cluster deltas above (which pool every source cluster).

| color | Jan cluster -> Jul cluster | cells |
|---|---|---|
| 0 | 30 -> 126 | 194 |
| 1 | 4 -> 6 | 185 |
| 2 | 43 -> 95 | 158 |
| 3 | 52 -> 82 | 155 |
| 4 | 99 -> 65 | 132 |
| 5 | 53 -> 105 | 132 |
| 6 | 121 -> 34 | 110 |
| 7 | 7 -> 75 | 105 |

## Interpretation notes

- **Stable geography + month-to-month flips near the minimum** ⇒ clusters are **geographic regimes** (region/surface type) that hold their territory year-round.
- **Clusters with high seasonality and a single summer/winter peak** ⇒ **seasonal regimes** (e.g. monsoon, sea-ice, snow cover); find them in the table above.
- **Jan↔Jul changes concentrate in one hemisphere** ⇒ a hemispheric seasonal cycle (opposite phases north/south).
- Monthly maps share one color scale, so a hue *appearing* in a region month-to-month is a real shift, not a recoloring.

*See the main clustering report (`runs/clustering/v6_subspace_big_d64report.md`) for convergence, variance decomposition, per-cluster spatial/temporal columns, and subspace affinity.*