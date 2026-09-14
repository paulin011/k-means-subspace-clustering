# Single-cluster probe — plan

**Goal.** Given a finished run and one cluster id, produce a *short* key report that answers
"what is this cluster, and what does it tell me about the embedding space" — using only the
information the existing reports do **not** already carry. The centrepiece is a **where map
with a working threshold**, plus the seasonal version of it. The end use is **outlier
discovery**: the probe must be able to hand back concrete `(date, cell)` cases, not just
another table of aggregates.

Proposed artefacts: `src/analysis/cluster_probe.py` → `runs/clustering/<run>/probe/c<NNN>.md`
plus `probe/shortlist.md` (the cluster selector). Written 2026-08-13, branch
`feat/overcluster-merge`. Every number quoted below was **measured on `v6_subspace_big_d64`
+ `runs/signatures/v6_d64_margin/` while writing this plan**, not assumed.

---

## 1. Why this, why now

The merge work (`merge_clusters.py`) made K a **budget knob**: v12 fits at K=256 and merges to
K=128 for a 0.57% objective gain, and the cascade showed there is no natural K at all (per-merge
cost rises at 100.0% of steps, no knee) — so K is chosen by what we can afford to *interpret*,
not by the data. Interpretation cost is linear in K and human attention is the scarce resource,
which makes a **per-cluster probe the unit of work**: everything below is O(1) per cluster and
lets one pick a handful out of 128 rather than reading 128 rows.

Scope discipline: the probe is a **reader**. It adds no data pass over `latents_2` and no new
producer script — every panel is computed from artefacts that already exist on disk (§8).

---

## 2. What already exists (do not rebuild)

| question | already answered by |
|---|---|
| how big, how flat, how truncated (`tokens`/`share`/`EVR`/`d80`) | `report.md` per-cluster table |
| how localized / how much territory (`cells@50%`, `owned`) | same |
| how bursty in time (`files@50%`, `tCV`) | same |
| who does it look like geometrically (`maxAff`, affinity table) | same |
| who contests its tokens (`margin`, `near%`, `runner_up`) | same, from `cluster_margin.csv` |
| which clusters are seasonal, and the global monthly/seasonal *dominant* maps | `temporal_report.md` |
| does the model generalize | `holdout.json` |

**The gap.** Every map in the repo is a **dominant-cluster** map: one colour per cell, all K
clusters at once. There is no map of *one* cluster, and the per-cluster spatial columns
(`cells@50%`, `owned`) are scalars — they say a cluster is spread out but never where. Nothing
in any report reaches the **token** level, so the outlier goal (`outlier-detection-goal` memory)
is not served at all: `residual_map.npy` is written and then only ever reduced away.

---

## 3. What the measurement pass found

These findings are the justification for each panel in §5, and for the rejections in §7.

### 3.1 The threshold problem is real, and it is a *mass* problem

Per-cell occupancy `f_j(cell) = P(cell carries label j)` over all 13,021 files (a 64 s pass over
`label_map.npy`). "Where does the cluster appear at all" spans **67 to 4,396 cells** (up to 36%
of the globe; mean 1,259). So a presence map is useless for exactly the clusters worth looking
at — the itinerant ones.

Worse, no *fixed* occupancy threshold works, because clusters are bimodal in peak occupancy
`maxf`: **77 of 128 clusters have maxf ≥ 0.9** (they own cells outright, permanently), while
**12 clusters never exceed 0.5 anywhere** — c13 peaks at 0.17. A τ=0.5 map is a solid blob for
the first group and *empty* for the second.

### 3.2 Two structurally different kinds of cluster

| kind | signature | example | reads as |
|---|---|---|---|
| **territorial** | `maxf`≈1, `near%` 1–5%, low residual | c27 (Arctic, 106 cells at f=1.000), c66 (Antarctic) | a fixed geographic region |
| **itinerant** | `maxf`<0.25, `near%`≈45%, high residual | c13 (equatorial band, 545 cells for half its mass), c36, c98 | a *moving weather state* |

`maxf` ↔ `near%` correlate at Spearman **−0.825**: crisp territory ⇒ crisp assignment. This is
one axis, not two, and it is already visible in the report via `near%` — so it is **not** a new
selector metric. Its value is that it dictates *how to draw the map* (§6).

### 3.3 The embedding's biggest outliers are documented extreme weather events

This is the headline. Normalising `residual_map.npy` per cell by **median/MAD** and taking the
top 0.01% (robust z > 8.19, 16,177 tokens), those tokens are strongly coherent in space and
time — **89%** of flagged cells on the worst timestep have a flagged HEALPix sibling (random:
3%), and **32.4%** are still flagged at the same cell at *t+1* (random: 0.010%, a 3,000×
enrichment). Grouping them into connected components (sibling ∪ t+1) gives 8,261 events; the
largest, by date and location:

| event | when | where | peak z | host clusters |
|---|---|---|---|---|
| #2, #7 | 2018-02-23 → 02-25 | Arctic, 84–87°N | 17.1 | c27, c14 |
| #3, #6 | 2022-03-17 → 03-19 | Antarctica, 75–81°S | 15.0 | c66, c77 |
| #8 | 2015-12-31 → 2016-01-01 | Arctic, 81°N 135°W | 14.7 | c27 |
| #1 | 2016-01-24 → 01-27 | SE Pacific, 27°S 101°W | 21.4 | c36, c31 |
| #5 | 2014-01-08 → 01-10 | NE Siberia, 69°N 135°E | 17.2 | c25 |

The top three coincide with the **February 2018 Arctic warming**, the **March 2022 Antarctic
heatwave** (the +38.5 °C Dome C record, 18 March 2022), and the **30 Dec 2015 North Pole
above-freezing event** — three of the most-documented anomalies in the ERA5 window, recovered
from latent residual alone with no physical variables involved. Treat this as a strong lead to
**spot-check against ERA5**, not as settled; but it is the evidence that the token-level tail is
where the interpretable signal lives, and it is why the probe must surface events, not just
means.

Caveat found while measuring: events fragment (#3/#6 and #2/#7 are each one physical event split
in two) because sibling adjacency (`cell >> 2`) only links within a HEALPix parent block and
never across block boundaries. A real neighbour query would merge them — see §9.

### 3.4 Two genuinely new per-cluster axes

Spearman |ρ| against every existing report column and against each other:

- **`zres`** — mean robust-normalized residual of the cluster's own tokens. Not the same thing
  as the report's residual/EVR: ρ = +0.27 with raw residual, −0.15 with `maxf`, +0.14 with
  `near%`. Range −0.62 (c6) … +1.37 (c36), 119/128 distinct. Answers "is this cluster hard
  *relative to where it sits*", which raw residual cannot, because per-cell residual spans 15.7×
  geographically.
- **`unmodelled = residual / persistence`** — the most orthogonal axis measured (max |ρ| 0.48,
  against `persist`; ρ = **+0.03** with raw residual). This is the "anomalous because
  unmodelled" vs "anomalous because rapidly changing" split that `CLAUDE.md` names as a goal and
  that nothing currently computes. Range 0.29 (c111 — hard only because volatile) … 1.45 (c66 —
  encoded poorly despite barely moving).

Also new but secondary: **`rival_km`**, the great-circle distance from a cluster's core to its
`runner_up`'s core. Rivals are usually neighbours — median **2,267 km** vs **10,105 km** for a
random pair, 44% within 2,000 km vs 6% at random — so near-ties are mostly a *map gradient*.
But **18 of the 72 contested clusters** (`near%` > 30) have a rival **> 4,000 km away**, e.g.
c26↔c40 at 16,532 km (near-antipodal) and c64↔c7 at 14,838 km with seasonality 3.2 — a
cross-hemisphere seasonal mirror. Those are real embedding-space confusions rather than
geographic blur, and nothing currently distinguishes the two cases.

### 3.5 No cluster is an isolated blob

`iso = ‖μⱼ − μ_global‖² / trace[j]` maxes out at **0.25** (c111) across all 128 clusters: the
between-cluster offset is at most a quarter of a cluster's own internal spread. Combined with the
33% near-tie rate and the knee-free merge cascade, this closes the question — **"outlier
cluster" is not a meaningful category in this embedding**. Outliers exist at the *token* level.
The probe should therefore not try to rank clusters by outlier-ness *per se*; it ranks them by
**how many outlier events they host** (§4) and then goes to the tokens.

---

## 4. Deliverable A — the shortlist (`--rank`)

One page, one table, sorted; the entry point for "which cluster should I look at". Columns are
only the ones that survived the redundancy audit (all mutual |ρ| ≤ 0.59):

| column | source | why it earns a slot |
|---|---|---|
| `tokens`, `near%` | existing | size and contestedness |
| `zres` | new (§3.4) | hard relative to its own geography |
| `unmodelled` | new (§3.4) | separates unmodelled from merely volatile |
| `events` | new (§3.3) | count of extreme events hosted — the direct outlier axis |
| `τ50`, `core cells` | new (§6) | how to read its map; territorial vs itinerant at a glance |
| `rival`, `rival_km` | new (§3.4) | whether its confusion is local blur or remote |
| `seasonality` | existing (temporal report) | reproduced here so the shortlist stands alone |

Deliberately excluded: `burstiness` (ρ 0.97 with `seasonality`), `persistence` alone (ρ 0.80
with residual), `maxf` (ρ −0.83 with `near%`), `rival_overlap` (ρ −0.73 with `maxf`; `rival_km`
is the non-degenerate version), `mu_dev`/`iso` (ρ 0.83 with each other, and §3.5 makes them a
one-line global statement rather than a per-cluster column).

Default sort: `events`, descending — that is the shortlist for the outlier goal. `--sort` picks
another column. Also print the three or four **preset shortlists** the audit suggests, each 5
rows: *hardest* (`zres`), *most unmodelled*, *most itinerant* (lowest `τ50`), *most remotely
confused* (`rival_km` among `near%`>30).

Follow the `report-metric-audit` rule when implementing: print `len(set(col))` for every column
and drop anything under ~40 distinct values of 128. Measured now: `zres` 119, `unmodelled` 112,
`rival_km` 99, `τ50` 108, `events` — currently only 27 distinct at the 0.01% cut, so **loosen the
event threshold to ~0.1%** (or report a rate rather than a share) before shipping that column.

---

## 5. Deliverable B — the per-cluster key report

Target: **one screen of text + 2 figures**. Five panels, in this order. Anything not listed here
is deliberately out (§7).

**P0 — identity line.** One sentence generated from the numbers: size and share; core extent
(`τ50` cells); weighted centroid lat/lon; latitude band; the zonal-band flag; territorial vs
itinerant; the headline diagnostics. Longitude must use the **circular** mean and be suppressed
when the circular concentration `R` is low — measured `R`: c109 = 1.00 (a single Pacific spot,
lon meaningful), c13 = 0.44 (equatorial band), c27 = 0.07 (wraps the pole, lon meaningless).
Report `R` with |lat| so a polar ring is not mistaken for a zonal band.

**P1 — the occupancy map with the coverage slider.** The centrepiece; see §6. A single figure
with three sub-panels at 50 / 80 / 95% mass coverage, reusing `worldmap.render_world_map`'s
coastlines and Mollweide projection with a sequential colormap on `f_j` instead of the
categorical palette.

**P2 — seasonal occupancy, as an anomaly.** Four maps of `f_j(cell | season) − f_j(cell)`, i.e.
*departure from the cluster's own annual field*, not the raw seasonal field. Rationale: the
absolute seasonal maps mostly restate P1, whereas the difference isolates the migration. Cheap
(already accumulated), and the signal is enormous — per-cell seasonal occupancy amplitude reaches
0.999 (99.9th pct 0.81). Add the 4 enrichment numbers (DJF/MAM/JJA/SON) as a one-line strip.

**P3 — the outlier events table.** Top ~10 events hosted by this cluster: date range, number of
steps, centroid lat/lon, peak robust-z, **and the raw residual next to it** so a reader can tell
a genuinely large residual from a small one in a low-variance cell (§7 explains why this column
is mandatory). This is the panel that converts a cluster into cases someone can look up.

**P4 — the rival panel.** `runner_up`, `runner_up_frac`, `near%`, `rival_km`, and one sentence
classifying it: *local blur* (< 2,000 km, the common case) or *remote confusion* (> 4,000 km,
18/72 contested clusters — say which hemisphere/season pairing it looks like). No extra figure;
optionally shade the rival's core outline onto P1.

Every panel gets the "How to read this" preamble the other reports use (`always-write-documentation`
memory), naming the exact array each number came from.

---

## 6. The threshold, done properly

**The failure.** A raw occupancy threshold cannot be shared across clusters (§3.1): τ=0.5 shows
a blob for c27 and nothing at all for c13.

**The fix — slide *mass coverage*, not frequency.** Ask for the smallest set of cells holding a
fraction *q* of the cluster's tokens, and derive τ from it:

```
sort f_j descending → cumulative / total → τ_q = f at the crossing of q
```

This is the highest-density-region construction, it is scale-free, it degrades gracefully for
both cluster kinds, and it has a fixed reference point the reader can hold onto — the same
property that made `cells@50%` and `files@50%` work. It is also exactly `cells@50%` generalized
to arbitrary *q*, so P1 is continuous with the existing report rather than a new vocabulary.

Measured, showing why one fixed τ could never have worked:

| cluster | maxf | τ50 (cells) | τ80 (cells) | τ95 (cells) |
|---|---|---|---|---|
| c27 territorial | 1.00 | 1.000 (106) | 0.997 (170) | 0.980 (201) |
| c114 | 0.99 | 0.845 (88) | 0.503 (157) | 0.072 (270) |
| c122 | 1.00 | 0.062 (262) | 0.023 (691) | 0.007 (1301) |
| c13 itinerant | 0.17 | 0.042 (545) | 0.017 (1298) | 0.006 (2283) |

Note c122: `maxf`=1.00 yet `τ50`=0.062 — it permanently owns a couple of cells while its mass is
smeared over 262. **`τ50` is the honest territoriality scalar, `maxf` is not**; report `τ50`.

**Slider mechanics.** Default is the static 3-panel ladder (50/80/95%) in Markdown — no
JavaScript, consistent with every other report here. `--coverage 0.5,0.8,0.95` overrides the
stops. Optional `--html` writes a self-contained page that pre-renders ~9 PNGs (q = 0.1 … 0.95)
and swaps them from an `<input type=range>`; that is a genuine slider at ~9× the render cost
(~0.8 s/map, so ~8 s) and no new dependency. Ship the ladder first; the HTML is a small add-on
once the numbers are trusted.

---

## 7. Rejected — with the measurement that killed each

Recording these so they are not re-proposed. (`report-metric-audit` discipline.)

- **Diurnal / hour-of-day panel.** Measured null. Across all 128 clusters the largest deviation
  of any UTC-hour share from 0.25 is **0.023**; only **2.6%** of cells change dominant cluster
  between 00Z and 12Z, against **42.8%** between DJF and JJA. Per-cell diurnal occupancy
  amplitude: 99.9th pct **0.07** vs the seasonal **0.81**. The 6-hourly latents carry a seasonal
  cycle and essentially no diurnal one. Do not build the panel; one line in the report stating the
  null is worth more.
- **Per-cell rank normalisation for outliers.** Degenerate by construction: ranking within a cell
  and cutting at a global quantile selects ~the same number of timesteps *for every cell*, so it
  can never say a cell is unusually anomalous. It disagrees with robust-z on top-10 host clusters
  despite Spearman 0.804 overall. Use **median/MAD z**.
- **Plain mean/std z** (what `signatures.npz` currently stores). It over-selects low-variance
  cells: c66/c27/c0/c77 — crisp polar clusters with `zres` ≈ 0 and persistence 559–1,131, the
  lowest in the run — took 4 of the top 10 host slots. MAD-z is robust to the very tail being
  detected. Compute median/MAD in the probe (seconds) and **always print raw residual beside z**
  so a small-absolute anomaly in a quiet cell is visible as such.
- **`rival_overlap`** (occupancy cosine with the runner-up): ρ −0.73 with `maxf`, because a
  cluster that owns its cells outright has ~zero overlap with everything by definition. Replaced
  by `rival_km` (ρ −0.36).
- **`burstiness`** (temporal CV of the file mix): ρ 0.97 with `seasonality`. Pick one.
- **Per-cluster persistence as its own column**: ρ 0.80 with residual. Keep only the *ratio*
  (`unmodelled`, ρ 0.03 with residual).
- **Silhouette and friends**: already ruled out in `METRICS.md` — wrong geometry for subspaces;
  `margin`/`near%` is the subspace-native analogue and already exists.
- **Per-cluster `mu_dev` / `iso`**: §3.5 — the finding is global and single-sentence, not a column.

---

## 8. Implementation

**Inputs — all already on disk, no pass over `latents_2`:**

| array | from | used for |
|---|---|---|
| `label_map.npy [13021,12288]` int16 | `file_signature.py` | occupancy `f_j`, seasonal split, event host attribution |
| `residual_map.npy [13021,12288]` f32 | same | `zres`, events |
| `margin_map.npy` f16, `signatures.npz` | same | near-tie context, `datetime_hour`, `cluster_runner_up` |
| `model.pt`, `assignments.pt` | the run | counts, means, `config`, schema check |
| `err_persist.npy [13020,12288]` | `persistence_error.py` | `unmodelled` |

**Dependency to state loudly in the README:** the probe needs a `runs/signatures/<run>/` for the
target run. v6 has two; **v12 has none**, so probing the merged run requires
`file_signature.py --dir runs/clustering/v12_k256to128_d64` first (~13 min, single GPU). Resolve
the signature dir the way `analyze_clusters.py --margins` already does — scan
`runs/signatures/*/manifest.json` for `model_dir == --dir` — and fail with that exact command
line if none matches.

**Cost, measured:** occupancy + seasonal accumulation over the full label map = **64 s**;
residual/event pass ≈ the same; map render ~0.8 s each. So `--rank` is a ~2-3 min job for all K,
and a single-cluster report is seconds once cached. **Cache the shared `[12288,K]` occupancy
tensor, the seasonal `[4,12288,K]`, and the per-cell median/MAD** to `probe/cache.npz` keyed by
the run's `sample_fingerprint`, so the second cluster costs nothing.

**CLI**

```bash
python3 src/analysis/cluster_probe.py --dir runs/clustering/v6_subspace_big_d64 --rank
python3 src/analysis/cluster_probe.py --dir runs/clustering/v6_subspace_big_d64 --cluster 13
python3 src/analysis/cluster_probe.py --dir <run> --cluster 13 --coverage 0.5,0.8,0.95 --html
```

Run from the repo root; keep the `sys.path.insert(...)` bootstrap above `from common.worldmap
import ...` per the layout contract. Algorithm-agnostic — read the `cluster_io` schema only, so a
merged run (`config["method"] == "subspace_kmeans_merged_ward"`) works unchanged, as
`analyze_clusters.py` and `holdout_eval.py` already do.

**Additions to `worldmap.py`** (keep geometry in one place): `render_scalar_map(values, out_png,
*, cmap, vmin, vmax, mask)` — the continuous-field twin of `render_world_map`, reusing the same
griddata + coastline path — and `core_region(f_j, q) -> (tau, cells)`.

**Docs:** `README.md` + `CLAUDE.md` entries for the new script, and a `docs/METRICS.md` section
defining `τq`, `zres`, `unmodelled`, `events`, `rival_km` with their formulas and their measured
correlation against the existing columns.

---

## 9. Validation (the checks the implementation must pass)

1. `f_j.sum(0) * n_files == counts_j` from the full-dataset label map (exact, integer).
2. `τ_q`'s selected cells hold ≥ q of the cluster mass, and `q=0.5` reproduces the report's
   `cells@50%` when computed on the same file set — a direct cross-check against a shipped column.
3. Seasonal occupancy fields, weighted by files per season, sum back to the annual field.
4. Event extraction: with a **real HEALPix neighbour** query replacing the sibling proxy, the
   2022-03-17→19 Antarctic components (#3, #6) must merge into one event, and likewise
   2018-02-23→25 (#2, #7). This is the test that the adjacency fix worked.
5. The shortlist's `zres` recomputed for the whole population reproduces
   `model['final_obj_per_token']` after undoing the normalisation (the same
   count-weighted-mean invariant `file_signature.py` already uses).
6. Distinct-value count printed for every shortlist column; anything < 40/128 is dropped or
   re-thresholded before shipping.

---

## 10. Open questions

**STATUS 2026-08-13: implemented as `src/analysis/cluster_probe.py`** (+ `healpix_nest_neighbours`,
`core_region`, `render_scalar_map` in `worldmap.py`; documented in README / CLAUDE.md /
METRICS.md). All five §9 checks pass on **both** v6 and v12. Three things resolved against
what is written above:

- **Event threshold → 0.1%, not 0.01%** — but **there is no knee**, contrary to what this
  section assumed. Measured on v6 (`--event-sweep`), the fraction of flagged tokens sitting
  in multi-token events climbs smoothly with the cut: 75.3% at 0.01% → 79.5% at 0.05% →
  81.0% at 0.1% → 84.8% at 0.5%, with distinct `events` values 37 / 70 / 84 / 120 of 128.
  So the threshold is a resolution-vs-tail-purity judgement, not a measured optimum: 0.1%
  is the loosest cut that still keeps the tail genuinely extreme (z ≥ 5.58) while clearing
  the distinct-value rule. `--event-pct` re-cuts it.
- **`events` is closer to `zres` than the plan assumed** — Spearman +0.75 on v6, above the
  0.7 line the audit uses elsewhere. It is kept because it is the sort key and it still
  surfaces different clusters at the top (v6 c127: `events` rank 3, `zres` only +0.20), but
  it is *not* the independent axis §3.4 implied. Re-check this if the threshold moves.
- **The `--html` slider shipped after all** (deferred once as "ship the ladder first", then
  built because a slider was the original ask): `--html` writes a self-contained
  `c<NNN>_slider.html`, `--html-stops` panels base64-embedded, ~2.5 MB.

And the cross-run question below is **answered**: re-running on v12 (a genuinely different
partition, NMI 0.684 vs v6), **5 of the top 10 events land on the identical peak cell and 3
on the same day**, with the 2018-02-22 Arctic event at peak z 22.4 (v6) vs 22.6 (v12) and
852 vs 798 tokens. `file_signature.py` independently names 2018-02-24T18 as v12's most
anomalous timestep. The events are a property of the embedding, not of the fit.

**§3.3 was verified against the literature (2026-08-13) and is partly overstated as written
above.** Four of the largest events do match documented extremes on the date — Feb-2018
Arctic SSW, 30-Dec-2015 North Pole thaw, 23-Aug-2016 Arctic cyclone (3-day lag), Mar-2022
East Antarctic heatwave — but "**the embedding's biggest outliers are documented extreme
events**" does not survive the control:

- The **Mar-2022 Antarctic heatwave, the largest surface temperature anomaly ever recorded on
  Earth, is only the 14th percentile of raw residual — below the global median** (raw 1,074,
  cell baseline 278). It is a 3.9× excursion *for that cell* and nothing at all in absolute
  latent terms.
- The `z` and raw rankings are biased in opposite directions: top-30 by z is 50% polar
  (14% of cells) with median cell baseline 744; top-30 by raw is 0% polar with baseline
  2,267. MAD/median is near-constant globally (0.105–0.134 by zone), so ratio-to-baseline
  reproduces the z answer (40% polar) — the dichotomy is **absolute vs locally-relative**,
  not an artefact of MAD.
- `z` asks "unusual for this cell", which is the same locally-normalised question a
  meteorological record asks. The agreement with the news record is therefore **partly
  built into the question**, not purely an emergent property of the embedding.
- Two of the strongest peaks do **not** resolve to a documented event: 2019-09-30→10-03 off
  Tokyo (extreme on every scale: z 26.6, raw 5,299 = 99.998th pct; Mitag tracked up the Sea
  of Japan side, not there) and 2019-01-25 offshore Perth (Australia's hottest January
  nationally, but BoM records Perth's *coolest* January in a decade).

Consequence, already implemented: P3 prints peak z, peak raw residual **and its global
percentile**, so a large z in a quiet cell can never again be read as a large anomaly. The
open ERA5 spot-check is now specifically about the unresolved peaks, not the matched ones.

- **Event threshold.** 0.01% gives 8,261 components, 67% of them singletons and max size 32 — too
  sparse for a per-cluster events column (27 distinct values). Sweep 0.01 / 0.05 / 0.1 / 0.5% once
  and pick the knee in "fraction of flagged tokens inside multi-token events".
- **Verify the events against ERA5.** The three headline matches (§3.3) are date/location
  coincidences with famous anomalies, which is suggestive but not proof. One spot check against
  ERA5 2 m temperature would settle it and, if it holds, is the strongest single result the latent
  analysis has produced.
- **Probe the merged run.** Everything above is measured on v6. v12 is the better partition
  (−0.57% objective, NMI 0.684 vs v6) and the resolution knob's whole point is that K is a budget
  — re-run `--rank` on v12 once its signature run exists and check whether the itinerant/
  territorial split and the event hosts survive re-clustering. If the same events surface under a
  different partition, they are properties of the embedding, not of the fit.
