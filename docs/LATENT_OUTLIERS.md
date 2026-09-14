# What the latent outliers are — verified findings

Record of the outlier analysis run on `v6_subspace_big_d64` + `runs/signatures/v6_d64_margin/`
with `cluster_probe.py`, and of the literature check that followed. Written 2026-08-13.

**One-line summary.** The embedding does register real meteorological extremes as coherent
geometric departures, and four of the largest match documented events on the date — but the
size of a latent residual is **not** proportional to the size of the physical anomaly, and the
answer you get depends entirely on whether you normalise per location. Both facts matter for
any downstream use of latent hardness as a selection signal.

---

## 1. Method

`residual_map.npy [13021, 12288]` holds the orthogonal residual
`‖x − μⱼ‖² − ‖Uⱼᵀ(x − μⱼ)‖²` of every token under its assigned cluster. Per-cell raw residuals
are not comparable (the per-cell mean spans 15.7× geographically), so each is scored against
its own temporal norm:

```
z = (residual − median_cell) / (1.4826 · MAD_cell)
```

Median/MAD rather than mean/std because mean/std is not robust to the very tail being
detected; **not** per-cell rank, which is degenerate (ranking within a cell and cutting at a
global quantile selects the same number of timesteps for every cell, so it can never say a
cell is unusually anomalous). The top 0.1% by `z` is cut and grouped into space-time connected
components — adjacency is a real HEALPix NESTED neighbour at the same timestep, or the same
cell at *t+1*.

**Note the 1.4826.** `probe/cache.npz` stores the **raw** MAD; the `z` above divides by
`1.4826 · MAD`. Any ad-hoc script that reads `cell_mad` directly and forgets the factor
inflates every z by 1.4826× — this produced a round of wrong numbers during this analysis.
Rankings are unaffected (it is a global constant); absolute z values are not.

---

## 2. The tail is coherent, not noise

Measured on v6:

| | flagged | random expectation | enrichment |
|---|---|---|---|
| flagged cell has a flagged HEALPix sibling (worst timestep) | 89% | 3% | ~30× |
| flagged cell still flagged at *t+1* | 32.4% | 0.010% | ~3,000× |

So the top-0.1% tokens are not scattered numerical noise; they form connected space-time
objects. That is what makes "event" the right unit of analysis rather than "token".

---

## 3. Four events match the documented record

Checked against the literature, not assumed. Dates are the probe's, reconstructed as
`2014-01-01 00:00 UTC + idx × 6 h`.

| probe event | tokens | peak z | peak residual (global pct) | documented event |
|---|---|---|---|---|
| 2018-02-22 → 02-27, 87°N | 852 | 22.4 | 4,628 (**99.98th**) | Feb 2018 Arctic warming / SSW — zonal wind reversal 12 Feb, North Pole above freezing late Feb, Cape Morris Jesup 61 h above freezing |
| 2016-08-26 → 08-30, 84°N 79°E | 1,030 | 17.3 | 2,746 (88th) | Arctic cyclone: second storm into the central Arctic 22 Aug, 970 hPa on 23 Aug 2016 — **3-day lag**, so this is the mature/decay phase, not the peak |
| 2015-12-28 → 01-03, 84°N | 936 | 19.5 | 4,007 (**99.80th**) | North Pole above freezing 30 Dec 2015 (+0.7 °C record), driven by Storm Frank |
| 2022-03-16 → 03-20, 75°S | 895 | 15.0 | 1,074 (**14th**) | March 2022 East Antarctic heatwave, 15–19 Mar; all-time record −9.4 °C at Concordia on 18 Mar; Conger Ice Shelf collapse 15 Mar |

**Two peaks do not resolve** and are recorded as unattributed rather than forced:

- **2019-09-30 → 10-03, 34°N 141°E (off Tokyo)** — the strongest outlier on *every* scale
  (z 26.6; raw 5,299 = 99.998th pct; 5.7× its cell baseline). Typhoon Mitag tracked up the
  **Sea of Japan** side and went extratropical 3 Oct near 38°N 132°E; Tapah (19–23 Sep),
  Faxai (9 Sep) and Hagibis (6–12 Oct) all fall outside the window.
- **2019-01-25 → 01-26, 33°S 114°E (offshore Perth)** — January 2019 was Australia's hottest
  month on record *nationally*, but the BoM records Perth's **coolest** January in a decade,
  so the local match fails.

### 3.1 A free validation of the time axis

The latents store only an integer `idx`; the calendar date is *reconstructed*
(`2014-01-01 + idx×6h`) and has never been checked against a real timestamp. Three
independent events landing inside their documented windows is meaningful evidence that the
reconstruction is correct — an assumption the entire temporal analysis rests on.

---

## 4. The control that limits the claim

Ranking the same events by **raw** residual instead of `z` gives a different answer, and the
two disagree systematically:

| ranking | top-30 events at \|lat\|>60 | median cell baseline | median z |
|---|---|---|---|
| by `z` | **50%** | 744 | 19.1 |
| by raw residual | **0%** | 2,267 | 11.4 |
| *(all 12,288 cells)* | *14%* | *1,874* | — |

`z` selects intrinsically **quiet** cells; raw selects intrinsically **hard** ones. The
sharpest single consequence:

> The **March 2022 Antarctic heatwave — the largest surface temperature anomaly ever recorded
> on Earth (+38.5 °C at Dome C) — produces a residual at the 14th percentile globally, below
> the median.** It is a 3.9× excursion for that cell (baseline 278) and nothing at all in
> absolute latent terms.

This is not a defect of MAD. `MAD/median` is near-constant globally — 0.127 polar / 0.134
mid-latitude / 0.105 tropics, 2.0× spread from the 5th to the 95th percentile across all
cells — so a ratio-to-baseline ranking reproduces the `z` answer (40% polar). **The real
dichotomy is absolute vs locally-relative**, and both are legitimate questions:

- **locally-relative (`z`)** — "unusual *for this place*". This is the same question a
  meteorological record asks (records are always local records), which is why it agrees with
  the news archive. The agreement is therefore **partly built into the question**, not purely
  emergent. Nothing forced the *dates* to line up, so it is still meaningful — but it is a
  weaker claim than "the embedding finds the big events".
- **absolute (raw)** — "where is the model worst". Its answer is a *permanent geographic
  property*, not events: the SH subtropics, the Peruvian/Bolivian Altiplano, the Namib, and
  the subtropical gyres sit at baseline residual 1,800–2,800 against 941 polar. That is a
  **model-capacity gap** at K=128/d=64, and it should not be called an outlier.

---

## 5. Robustness

- **Not an artefact of one partition.** Re-running the whole pipeline on `v12_k256to128_d64`
  — a genuinely different clustering (NMI 0.684 vs v6, *inside* the seed-to-seed band) —
  puts **5 of the top 10 events on the identical peak cell and 3 on the same day**. The
  2018-02 Arctic event scores z 22.4 (v6) vs 22.6 (v12), 852 vs 798 tokens.
- **Independent corroboration.** `file_signature.py` on v12, which knows nothing about event
  extraction, names **2018-02-24T18** as v12's single most anomalous timestep.
- **Still model-relative.** The residual is measured against a K-subspace fit, so "outlier"
  means "far from the 64-dim flat its cluster spans". The v6/v12 agreement bounds this
  concern but does not remove it.

---

## 6. Conclusions about the embedding

1. **The encoder preserves extremes.** At a documented extreme, tokens depart from their
   cluster's subspace by 3.6–5.4× the local norm, coherently in space and time, reproducibly
   across partitions. Extremes are not smoothed away by the 2048-d encoding.
2. **Latent residual is not calibrated to physical anomaly magnitude.** The largest recorded
   temperature anomaly on Earth is a below-median residual. Latent distance is therefore
   **not** a safe proxy for physical significance.
3. **Latent variability is geographically *scaled*, not *shaped*.** Cell baselines run 278
   (East Antarctic plateau) to ~2,800 (Altiplano) — roughly 10× — while the distribution
   shape (`MAD/median`) is nearly constant. The encoder allocates very different amounts of
   latent room per region, but the same relative spread.
4. **Absolute-scale hotspots are a capacity gap, not events.** They are stable, geographic,
   and actionable (where to add clusters or dimensions) — a different question from anomaly
   detection, and one the same data answers.
5. **"Outlier cluster" is not a category this embedding contains.** `iso = ‖μⱼ−μ_global‖²/trace`
   peaks at 0.25 across all 128 clusters: no cluster sits apart from the rest. Combined with
   the ~33% near-tie rate and the knee-free merge cascade, outliers exist only at the **token**
   level.

---

## 7. Consequences for the timestamp-selection goal

Conclusion 2 is the one that bites. `analyze_forecast_error.py`'s goal-1 weights use latent
hardness as one of three ranked axes; conclusion 2 says that axis **systematically
down-weights polar extremes and up-weights the SH subtropics**, because it inherits whichever
normalisation it was built on. Concretely:

- Selecting by **raw** hardness buys mostly permanently-hard subtropical cells — the
  capacity gap — and will barely surface a polar extreme.
- Selecting by **`z`** buys locally-unusual states (the weather events), at the cost of
  over-representing quiet regions where a small absolute change scores large.
- Neither is wrong, but the choice **is** the selection policy and must be made
  deliberately, not inherited by accident from a normalisation default.

See `docs/ideas/latent_selection.md` for what follows from this for a TAROT-style selection
method, and whether the subspace model can supply the geometry it needs.

---

## Sources

- [NOAA Climate.gov — February 2018 heatwave across the Far North](https://www.climate.gov/news-features/event-tracker/february-2018-heatwave-across-far-north)
- [The Conversation — 'Beast from the East' and freakishly warm Arctic temperatures](https://theconversation.com/beast-from-the-east-and-freakishly-warm-arctic-temperatures-are-no-coincidence)
- [Journal of Climate — The Extraordinary March 2022 East Antarctica "Heat" Wave, Part I](https://journals.ametsoc.org/view/journals/clim/37/3/JCLI-D-23-0175.1.xml)
- [Berkeley Earth — Rapid analysis of the March 2022 Dome C record heatwave](https://berkeleyearth.org/antarctic-heatwave-rapid-attribution-review-dome-c-record/)
- [GRL — The Largest Ever Recorded Heatwave: attribution of the Antarctic heatwave of March 2022](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL104910)
- [Nature Scientific Reports — The December 2015 North Pole Warming Event](https://www.nature.com/articles/srep39084)
- [CNN — North Pole temperature climbs above freezing, 30 Dec 2015](https://www.cnn.com/2015/12/30/us/north-pole-high-temperature-feat/)
- [The Cryosphere — The Arctic sea ice cover of 2016](https://tc.copernicus.org/articles/12/433/2018/)
- [BoM — Australia in January 2019](https://www.bom.gov.au/climate/current/month/aus/archive/201901.summary.shtml)
- [Typhoon Mitag (2019)](https://en.wikipedia.org/wiki/Typhoon_Mitag_(2019))
