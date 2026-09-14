# Can subspaces do latent selection? — TAROT, and what our geometry can and cannot supply

Assessment of the original goal — **select the best ~20% of the 13,021 timestamps to fine-tune
the WeatherGenerator** — against what `subspace_kmeans.py` / `file_signature.py` /
`cluster_probe.py` actually produce. Written 2026-08-13. Companion to
`docs/LATENT_OUTLIERS.md` (the verified outlier findings this builds on).

**Verdict up front.** The subspace model supplies the *right correction* (a local whitening)
to the *wrong space* (encoder features, not loss gradients), aggregated at the *wrong
resolution* (per-file means, not per-cell events), against an *undefined target*. Two of
those three are fixable on this box today and one is not. Selection built on the current
per-file `cluster_mix.csv` is close to worthless — measured below, that space is effectively
**1.8-dimensional and 96% seasonal cycle**, so "diversity" selection in it is a calendar
shuffle. Selection built on per-cell event statistics with an explicit target is defensible
and worth building; but the honest ceiling on any latent-only method is set by the fact that
**latent geometry is not loss geometry**.

**§7 (added 2026-08-13) settles that empirically.** A local proxy forecaster was built and run
(`proxy_forecast.py`, +0.58 skill over persistence, no generalisation gap). Its error is
**uncorrelated with the latent hardness axis currently used for selection** (`resid_frac`
ρ −0.04, top-20% overlap 18.2% against a 20% random baseline). Latent geometry does not track
model error. Among checkpoint-free axes only `rare_exposure` (31.1%) and `zres_mean` (28.9%)
positively track where the model actually fails.

---

## 1. What TAROT actually is

[TAROT: Targeted Data Selection via Optimal Transport](https://arxiv.org/abs/2412.00420)
(Feng et al., ICML 2025, EPFL VITA). Not an anomaly detector — a **targeted data selection**
framework. Pipeline:

1. **Features are loss gradients**, not embeddings: `φ(z) = Σ_{i=1..T} ∇L(z; θᵢ)` summed over
   `T` training checkpoints, then randomly projected (TRAK-style) to a tractable dimension.
2. **Whitened Feature Distance (WFD).** Centre the projected gradients, take the covariance
   `Σ = LLᵀ` (Cholesky; ZCA also offered as a structure-preserving but costlier variant),
   apply `φ^w = L⁻¹φ`, normalise to unit length, and use plain Euclidean distance on that.
   The stated motivation: gradient features are correlated, the covariance is
   **ill-conditioned**, and a few dominant directions otherwise swamp the distance.
3. **Selection = minimise the optimal-transport distance** (Kantorovich) between the selected
   candidate set and the **target** distribution, with WFD as the ground cost. Two schemes:
   fixed-size (k-NN growth + OT potentials for the final ranking) and **OTM**, which uses
   10-fold CV over the target and stops when the OT distance starts rising — this is what
   makes the selection *ratio* an output rather than a hyperparameter.
4. **Measured ratios**: 24% (GTA5→Cityscapes segmentation), 29.8% (motion prediction),
   <0.5% (instruction tuning). Beats LESS, DsDm, TRAK, TSDS and, on two tasks, beats training
   on the *entire* dataset.
5. **Cost**: gradient features ~32 H100-hours (one-off, amortised across targets); the
   selection itself 59–118 s.
6. **Stated limitation**: with a small or highly specific target it overfits the target;
   future work is named as "incorporating distribution diversification into the objective".

Two things to hold onto: the ~20% budget is squarely in TAROT's own measured range, so the
goal is not unreasonable — and the method is **target-conditioned and gradient-based**, which
is where our setup diverges.

---

## 2. The one place our geometry is genuinely the right tool

TAROT's core diagnosis is that a few dominant components hijack high-dimensional distances.
**That is precisely the pathology measured in `docs/LATENT_OUTLIERS.md`**: raw latent residual
is dominated by a ~10× geographic scale factor (cell baseline 278 on the East Antarctic
plateau vs ~2,800 on the Altiplano), which is why the March 2022 Antarctic heatwave — the
largest surface temperature anomaly ever recorded — lands at the *14th percentile* of raw
residual. Any selection on un-normalised latent distance inherits that bias wholesale.

So the correction TAROT applies is the correction we need, and the subspace model already
computes a stronger version of it than the per-cell median/MAD used so far:

| | whitening used | scope |
|---|---|---|
| `cluster_probe.py` today | `(r − median_cell)/(1.4826·MAD_cell)` | diagonal, per **cell**, on a scalar |
| TAROT WFD | `L⁻¹(φ − φ̄)`, `Σ = LLᵀ` | full covariance, global, on gradients |
| **available here** | MPPCA Mahalanobis `R_j/σ²_j + Σᵢ zᵢ²/λᵢ` | full covariance, per **cluster**, on 2048-d tokens |

That third row already exists and is already implemented — it is the `--soft` scoring in
`subspace_kmeans.py`, using `U_j`, `λ_j`, `σ²_j` from `model.pt`. **It is the closest
latent-space analogue of WFD obtainable without a checkpoint**, it is exact rather than a
diagonal approximation, and it costs two einsums. If any latent-space selection is built,
this is the distance it should use — not Euclidean, not raw residual.

---

## 3. Three measured reasons the current setup cannot carry a TAROT-style selection

### 3.1 The per-file mix space is a clock (the decisive one)

`cluster_mix.csv` `[13021, 128]` — each timestamp's fraction of cells per cluster — is the
obvious OT ground space: 13,021 histograms on a 128-simplex, OT between them is trivial.
Measured on v6:

- **Effective dimensionality (participation ratio) = 1.8 of 128.** PC1 alone is **73.4%** of
  the variance; 12 components reach 90%.
- Regressing the top PCs on annual + semiannual harmonics of day-of-year: **R² = 0.992 for
  PC1**, and **96% of the top-5 variance is pure seasonal cycle**.

So the regime fingerprint is, to first order, a calendar. Optimal transport between mix
histograms would match *seasonal distributions* and almost nothing else, and any "diversity"
term over this space is a calendar shuffle dressed up as geometry.

- Coverage in it saturates: a **random 20%** (2,604 files) reaches max-coverage-distance
  0.0199, while a **10× smaller** 2% k-center set reaches 0.0231, against a full-set median
  nearest-neighbour distance of 0.0066. The entire cloud spans about three NN-distances.
  There is nothing left to cover at a 20% budget.

### 3.2 Per-file aggregation destroys exactly the signal that works

The outlier result is a *per-cell, per-timestep* phenomenon: 89% of flagged cells have a
flagged HEALPix neighbour (3% at random), 32.4% persist to *t+1* (0.010% at random). A real
event is ~150–250 cells of 12,288. Averaging a file's 12,288 residuals into one
`mean_residual` dilutes a 152-cell event by ~80×. **The per-file summary is the wrong unit**;
`residual_map.npy` / `label_map.npy` / `margin_map.npy` already hold the per-cell version and
`cluster_probe.py` already extracts events from it.

### 3.3 The existing selection axes disagree, so the fusion is doing the deciding

`analyze_forecast_error.py` fuses hardness, rare-exposure and mix-space diversity into
`weights.csv`. Overlap of the top-20% sets each axis picks (random baseline = 20.0%):

| | hard | rare | margin | diverse |
|---|---|---|---|---|
| **hard** | — | **6.7%** | 61.1% | 34.2% |
| **rare** | 6.7% | — | 9.2% | 7.5% |
| **margin** | 61.1% | 9.2% | — | 40.2% |
| **diverse** | 34.2% | 7.5% | 40.2% | — |

`hard` and `rare` are **anti-correlated** (6.7% against a 20% baseline) — the hardest
timestamps are the ones *least* exposed to rare regimes. So the three axes are not three
views of one quantity; they pick nearly disjoint sets, and the fused score is determined by
the fusion weights rather than by the data. That is a real fragility in the current
`weights.csv`, independent of everything else here.

---

## 4. The structural objection: latent geometry is not loss geometry

This one is not fixable by better normalisation, and it is the reason to be cautious about
any latent-only selector.

- The data-selection literature consistently finds **gradient/influence features beat
  embedding-similarity features**, because gradients encode a task-specific error signal
  while embeddings encode generic similarity. Work quantifying the two directly reports
  meaningful but imperfect agreement, with the practical recommendation that similarity is a
  cheap **screening** proxy, not a substitute for influence when precision matters.
- [Contributing Dimension Structure](https://arxiv.org/abs/2401.16193) makes the sharpest
  version of the point for us: plain distance in feature space is insufficient precisely
  because only *some* dimensions drive predictions, and it identifies those dimensions with
  **gradients**. Our `U_j` are the top-**variance** directions, found unsupervised. Maximum
  variance is not task relevance; there is no guarantee that the residual off a variance-optimal
  64-dim flat is what the forecast engine finds hard.
- The one piece of counter-evidence in our own data is encouraging but weak: per-cluster
  residual correlates **ρ 0.80** with the persistence baseline `err_persist`, i.e. what the
  subspaces fail to capture is largely what changes fastest. That is a relationship to
  physical dynamics, still not to *model loss*.
- And `docs/LATENT_OUTLIERS.md` conclusion 2 stands over all of it: latent residual magnitude
  is not calibrated to physical anomaly magnitude, so hardness-ranked selection is
  geographically biased under **every** normalisation — the choice of normalisation *is* the
  selection policy.

### 4.1 Plus a pruning-rate warning

Keeping 20% is **80% pruning** — a high rate, and
[Coverage-centric Coreset Selection](https://arxiv.org/abs/2210.15809) shows difficulty-based
selection degrades badly there: concentrating on the hardest examples drops the easy examples
that carry the foundational signal, and coverage-preserving stratified sampling wins instead.
A naive "take the hardest 20% of timestamps" is the documented failure mode. Note this does
*not* rescue §3.1 — coverage must be preserved in a space that has structure, and the mix
space does not.

---

## 5. What follows — a defensible plan

**Tier 0 — define the target.** TAROT is *targeted* selection; without `𝒟_target` the OT has
nothing to transport to and the method degenerates. "Fine-tune WGen" is not a target.
Candidates, in increasing order of usefulness: (a) the extreme-event set from
`cluster_probe.py` (well-defined, ~45k events, and verified against the documented record);
(b) a held-out period; (c) a downstream skill metric. **This decision is upstream of every
technical choice below** and is currently unmade.

**Tier 1 — fixable here, today, no checkpoint.**
1. Move the selection features from per-file means to **per-cell event statistics** (§3.2):
   per timestamp, the count/area/peak-z of extreme events it contains, per cluster. This is a
   reduction of arrays that already exist.
2. Use the **MPPCA Mahalanobis** as the ground distance (§2), not Euclidean or raw residual.
3. Stratify rather than threshold (§4.1): CCS-style strata over cluster × robust-z band,
   sampled within strata — and explicitly *not* over the mix space (§3.1).
4. Fix the axis fusion or drop it (§3.3): report the three axes separately with their
   overlaps rather than fusing anti-correlated signals into one number.

**Tier 2 — the forecast-error route, now with a known defect.** The correct selection signal
is not "where is the encoder's latent unusual" but "where is the model wrong", and
`err_forecast[t, cell]` is the per-cell loss. **But the naive latent form of it is
confounded**: the forecast engine's output is not in the same space as its input. An
un-normalised encoder output goes in (`ae_global_trailing_layer_norm: False` everywhere), and
a LayerNorm-anchored token comes out (`fe_layer_norm_after_blocks: [7]`, parameter-free), so
`‖forecast_engine(latents_2[t]) − latents_2[t+1]‖²` mixes forecast error with a systematic
scale/offset mismatch. Full derivation in `docs/wgen_architecture.md` §5b. Usable variants,
in order of cost: **(a)** apply the checkpoint's frozen `latent_pre_norm` to *both* sides
before differencing — measures pattern error, not magnitude; **(b)** use the physical-space
per-cell loss, which is the model's real objective, at the cost of the decoder + ERA5 targets
+ an unconfirmed `ipoint`→cell mapping. Either way, run `--selfcheck-only` first: the script's
existing `skill ≤ 0` abort is a direct test of exactly this defect.

**Tier 2b — a local proxy model, which may be strictly better than waiting. IMPLEMENTED as
`src/forecast/proxy_forecast.py` (2026-08-13); results in §7.** If the
checkpoint route stays blocked, the literature supports substituting a *small* model:
[Selection via Proxy](https://arxiv.org/abs/1906.11829) shows that a stripped-down proxy —
fewer layers, fewer epochs, an order of magnitude cheaper — gives a data-selection signal
that transfers to the target model, and [Small-to-Large Generalization](https://arxiv.org/pdf/2505.16260)
finds data influence is consistent across scale. TAROT's own transferability result is the
same phenomenon: data selected with AutoBots improved Wayformer, a 10× larger model.

Concretely here: **train a small latent→latent forecaster on `latents_2` on this box** — the
inputs *and* targets are already on disk (1.2 TB of consecutive pairs), two A40s (44.4 GiB) are
available, and no checkpoint, decoder, ERA5 reader or supercomputer allocation is needed. Its
per-cell error is a *model* error, not an encoder-geometry statistic, which is the categorical
upgrade over everything in Tier 1. Two cautions: SVP's own results show proxy-based selection
can be high-variance when extrapolating from very few models, and a proxy trained on the raw
latents inherits the same space question — but here it is harmless, because the proxy is
trained *and* evaluated in one consistent space.

**Tier 2c — skip offline selection entirely.** If the fine-tune is being run anyway, online
batch selection ([RHO-loss](https://arxiv.org/pdf/2107.02565): prioritise points that are
learnable, worth learning, and not yet learned) picks per batch using the *current* model's
loss against a holdout-trained reference. It needs no proxy, no offline scoring pass, and no
latent geometry at all, and it targets influence more directly than any static score. The
trade-off is that it changes the training loop rather than producing a reusable 20% subset.

**Tier 3 — full TAROT**, if Tier 2 lands: gradient features over checkpoints → random
projection → Cholesky/ZCA whitening → OT against the chosen target, with OTM estimating the
ratio instead of assuming 20%. TAROT's own measured ratios (24%, 29.8%) suggest ~20% is a
reasonable prior, but it should be an output.

---

## 6. Direct answer to "can subspaces be used for latent selection?"

**Yes, in one specific role, and no in the role the goal needs.**

- **Yes** as a *whitening and stratification* device. The per-cluster `U_j`/`λ_j`/`σ²_j` give
  a principled local metric (§2) that fixes the dominant-component bias TAROT identifies, and
  the cluster labels give a stratification for coverage-preserving sampling. That is a real
  contribution and it is cheap.
- **No** as the *selection criterion itself*. Latent residual measures distance from a
  variance-optimal subspace, not influence on training loss; it is uncalibrated to physical
  significance; and its per-file aggregate is a seasonal clock. Selecting the top 20% by
  latent hardness would produce a geographically biased, seasonally skewed set, at a pruning
  rate where the literature says difficulty-only selection fails anyway.

The subspaces earn their place as **the coordinate system and the metric**, not as the score.

### 6.1 Ranked shortlist of alternatives

Best first, given that the checkpoint is not on this box:

1. **Local proxy forecaster** (Tier 2b) — train a small latent→latent model on `latents_2`,
   select on its per-cell error. Turns the signal from "unusual encoding" into "model error",
   which is the categorical upgrade. Runs here, today, no checkpoint. Grounded in SVP and
   small-to-large influence consistency.
2. **Online batch selection / RHO-loss** (Tier 2c) — if the fine-tune is being run anyway,
   skip offline selection. Most direct proxy for influence; changes the training loop.
3. **Forecast error with the LN fix** (Tier 2a) — correct in principle, blocked on the
   checkpoint, and now known to need normalising on both sides. Still the reference signal.
4. **Persistence-relative difficulty** — `err_persist [13020,12288]` already exists, is
   verified, and is completely model-free: it measures 6 h latent tendency. Weaker than a
   model error, but it is honest, needs nothing new, and it does not depend on the clustering
   at all. A good baseline to beat.
5. **Event-based coverage over subspace strata** (Tier 1) — the best purely-latent option;
   use it for *stratification and the metric*, not as the score.
6. **Per-file mix-space diversity** — effectively a calendar shuffle (§3.1). Do not use.

---

## 7. RESULTS — the proxy was built and run (2026-08-13)

Tier 2b is no longer a proposal. `src/forecast/proxy_forecast.py` trains a small
latent→latent forecaster on `latents_2` (own token + 8 HEALPix neighbours → 512-d → 6
residual MLP blocks → tendency head, 10.8 M params, 20 k steps, 42 min on one A40 including
full-dataset inference). `runs/proxy/v1/err_proxy.npy [13020,12288]` is a **model** error
field with the same contract as `err_persist.npy`.

### 7.1 The proxy is real

| | skill vs persistence |
|---|---|
| all 13,020 transitions | **+0.5822** |
| transitions inside the training chunks | +0.5844 |
| **transitions fully unseen (76.8% of the record)** | **+0.5815** |

No generalisation gap at all (0.003), so the field is trustworthy everywhere. Per-cell skill
is **positive in every one of the 12,288 cells** (min +0.104, median +0.574, max +0.720);
by zone, mid-latitudes +0.618 > tropics +0.559 > polar +0.506. The map
(`runs/proxy/v1/maps/map_proxy_skill.png`) is physically coherent — high predictability over
the subtropical gyres, low over the storm tracks and the Antarctic interior.

**Ablation:** removing the 8 neighbours (cell-only context) drops skill from **+0.586 to
+0.491**. So spatial context is worth ~9.5 points, but a *per-cell* map of the token alone
already recovers +0.49 — meaning **the 2048-d token at one cell already encodes most of what
is needed to predict its own 6 h evolution**. That is a finding about the embedding in its own
right: these are not instantaneous snapshots, they carry local dynamical state.

### 7.2 The central claim, now measured rather than argued

Rank correlation and top-20% overlap of `proxy_err` against the latent-geometry axes
(`runs/selection/v1/report.md`; random-baseline overlap = 20.0%):

| axis | Spearman ρ with `proxy_err` | top-20% overlap |
|---|---|---|
| `resid_frac` (**the axis currently in `weights.csv`**) | **−0.04** | **18.2%** |
| `near_frac` | −0.01 | 21.4% |
| `zres_mean` (geography divided out) | +0.18 | 28.9% |
| `persist` | +0.67 | 59.4% |

**Latent geometry does not track model error.** The current hardness axis is *uncorrelated*
with where the model actually fails, and its top-20% set overlaps the model-error set at
**below** the random baseline. §4's structural objection is confirmed empirically: selecting
on encoder-side statistics selects something other than where the model is wrong.

### 7.3 What to select on instead, if no checkpoint is available

`proxy_err` is dominated by "the weather moved a lot" (ρ +0.67 with persistence). The
sharper target is **`proxy_skill` = 1 − err/persist** — where the model fails *relative to
how much the state changed* — which is nearly orthogonal to raw tendency (ρ −0.08 with
`persist`). Overlap of each latent axis with the hardest 20% by `proxy_skill`:

| axis | overlap with hardest-20% by `proxy_skill` |
|---|---|
| `rare_expo` | **31.1%** |
| `zres_mean` | **28.9%** |
| `n_extreme` | 26.0% |
| `persist` | 11.2% |
| `near_frac` | 4.2% |
| `resid_frac` | **6.6%** |

So among the checkpoint-free axes, **`rare_exposure` and `zres_mean` are the only ones that
positively track model difficulty**, and `resid_frac` / `near_frac` are actively
anti-correlated with it. Concrete recommendation for `analyze_forecast_error.py`'s
`weights.csv`: **drop `residual_frac` as a hardness axis** — on this evidence it steers
selection away from the timestamps the model finds hard — and prefer `rare_exposure`,
`zres_mean`, and (best of all, since it is a genuine model error) `proxy_skill` itself.

### 7.4 Caveats

- The proxy is ~11 M parameters against the WeatherGenerator's ~0.8 B, and it predicts a 6 h
  latent tendency rather than the physical fields the real loss scores. SVP's own results warn
  that proxy-based selection is high-variance when extrapolating from very few models. Treat
  `proxy_skill` as the best *available* difficulty signal, not as a substitute for the real one.
- Everything is still measured in the raw encoder space. That is legitimate here — the proxy
  is trained *and* evaluated in that one space, so §5b's mismatch cannot arise — but it means
  `proxy_err` and a future `err_forecast` are not on a common scale.

---

## Sources

- [TAROT: Targeted Data Selection via Optimal Transport (arXiv 2412.00420, ICML 2025)](https://arxiv.org/abs/2412.00420) · [proceedings](https://proceedings.mlr.press/v267/feng25l.html) · [code](https://github.com/vita-epfl/TAROT)
- [Coverage-centric Coreset Selection for High Pruning Rates (arXiv 2210.15809)](https://arxiv.org/abs/2210.15809)
- [Contributing Dimension Structure of Deep Feature for Coreset Selection (arXiv 2401.16193)](https://arxiv.org/abs/2401.16193)
- [Quantifying the Agreement Between Data-Influence and Data-Similarity (arXiv 2606.23591)](https://arxiv.org/pdf/2606.23591)
- [Beyond Similarity: A Gradient-based Graph Method for Instruction Tuning Data Selection (arXiv 2502.11062)](https://arxiv.org/html/2502.11062)
- [D2 Pruning: Message Passing for Balancing Diversity and Difficulty (arXiv 2310.07931)](https://arxiv.org/pdf/2310.07931)
