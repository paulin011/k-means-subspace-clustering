# Forecast/persistence error analysis — `forecast_error/persist_v6`

*Bridge mode (persistence only) — forecast columns await `err_forecast.npy` from the supercomputer. Methodology: design+verify workflow spec, then adversarially code-reviewed (7+6 agents).*

## 1. Headline

- Persistence baseline only: global mean per-cell 6h error = **3149.6** (this is the skill denominator the forecast must beat).
- Mode: **bridge**. Masked (eps-floor) fraction: 0.00e+00 (≈0 ⇒ persistence has no near-zero cells; skill is well-defined everywhere once err_forecast lands).
- **Latent-vs-physical caveat:** skill is computed in the model's native 2048-d rollout space (the ForecastingEngine IS latent→latent), so err-derived weights are *aligned with the training loss*. Absolute levels decouple from physical skill only via the (absent here) decoder.

## 2. Attribution quality

- Static dominant-per-cell map: per-cell purity mean = **0.656** (685 pure cells); mean token mis-attribution ≈ **34.4%**.
- Zero-cell clusters: **[13, 98]** (non-empty, non-rare; tokens attributed to their neighbours — require `--dynamic` for a clean budget).
- Static-vs-dynamic gate (7000∩err, free): agreement 0.63–0.67, NMI 0.76–0.78 across months; none contested ⇒ static trustworthy.

## 3. Per-cluster forecast/persistence error

See `per_cluster.csv` — in bridge mode the ranking is by **mean_ps** (cluster changeability: which regimes move most between 6h steps). Hardest clusters: #54 (4993, 47 cells), #105 (4966, 126 cells), #40 (4851, 45 cells). Skill/error-budget columns activate in full mode.
 Cluster subspace size d80: median 37, max 40.

## 4. Temporal structure

- **Synoptic hour** (lead lens — persistence IS the 6h tendency):
  - 00:00 UTC  n=3255  mean_ps=3085.2  (skill N/A in bridge)
  - 06:00 UTC  n=3255  mean_ps=3203.1  (skill N/A in bridge)
  - 12:00 UTC  n=3255  mean_ps=3136.7  (skill N/A in bridge)
  - 18:00 UTC  n=3255  mean_ps=3173.4  (skill N/A in bridge)
- **Calendar month** (seasonal):
  - 01  n=1116  mean_ps=3053.6  
  - 02  n=1016  mean_ps=3073.4  
  - 03  n=1116  mean_ps=3184.8  
  - 04  n=1080  mean_ps=3241.0  
  - 05  n=1116  mean_ps=3247.1  
  - 06  n=1080  mean_ps=3171.7  
  - 07  n=1116  mean_ps=3122.4  
  - 08  n=1116  mean_ps=3112.4  
  - 09  n=1080  mean_ps=3168.5  
  - 10  n=1116  mean_ps=3164.4  
  - 11  n=1076  mean_ps=3142.1  
  - 12  n=992  mean_ps=3106.1  
- **Year** (stationarity, NOT climate trend; 2022 partial):
  - 2014  n=1460  mean_ps=3155.6  
  - 2015  n=1460  mean_ps=3132.4  
  - 2016  n=1464  mean_ps=3178.0  
  - 2017  n=1460  mean_ps=3153.0  
  - 2018  n=1460  mean_ps=3124.3  
  - 2019  n=1460  mean_ps=3178.9  
  - 2020  n=1464  mean_ps=3173.5  
  - 2021  n=1460  mean_ps=3125.8  
  - 2022  n=1332  mean_ps=3122.4   [PARTIAL]

## 5. Maps

- `maps/map_persist.png` — 6h-tendency climatology (reference frame).
- `maps/map_dominant.png` — dominant-cluster legend (co-locate scalar maps with regime geography).

## 6. Goal-1 training-timestamp weights

Per-source-transition weights in `weights.csv` (13020 rows; file 13020 is only ever a target ⇒ no weight). Each axis is rank-normalized to a percentile in [0,1] so a tightly-clustered raw signal (rare_exposure) still contributes its true ordering; the **composite = weighted mean of the percentiles** maps through `1+(Wmax-1)·s^α` and is renormalized to mean 1 (a product of three ~[1,Wmax] multipliers was rejected — it saturates the clip and erases the high-tier gradient):
- **(A) hardness** percentile of h_t = mean per-timestamp error (err_persist (bridge proxy)).
- **(C) rare-regime** percentile of rare_exposure (bottom-quartile-rarest cluster share).
- **(D) diversity** percentile of 1/local-density in regime-mix space (k-center-style anti-redundancy via 1-NN distance in cluster_mix.csv features).
- Component weights (wh/wrare/wdiv) default equal (1.0/1.0/1.0); Wmax=8.0, α=2.0. Range after renorm: w_final ∈ [0.34, 2.45], σ=0.41.
- **Anti-leakage:** clusters optimize an *encoder* objective, err is a *forecast* objective (different losses); the static map is a frozen partition. Encoder features (mix/rare/residual) are leakage-free for diversity IF the encoder is frozen in retrain; if the full model retrains, let forecast-side err drive weight magnitude. In-sample upweighting is intended but ⇒ validate prospectively on a buffered temporal hold-out (never on upweighted timestamps).
- **Handoff:** WGen uses a uniform-random `IterableDataset` sampler; applying these weights needs an inlined weighted/stratified sampler on the training side — this script only *emits* the weights.

## 7. Subspace decomposition (deferred)

Cannot split `err_forecast` along/orthogonal to `U_j` from the scalar `.npy` alone. The local option is to decompose `err_persist` (vector on disk); the forecast side needs `extract_forecast_error.py` extended to emit `err_fc_along.npy` (project onto frozen v6 `U`). TODO.

## 8. Correctness checks

- shapes/dtypes ✓; source contiguous & target=source+1 ✓; synoptic hours [3255, 3255, 3255, 3255] ✓; 2022 partial (1332 steps); pure cells = 685 (=685) ✓; dom_frac(per-file) mean 0.022 (≠ purity 0.656) ✓; masked frac 0.00e+00; w_final std 0.407>0 ✓.
- **err_persist integrity (checklist #3):** ‖x_t−x_{t+1}‖² recomputed from raw latents_2 for a scattered 12-row sample matches `err_persist.npy` exactly (max rel-error 0.00e+00); file `idx` keys match `source_idx` positionally. ✓
- **File-count self-check (#20):** the count-weighted encoder residual reproduces `final_obj_per_token`=1887.83 (cf. signatures 1888.12 vs 1887.83). ✓ (already validated in file_signature.py).
