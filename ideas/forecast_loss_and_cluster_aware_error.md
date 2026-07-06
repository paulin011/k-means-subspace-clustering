# Forecasting-Engine Loss Functions & Cluster-Aware Error Functions

*Written 2026-07-06. Grounded in the actual WeatherGenerator source at `/home/psaher/WeatherGenerator` (citations are `file:line` into that clone) and in our own cluster runs (`subspace_kmeans_runs/v6_subspace_big_d64`). Companion to `wgen_architecture.md` and `CLAUDE.md`.*

---

## TL;DR

- The `ForecastingEngine` has **no loss of its own** — it is a pure latent→latent forward pass (self-attn + MLP, 2048-d in and out, `engines.py:564-636`). The training loss is computed **downstream, in physical space, after a decode**.
- The actual training objective is **`LossPhysical` with a single MSE** on decoded physical channels vs physical targets (`config_forecasting.yml:170-175`). It is unweighted by default: no dynamic channel reweighting, no latitude weighting, no forecast-step decay.
- A latent-space loss module (`LossLatentSSLStudentTeacher`, DINO/iBOT/JEPA with EMA/frozen teacher) **exists but is not used for forecasting** — only in pretraining configs. So the latent rollout is supervised *only indirectly* through decoded physical MSE. This is the instability `wgen_architecture.md` §5 flags.
- **Our 128 clusters are a natural coordinate system for the latent forecast error** (same 2048-d space). The strongest, no-retraining wins are on the **analysis side**: (A) subspace-decomposed error `‖r‖² → ‖r∥‖² + ‖r⊥‖²`, and (B) per-cluster skill vs persistence. Training-side uses (auxiliary on-manifold loss, stratified sampling) are research bets worth stating but lower-priority.

---

## 1. The loss functions in the forecasting engine

### 1.1 Where the loss lives

`ForecastingEngine` (the thing that does the autoregressive rollout `tokens = forecast_engine(tokens, step, rope_coords)`, `model.py:703`) is a pure forward pass. **It contains no loss.** The loss is external, computed on the **decoded physical output**:

```
latent tokens ──forecast_engine──► predicted latent ──predict_decoders──► physical preds
                                                                          │
                                          LossPhysical( preds.physical, targets.physical )
```

So: *forecast prediction* = latent (free, lives in our cluster space); *training loss* = physical (decoded). This split is the central fact for everything below.

### 1.2 The training stack (what actually runs)

`config_forecasting.yml:170-175` configures exactly **one** loss term:

```yaml
losses : {
  "physical": { type: LossPhysical, loss_fcts: { "mse": { } } }
}
```

Orchestration (`loss_calculator.py:81-107`): `loss = Σᵢ weightᵢ · LossModuleᵢ.loss`.

Inside `LossPhysical.compute_loss` (`loss_module_physical.py:240-453`) the objective is a nested weighted mean over **physical** tensors (`preds.physical` / `targets.physical`):

```
loss = Mean_stream( stream_weight · Mean_output_steps( step_weight · Mean_loss_fcts( mse(target, pred) ) ) )
```

The base metric is `mse` → `lp_loss(p_norm=2, with_mean=True)` (`loss_functions.py:206-224`): ordinary per-channel MSE, then averaged.

### 1.3 The weighting machinery (all optional, all OFF in the default forecasting config)

`LossPhysical` carries three nested weighting systems that are the natural knobs a "better loss" would use, but none are enabled in `config_forecasting.yml`:

| Weighting | Code | On by default? |
|---|---|---|
| Static per-channel (`target_channel_weights`) | `loss_module_physical.py:140-146` | No (None) |
| **Dynamic EMA channel** — "Samudra 2" inverse-MSE EMA, clamped to `L·min`, mean-renormalized | `DynamicLossEMA` `:34-80` | No (no `dynamic_loss` key ⇒ disabled) |
| Location (e.g. `cosine_latitude` for equal-area) | `_get_location_weights` `:173-182`; `loss_functions.py:290-292` | No (`location_weight` unset) |
| Forecast-step decay (`gamma_decay`) | `_get_output_step_weights` `:165-171`; `loss_functions.py:295-298` | No (`timestep_weight` unset ⇒ uniform 1.0) |

Available loss **functions** beyond MSE (`loss_functions.py`): `mae`, `rmse`, `rss` (the Lp family), and probabilistic `kernel_crps`, `mse_ens`, `gaussian_crps`, `stats*`. None are wired into the forecasting config.

### 1.4 A latent loss exists — but only for pretraining

`LossLatentSSLStudentTeacher` (`loss_module_ssl.py`) operates **in latent space** with an EMA or frozen teacher, implementing DINO / iBOT / JEPA self-supervised losses (`target_and_aux_ssl_teacher.py:34-164`). It is referenced only by pretraining configs (`config_jepa.yml`, `config_dinov2.yml`, `config_physical_jepa.yml`). The forecasting config uses `training_mode: ["masking"]` and excludes it. **Net: in forecasting training there is no direct latent-space loss term.**

### 1.5 Eval metrics

`score.py` (see `wgen_architecture.md` §2f): RMSE, MAE, MSE, ACC, RPSS, CRPS, etc., computed in **denormalized physical space**, averaged over `ipoint` (cells) by default — which **erases per-cell structure**. No persistence baseline exists anywhere (hence our `persistence_error.py`).

### 1.6 Doc-accuracy note

`config_forecasting.yml:187` has `num_steps: 3` (offset 1, 6 h step ⇒ targets at base+1 / +2 / +3), whereas `wgen_architecture.md` §2e states "num_steps=2". Minor, but worth syncing in `wgen_architecture.md` / `CLAUDE.md`.

---

## 2. Can the clusters yield a better error function?

Our clusters are **128 affine subspaces in the 2048-d latent** (K-subspaces, d=64). Every cell has a mean `μ_k`, a 64-d orthonormal basis `U_k` (saved in `model.pt`), and a label (in `assignments.pt`). The latent forecast error `‖pred − truth‖²` that `extract_forecast_error.py` computes lives in the **same space**. So the cluster geometry is a natural coordinate system for the residual.

Separate two regimes, because the project is currently on the **analysis side** (forward-only extraction, no retraining) and that is where the clusters are an unambiguous win.

### 2.1 Analysis / eval side — cheap, no retraining (STRONG)

#### (A) Subspace-decomposed forecast error

Today the error is `‖pred − truth‖²` summed over all 2048 dims. With each cell's basis, split the residual:

```
r   = pred[cell] − truth[cell]
r∥  = U_k U_kᵀ r          # the 64 "regime / manifold" directions the data actually occupies
r⊥  = r − r∥               # the ~1984 off-manifold directions
```

Report skill separately on `‖r∥‖²` and `‖r⊥‖²`. Our K-subspaces fit showed ~31.5% of latent variance is *within* subspaces, ~60% between — so this decomposes the forecast error into "did we move along the right regime axis?" (∥) versus "did we fall off the manifold?" (⊥). These are genuinely different failure modes, and this is a **strictly more informative error function** for goal-2 (embedding understanding) than the scalar residual. It uses `U_k` already in `model.pt`; it is a post-hoc reduction on `err.npy`-equivalent data — no model change.

#### (B) Per-cluster skill vs persistence

Replace the single global `skill = 1 − Σfc/Σps` (ratio-of-sums, as in `analyze_forecast_error.py`) with

```
skill_k = 1 − Σ_{cells ∈ k} fc / Σ_{cells ∈ k} ps
```

`analyze_forecast_error.py` already attributes error **magnitude** per cluster; the natural next metric is per-cluster **skill**. Rare-regime clusters with low skill are the highest-value training timestamps — this directly sharpens the goal-1 `weights.csv`.

> **Prioritize A + B.** Both are free, use cluster geometry already on disk, and are strictly better error *measurements*.

### 2.2 Training side — requires supercomputer retraining (MIXED)

#### (C) Auxiliary on-manifold consistency loss (PLAUSIBLE — scaffold exists)

`wgen_architecture.md` §5 mitigation is real: the SSL teacher path already exists. Add a latent term penalizing `‖r⊥‖²` (forecast drifting off the data manifold) alongside the physical MSE. This gives the rollout the **direct latent correction signal** the architecture currently lacks — the most defensible way clusters improve the *training loss*.

Caveats: it is a research bet; the SSL machinery is heavy; a hard 128-way cluster cross-entropy would over-regularize. The soft `‖r⊥‖²` penalty is more principled than a cluster-classification term.

#### (D) Stratified sampling — not a loss, but the real goal-1 lever (STRONG)

This is *sampling*, not a loss, but it is the most direct way clusters improve forecast quality: replace the uniform `rng.permutation` in `multi_stream_data_sampler.py` with regime-stratified draws so rare clusters are seen more often. The loss stays MSE; the gain is in **what the MSE sees**. This is goal-1 as already scoped in `wgen_architecture.md` §2e and `CLAUDE.md`.

#### (E) Cluster-reweighted physical MSE (WEAK)

`LossPhysical` weights physical target points and channels, not latent tokens — a latent cluster label has no clean entry point into the physical MSE without a learned mapping. Low leverage relative to C/D. Do not pursue.

### 2.3 The one caveat that applies to everything

The subspaces `U_k` were fit to **encoder analysis states** (the `latents_2[t]` side), not to **forecast-step latents** (the `pred` side). Whether forecast residuals naturally align with these subspaces — i.e. whether decomposition (A) actually separates meaningful signal — is **empirical and testable for free** with data already on disk: split `err_persist.npy` into ∥/⊥ and check whether the ∥/⊥ energy ratio and spatial pattern match the physically-coherent persistence map. If forecast-step latents drift off the encoder manifold, (A) weakens — but (B) is unaffected, since it uses only labels, not bases.

**Stationarity of the embedding (relevant aside):** the encoder is **frozen during finetuning** (`config_forecasting_finetuning.yml:10` — `freeze_modules` regex `global|local|adapter|q_cells|latent|ERA5` freezes the whole assimilation stack; only `forecast_engine` + decoder `pred_heads` refine), and `latents_2`'s 2014–2022 window matches that finetuning config (not the base config's 1979→2022). So `latents_2` is very likely a **fixed-encoder** output — a stationary 2048-d target the forecast engine was trained against, which makes the cluster subspaces `U_k` a stable coordinate system for the forecast residual. See `wgen_architecture.md` §2(g).

---

## 3. Recommendation

1. **Do A + B now** on the analysis side — strictly better error functions, free, using cluster geometry already on disk.
2. **Before committing to C**, run the (A) decomposition against persistence to confirm forecast residuals actually live on encoder-fit subspaces. That single check tells you whether the cluster geometry is the right coordinate system for forecast error at all.
3. Treat **C** as the training-loss research direction (hook = existing SSL teacher) and **D** as the practical goal-1 sampling lever. Skip **E**.

### Concrete next step (proposed, not yet implemented)

Extend `analyze_forecast_error.py` with a subspace-decomposition mode: load `U`/`means` from `model.pt`, gather per-cell bases via `assignments.pt`, and split `err_persist.npy` (and eventually `err_forecast.npy` from the supercomputer run) into `‖r∥‖²` / `‖r⊥‖²` components, reported per-cluster and as maps. This validates (A) cheaply on the persistence baseline before the forecast-error extraction even lands.

---

## References

- Source (clone): `/home/psaher/WeatherGenerator`
  - `src/weathergen/train/loss_calculator.py` — loss orchestration
  - `src/weathergen/train/loss_modules/loss_module_physical.py` — `LossPhysical`, `DynamicLossEMA`
  - `src/weathergen/train/loss_modules/loss_functions.py` — `mse`/`mae`/`rmse`/`rss`/`kernel_crps`, weighting helpers
  - `src/weathergen/train/loss_modules/loss_module_ssl.py` — `LossLatentSSLStudentTeacher` (DINO/iBOT/JEPA)
  - `src/weathergen/train/target_and_aux_ssl_teacher.py` — EMA / frozen teacher
  - `config/config_forecasting.yml` — forecasting loss config
- Ours: `wgen_architecture.md` (§2d, §2e, §2f, §5), `CLAUDE.md`, `extract_forecast_error.py`, `persistence_error.py`, `analyze_forecast_error.py`, `subspace_kmeans_runs/v6_subspace_big_d64/`
