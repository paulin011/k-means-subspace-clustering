# ECMWF WeatherGenerator — Architecture Report for Per-Cell Forecast-Error Attribution

*Generated 2026-06-26 by a 6-reader dynamic workflow over the cloned repo (`/home/psaher/WeatherGenerator`, `git clone --depth 1` of https://github.com/ecmwf/WeatherGenerator). All six readers (encoder, forecast/decoder, backbone+MoE, training/loss, data/sampling, eval/metrics) returned HIGH-confidence, mutually consistent findings; no cross-reader disagreements. Every claim below carries a `file:line` citation into that clone.*

## 1. TL;DR

- **The forecast engine predicts next-step LATENT tokens**, not physical fields. `ForecastingEngine` is a latent-to-latent transformer (in/out both `ae_global_dim_embed`=2048, `engines.py:564-606`); the rollout comment at `model.py:695` literally says "roll-out in latent space." The *training loss* is later computed in physical space after a decode, but the latent prediction itself is produced for free and lives in the **same 2048-dim space as our 128 clusters**.
- **`latents_2/latent_{i}.pt` `[1,12288,2048]` IS the encoder output `tokens_global`** (`encoder.py:140`), which is exactly the input fed to `forecast_engine` (`model.py:685,703`). So `forecast_engine(latents_2[t])` predicts `latents_2[t+1]` — both sides of the error already exist on disk; **no encoder re-run is needed**.
- **The model is fully DENSE — zero MoE** anywhere (`grep` for moe/expert/routing = 0 hits). The per-block MLP FFN (`layers.py:31-95`) is the natural MoE slot.
- **Timestep selection is UNIFORM RANDOM with no weighting** (`multi_stream_data_sampler.py:282-331`). The goal-1 weighted-sampling hook is `reset()`/`_calc_baseperms`; because it's an `IterableDataset`, PyTorch's `WeightedRandomSampler` cannot drop in — weighting must be added inside the sampler.
- **No existing code computes per-cell LATENT forecast error.** The eval pipeline scores in physical space and averages out cells by default (`agg_dims='ipoint'`). The latent error is cheap to add and is the right quantity for our clusters.

---

## 2. The 6 Decisions

### (a) Does the forecast predict next-step LATENT tokens or PHYSICAL fields? — **HIGH**

**Definitively LATENT tokens.** The `ForecastingEngine.forward` (`engines.py:623-636`) applies only self-attention + MLP blocks where *every* block is defined with `self.cf.ae_global_dim_embed` (2048) for **both input and output** (`engines.py:564,580,599`). Input and output tensors are the same `[B,12288,2048]` latent shape. `model.py:703` does `tokens = self.forecast_engine(tokens, step, rope_coords)` and the comment at `model.py:695` says "roll-out in latent space."

The physical prediction is a **separate downstream branch**: after the latent step, `predict_decoders` (`model.py:705`) runs `OriginalPredictionBlock` (`blocks.py:180-263`) + `EnsPredictionHead` to decode latent→physical. The training **loss** (`LossPhysical`, `config_forecasting.yml:170-175`) compares decoded physical preds vs physical targets (`loss_module_physical.py:303-385`).

**Implication for us:** the forecast *prediction* we care about is latent; the *loss* happens to be physical. We can extract the latent prediction directly (free) and compare it to a latent target — perfectly aligned with our 2048-dim clusters. We do **not** need the decoder.

### (b) Dense or MoE today? Where would MoE plug in? — **HIGH**

**Dense today. Definitively no MoE.** A full case-insensitive grep across `src/weathergen/` and `config/` for `moe|mixture.of.experts|num_experts|router|expert|routing|conditional.compute` returned **zero hits**. The only "gating" is AdaLayerNorm's zero-init DiT gate (`norms.py:141-159`); `EnsPredictionHead` runs all heads unconditionally (`engines.py:639-691`).

**MoE insertion point:** the per-block post-attention FFN, `layers.py` `MLP` (`layers.py:31-95`: pre-norm → `Linear(d,d·hidden_factor)` → GELU → `Linear(h,d)`), which sits with a residual in every block (`blocks.py:46-53,137-145`; used at `engines.py:230-239,432-443,509-519,597-607`). `hidden_factor=2` in global/forecast engines, `4` in decoder blocks. A `SwiGLU` (`norms.py:96-102`) is defined but is **dead code** (no imports) — the natural GLU-FFN/MoE replacement slot. The highest-leverage targets are the **`ForecastingEngine` blocks** (`fe_num_blocks=16` in the forecasting config — dominant compute).

### (c) Which exact tensor = the per-cell 2048-dim token? How to tap the encoder output? — **HIGH**

**The tensor is `tokens_global`**, the first element returned by `EncoderModule.forward()` at `encoder.py:140` (`return tokens_global, posteriors`).

Materialization: aggregated per-cell tokens `tokens_global_unmasked` are mask-filled back into the full grid at `encoder.py:345` (`tokens_global[mask] = tokens_global_unmasked`), then reshaped to `[rs, num_tokens_tot, num_queries, 2048].flatten(1,2)` at `encoder.py:350-353`. With defaults (`num_register_tokens=0`, `num_class_tokens=0`, `ae_local_num_queries=1`) → `[rs, 12288, 2048]`; for a single sample/step `rs=1` → `[1,12288,2048]` = the stored `latent`. It is then refined by `ae_global_engine` (`encoder.py:133-138`).

Geometry confirmed: `healpix_level=5` → `num_healpix_cells = 12·4^5 = 12288`, `nside=32` (`encoder.py:44-45`, `default_config.yml:67`); `ae_global_dim_embed=2048` (`default_config.yml:27`). Cell ordering is **NESTED**, set in the **data/tokenizer layer** (`datasets/utils.py:111,167-184,226` `order='nested'`/`nest=True`; `tokenizer.py:106-108`), not in the encoder.

**Tap point:** register a forward hook on `model.encoder.ae_global_engine` — its output (`engines.py:529`) is the `[rs,12288,2048]` grid, identical to the stored `latent` for `rs=1`. Equivalently intercept `tokens` at `model.py:685` right after `self.encoder(...)`.

> Caveat (does not affect cell ordering): `encoder.py:98` calls `healpy.pix2ang(...)` without `nest=`, which defaults to RING — but this branch only runs when `ae_local_queries_per_cell=True`, which is **False by default** (dead code under the default config).

### (d) How to extract per-CELL forecast error — what exists, what to add, in which space — **HIGH (latent route) / MED (physical route)**

**What exists:**
- **Latent side:** the predicted-next-step latent is the variable `tokens` immediately after `tokens = self.forecast_engine(tokens, step, rope_coords)` (`model.py:703`), also saved per step as `latent_state.z_pre_norm` (`model.py:725-726` via `tokens_to_latent_state`, `model.py:660-670`). It is `[B,12288,2048]`. **No decode needed.**
- **Physical side:** the eval pipeline writes per-cell raw pred/target **zarr** (with `ipoint` + lat/lon coords, `dataarray_builders.py:146-151`) **only if `output.num_samples>0`** (defaults to 0, so nothing is written by default — `config/default_config.yml:215-222`). Skill metrics (RMSE/MAE/...) are computed offline in **denormalized physical space** and **averaged over `ipoint` by default** (`score_orchestration.py:182,358-370`), so per-cell error is erased.

**What to add (two options):**

| | **Latent-space error (recommended)** | Physical-space error |
|---|---|---|
| Space | 2048-dim latent (== our clusters) | normalized/denormalized weather channels |
| Prediction | `tokens` after `forecast_engine` (`model.py:703`) — free | decoder output, needs full data pipeline + ERA5 targets |
| Target | `latents_2[t+1]` — already on disk | ERA5 physical fields at t+1 (needs Anemoi reader) |
| Cell join | **Direct** — latents_2 & assignments.pt both NESTED | **Ambiguous** — eval `ipoint` ordering is reader-determined (NESTED vs RING unconfirmed, reader-6 open Q); must map lat/lon→cell→cluster |
| Code to write | ~50-line forward loop over latents_2 through `forecast_engine` | set `num_samples>0` + re-score with `agg_dims` excluding `ipoint`, or post-process zarr |

**Verdict:** the **latent route is strictly better for our purpose** — it lives in the cluster space, needs no decoder, no ERA5 targets, no lat/lon→cell join, and both prediction and target are trivially available. The only missing piece is a forward loop on the supercomputer (see §4).

> **LOW/MED caveat:** the latent route assumes `latents_2[t]` is the **step-0 encoder/analysis state** that feeds `forecast_engine`, not a forecast-step latent. Shape, NESTED ordering, and CLAUDE.md's "encoder trained on ERA5" description strongly support this, but the exact export harness that wrote `latents_2/` was not located (reader-1, reader-2 open Q). Verify on the supercomputer by checking that `forecast_engine(latents_2[t])` decodes (via `predict_decoders`) to a sensible physical field at t+1.

### (e) Training-data sampling: how timesteps are chosen, hook for goal-1 — **HIGH**

**Uniform random shuffle, no weighting.** Each mini-epoch, `MultiStreamDataSampler.reset()` builds `perms = np.arange(max_input_steps, perms_len)` (`_calc_baseperms`, `multi_stream_data_sampler.py:212-218`) then, if `shuffle`, `perms = self.rng.permutation(perms)` (`multi_stream_data_sampler.py:282-331`). `__iter__` walks that permutation. The RNG is seeded from `data_loading.rng_seed` (default `int(time.time())`, `run_train.py:172`), made unique per rank/worker/mini_epoch. There is **no `WeightedRandomSampler`, no chronological mode, no per-timestep or per-file weighting.** Forecasting samples are consecutive timesteps: source = `[base−input+1 … base]`, targets = `base + (time_step·k)//step_timedelta` for k in `[offset, num_output_steps)` (`multi_stream_data_sampler.py:561-611`); with the forecast config (`time_step=6h, num_steps=2, offset=1, policy=fixed`) targets land at `base+1` and `base+2`.

**Goal-1 hook:** `reset()`/`_calc_baseperms` (`multi_stream_data_sampler.py:212-218,282-331`) — replace the uniform `rng.permutation(perms)` with a **weighted draw** over the integer time-window indices, or filter `perms` to a curated timestamp subset. The valid-timestamp universe = `TimeWindowHandler.get_index_range()` (`data_reader_base.py:108-126`) ∩ each stream's `ds.dates`. Because it is an `IterableDataset`, `torch.utils.data.WeightedRandomSampler` **cannot** be dropped in directly — weighting must live inside the sampler. **No config key for per-timestep weights exists today**; the only existing weights are channel-level (`target_channel_weights`, `channel_weights`) and loss-term weights. **You must add the key + the draw.**

### (f) Skill metrics & rollout (single-step vs autoregressive; baselines) — **HIGH**

- **Autoregressive rollout, not single-step.** `model.py:695-707` loops `for step in batch.get_output_idxs(): tokens = self.forecast_engine(...)` then decodes per step. Step cadence 6h (`forecast.time_step=06:00`, `num_steps=2` default). Scoring is per-`forecast_step` independently; `lead_time = valid_time − init_time` (`dataarray_postprocessing.py:88-142`).
- **Metrics implemented** (`packages/evaluate/.../scores/score.py:178-207`): deterministic — rmse, mae, mse, l1/l2, bias, vrmse, **ACC**, ets, pss, fbi, rps, **rpss**, nse, psnr, seeps, qq_analysis, grad_amplitude; probabilistic — crps, spread, ssr, rank_histogram. Default eval config runs only rmse+mae.
- **Baselines: NO persistence baseline exists anywhere.** Skill-vs-climatology is realized only implicitly via **ACC** (consumes a climatology `c`, `score.py:980`) and **RPSS** (`score.py:1103`). A generic `_get_skill_score` (Wilks 7.4, `score.py:38`) is defined but **not wired into the scoring path**. Skill-vs-another-model exists only in plotting (`score_cards.py:237-264`). If you need a persistence/climatology baseline for the error experiment, you must add it.

### (g) Is the encoder/embedding engine frozen during training, or trained jointly? — **HIGH (two recipes)**

**Two distinct recipes live in the repo; only the finetuning one freezes the encoder:**

- **Base / from-scratch forecasting (`config_forecasting.yml:85`): `freeze_modules: ""`** → nothing frozen; the embedding engine + local/global assimilation engines train **jointly** with the forecasting engine.
- **Finetuning (`config_forecasting_finetuning.yml:10`): `freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*latent.*|.*ERA5.*"`** → applied with `re.fullmatch` over `model.named_modules()` (`model/utils.py:47-50`, called at `model/model_interface.py:64`). The pattern freezes the **entire encoder assimilation stack** — `ae_local_engine` (matches `local`), `ae_local_global_engine`/`ae_adapter` (matches `adapter`), `ae_global_engine` (matches `global`), `q_cells` (matches `q_cells`, also forced at `model_interface.py:67-68`), and the `ERA5` stream embeddings (`embed_engine.embeds.*ERA5*`) — **plus `latent_heads`** (matches `latent`). The only top-level modules that do **not** match any alternative and therefore **stay trainable** are `forecast_engine` (`fe_blocks`) and the decoder `pred_heads` (`EnsPredictionHead`). So the finetuning recipe is exactly: **embedding/assimilation engine frozen, forecasting engine (+ decoders) refined.**

**Corroborating signal for `latents_2`:** its date window 2014-01-01 → 2022-11-30 matches the **finetuning** config (`start_date: 2014-01-01`, `end_date: 2022-12-31`, `config_forecasting_finetuning.yml:40-41`), *not* the base config's 1979→2022 (`config_forecasting.yml:142-143`). This is consistent with `latents_2` being encoder outputs from a frozen-encoder finetuning run — i.e. a **fixed embedding** the forecasting engine was trained against. (Caveat: the production checkpoint's actual config is still unconfirmed — see §6.3; a run-specific override could differ.)

**Implication for the cluster analysis:** if `latents_2` is a frozen-encoder output, the 2048-d embedding is a stationary target and our 128 subspaces are a stable coordinate system for the forecast residual. If it were a joint-training snapshot, the encoder co-evolved and the embedding is run-step-dependent. Either way the on-disk tokens are self-consistent for clustering.

> **No reader disagreements.** All six readers are internally HIGH-confidence and mutually consistent; the cross-reader story is coherent. All residual uncertainty is concentrated in the open questions (§6), not in conflicting claims.

---

## 3. Data-Flow Diagram

```
raw ERA5 (Anemoi, dates pre-scoped at open: data_reader_anemoi.py:93-95)
  │
  ▼
[EmbeddingEngine]        per-stream source_tokens → [num_tokens,1024]
  │                      scatter to cell-contiguous order + per-cell PE
  │                      (engines.py:81-132)
  ▼
[LocalAssimilationEngine] varlen flash self-attn WITHIN each HEALPix cell @1024
  │                      (engines.py:241-244; attention.py:107-119)   "local" = within-cell
  ▼
[Local2GlobalAssimilationEngine]  Perceiver cross-attn bottleneck:
  │                      Q = learnable q_cells [1,1,2048]  ×  KV = local 1024
  │                      → one 2048-dim token per unmasked cell   (1024→2048 projection)
  │                      (engines.py:308-316)
  ▼
[QueryAggregationEngine] cross-cell mixing @2048 (num_blocks=0 default ⇒ identity)
  │                      (engines.py:376-452)
  ▼
scatter→grid [rs,12288,2048]  (mask-fill encoder.py:345; reshape :350-353)
  │
  ▼
[GlobalAssimilationEngine] DENSE global self-attn over all 12288 cells @2048, 2 blocks
  │                      (att_dense_rate=1.0 asserted; engines.py:455-529)
  │
  ▼
tokens_global  [rs,12288,2048]      ◄═══ latents_2/latent_{i}.pt [1,12288,2048] ATTACHES HERE
  │  (encoder.py:140 `return tokens_global, posteriors`)   (NESTED HEALPix, nside=32)
  │
  ▼  sum over input steps (model.py:689-691; no-op when num_input_steps=1)
  │
  ▼
[ForecastingEngine]  latent→latent, autoregressive rollout @2048  (fe blocks, engines.py:543-636)
  │   for step in output_idxs:  tokens = forecast_engine(tokens, step, rope_coords)   (model.py:703)
  │   ┌──────────────────────────────────────────────────────────────────────┐
  │   │  ★ PREDICTED-NEXT-STEP LATENT = `tokens` after each step  [12288,2048]│
  │   │  ★ per-cell LATENT forecast error computable HERE (vs latents_2[t+1]) │
  │   └──────────────────────────────────────────────────────────────────────┘
  │
  ├──► predict_decoders: [OriginalPredictionBlock → EnsPredictionHead]  latent→physical
  │        (model.py:705,833; blocks.py:180-263; engines.py:924,949)
  │        ▼  physical preds  [B, cells, channels]   (normalized)
  │        ▼
  │    LossPhysical:  preds.physical vs targets.physical   (loss_module_physical.py:303-385)
  │        ┌──────────────────────────────────────────────────────────────┐
  │        │ ★ per-cell PHYSICAL forecast error computable HERE           │
  │        │   (normalized physical channel space; denorm only at write)   │
  │        └──────────────────────────────────────────────────────────────┘
  │
  └──► predict_latent:  latent_state.z_pre_norm = tokens   (model.py:725-726)
```

**latents_2 attaches** at the encoder output (`tokens_global`). **Per-cell error is computable at two points:** latent (after `forecast_engine`, `model.py:703`) and physical (inside `LossPhysical`). The latent point is the one that matches our 128 clusters.

---

## 4. Concrete Plan for the Error Experiment

**Goal:** a per-cell forecast-error array that joins directly to our 128-cluster `assignments.pt`, for (1) timestamp selection and (2) embedding understanding.

**Key insight that makes this cheap:** since `latents_2/[t]` IS the encoder output that feeds `forecast_engine`, and `latents_2/[t+1]` IS the ground-truth next latent, we need **only the `forecast_engine` forward** on the supercomputer — no encoder, no decoder, no ERA5 reader, no Anemoi.

### Run on the supercomputer

1. **Load the trained checkpoint**, extract `model.forecast_engine` (and confirm its config: `fe_num_blocks`, `ae_global_dim_embed=2048`).
2. **Obtain `rope_coords`** for the 12288 cells — fixed HEALPix NESTED geometry; extract once from any training data sample or compute from NESTED cell→lon/lat (`datasets/utils.py`). Same for all timesteps.
3. **Single-step error (primary), for each t in 0..13019:**
   - `x_t = latents_2[t][0]` → `[12288,2048]` (squeeze batch dim).
   - `pred = forecast_engine(x_t.unsqueeze(0), step=0, rope_coords)[0]` → predicted latent at t+1.
   - `truth = latents_2[t+1][0]`.
   - `err[t, cell] = ||pred[cell] − truth[cell]||₂²`  (per-cell squared residual in 2048-d).
4. **(Optional) multi-step rollout** for lead-time analysis: chain `forecast_engine` k steps, compare to `latents_2[t+k]` — gives `err[t, cell, lead]`. Start with k=1; add k∈{1,2,4,8} if lead-time structure is wanted.
5. **Stream, don't load all:** I/O dominates (~100 MB/file); group reads, batch the forward on GPU. ~13k files at single-step is a few GPU-hours.

### Array to extract to the analysis machine

```
err.npy   shape [13020, 12288]   float32   ≈ 0.6 GB   (single-step)
        + shape [13020, 12288, n_leads]   if multi-step
```

Only the residual array is transferred — not the predicted latents (which would be ~1.2 TB).

### Join to clusters (the easy part)

Our `assignments.pt` stores `cluster[cell]` for `cell ∈ [0,12288)` in **NESTED** HEALPix order (the clustering was run on `latents_2` tokens, which are NESTED — confirmed reader-1 + CLAUDE.md). `err[t,cell]` uses the **same NESTED index** (it's computed from `latents_2`). Therefore the join is a **direct elementwise gather — no lat/lon→cell remapping, no RING↔NESTED conversion**:

```python
# err: [N, 12288],  labels: [12288]  from assignments.pt
for k in range(128):
    cluster_err[k] = err[:, labels == k]   # all residuals for cells in cluster k
```

Then per-cluster mean/variance/distribution, per-timestamp regime signatures (mean error per cluster over the file's cells), and rare-regime/extreme-timestamp upweighting for goal-1.

### Why this is the right error for the clusters

Our clusters are **128 affine subspaces in the 2048-dim latent** (`holdout_eval.py` residual = `‖x−μ_j‖²−‖U_jᵀ(x−μ_j)‖²`). The latent forecast error `‖pred−truth‖²` is the *forecast* analogue in the *same space* — directly attributable. A physical-space error would require an extra decode + an ambiguous cell-index join and would measure a different thing.

---

## 5. MoE Feasibility

**Where MoE enters:** replace the dense post-attention FFN (`layers.py` `MLP`, `layers.py:31-95`) in the transformer blocks. Highest leverage is the **`ForecastingEngine`** (`fe_num_blocks=16` in the forecasting config — the bulk of rollout compute), followed by `GlobalAssimilationEngine`. `SwiGLU` (`norms.py:96-102`) is already defined as dead code — the natural GLU/expert FFN scaffold. Routing would be per-token (per-cell) over the 12288 tokens. The dense global self-attention stays; only the FFN is sparsified (standard "attention dense, FFN MoE" recipe).

**Single biggest risk specific to THIS architecture: the autoregressive latent rollout is supervised only indirectly (through a physical-space loss), so MoE routing instability compounds across steps without a direct latent-space correction signal.**

Concretely: `ForecastingEngine` advances tokens in **latent** space (`model.py:703`), errors compound over rollout steps, but the **loss sees only decoded physical output** (`LossPhysical`, `config_forecasting.yml:170-175`). A small per-step latent routing perturbation (e.g., a cell flipping experts between steps, or load-balancing collapse onto one expert for high-variance storm cells) is invisible to the loss until the decoder amplifies it, and by then it has compounded over several 6h steps into a hard-to-attribute drift. Dense latent rollout + physical-only supervision is the unstable combination; MoE on the latent FFN widens it. Secondary risks: (i) load imbalance driven by the geographically non-uniform variance of atmospheric states (storm vs quiescent cells); (ii) interaction with the existing latent-subspace structure (our 128 clusters) — expert specialization could entangle with cluster boundaries in unpredictable ways.

**Mitigations to evaluate:** add an auxiliary latent-consistency/teacher term (the `LossLatentSSLStudentTeacher` + EMA-teacher path *already exists*, `target_and_aux_ssl_teacher.py:34-164` — wire it in to give direct latent supervision alongside the MoE); router load-balancing + auxiliary-loss; per-step routing-dropout to prevent step-to-step flip-flops.

---

## 6. Open Questions / Things to Verify on the Supercomputer

1. **[BLOCKING for §4] Is `latents_2/[t]` the step-0 encoder/analysis state that feeds `forecast_engine`?** Shape/NESTED ordering match it, but the export harness was not located (reader-1, reader-2 open Q). **Verify:** load a checkpoint, run `forecast_engine(latents_2[t])`, decode via `predict_decoders`, confirm the decoded physical field resembles ERA5 at t+1. If `latents_2` is instead a forecast-step latent, the error formula's target index shifts.
2. **Are the latents deterministic samples or VAE means?** `latent_noise_deterministic_latents=True` is set, and `forward` returns `posteriors` (a VAE posterior) alongside `tokens_global`. Confirm whether re-encoding the same input reproduces `latents_2` bit-for-bit (matters only if you ever re-run the encoder; the §4 plan does not, so this is low-risk).
3. **Which config is the released/production checkpoint** — `default_config.yml` (ae_global 2 blocks, fe 6) vs `config_forecasting.yml` (ae_global 4, fe 16) vs a finetuning config? The `latents_2` tokens have shape `[12288,2048]` consistent with all of them (doesn't disambiguate). Load the actual checkpoint and read its config + `print_num_parameters()` (`model.py:592-658`) — total param count is **not** statically in any config. **Freeze-status hint (see §2g):** `latents_2`'s 2014–2022 window matches `config_forecasting_finetuning.yml` (which freezes the encoder: only `forecast_engine` + `pred_heads` train), not the base config's 1979→2022 window — suggesting a frozen-encoder run; confirm from the checkpoint's saved config.
4. **`forecast_engine` call convention:** exact `step` index for single-step prediction (step 0 vs `output_offset`), and whether any step-embedding/conditioning must be supplied. Confirm from the checkpoint's config + one forward.
5. **`rope_coords` source/format** for the 12288 cells — extract from a real batch's `model_params.rope_coords` (`model.py:703`) to guarantee a match; do not hand-roll.
6. **Eval `ipoint` cell ordering (only if you ever go the physical/eval-zarr route):** reader-6 could not confirm whether eval's `ipoint` is NESTED, RING, or reader-specific (`dataarray_builders.py:146-151` attaches lat/lon but not the convention). The §4 latent route **sidesteps** this entirely — another reason to prefer it.
7. **Does any on-disk run already have validation zarr** (`output.num_samples>0`, default 0)? Check `results/`. If yes, physical per-cell error could be cross-checked cheaply; if no, don't bother (latent route suffices).
8. **Persistence/climatology baselines are absent** from all scoring and config (`score.py` has no persistence path). If goal-1 timestamp selection needs skill-vs-persistence, it must be built — straightforward from stored targets (`targets.physical[t]` used as the t+1 prediction) but net-new code.
9. **`astropy_healpix` not importable in the readers' env** — the claim that `encoder.py:98` defaults to RING rests on the standard healpy API. Moot under default config (`ae_local_queries_per_cell=False`), but if a custom config enables per-cell queries, re-confirm the intended ordering there.
