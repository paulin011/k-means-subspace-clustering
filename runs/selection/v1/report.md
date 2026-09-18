# Selection-signal comparison — top 20% of 13,021 timestamps

*Generated 2026-08-13 19:18 by `compare_selection_signals.py`. Signatures `runs/signatures/v6_d64_margin`, persistence `runs/persistence/v6/err_persist.npy`, proxy `runs/proxy/v1/err_proxy.npy`.*

## What this answers

Every axis below scores each timestamp for 'should I fine-tune on this'. They are not interchangeable. The decisive comparison is **`proxy_err` (a model being wrong) against the latent-geometry axes (`resid_frac`, `zres_mean`, `near_frac`)** — if they agree, latent geometry is an adequate stand-in for model error and the cheap axes suffice; if they disagree, selecting on latent geometry selects the wrong thing. See `docs/ideas/latent_selection.md`.

## Rank correlation (Spearman)

*|ρ| > 0.7 ⇒ the two axes are near-duplicates; near 0 ⇒ they measure different things.*

| | `resid_frac` | `zres_mean` | `n_extreme` | `rare_expo` | `near_frac` | `persist` | `proxy_err` | `proxy_skill` |
|---|---|---|---|---|---|---|---|---|
| **`resid_frac`** | — | +0.12 | +0.02 | -0.26 | +0.80 | +0.19 | -0.04 | +0.22 |
| **`zres_mean`** | +0.12 | — | +0.50 | -0.09 | +0.18 | +0.12 | +0.18 | -0.13 |
| **`n_extreme`** | +0.02 | +0.50 | — | -0.02 | -0.00 | +0.04 | +0.12 | -0.12 |
| **`rare_expo`** | -0.26 | -0.09 | -0.02 | — | -0.20 | +0.17 | +0.11 | -0.00 |
| **`near_frac`** | +0.80 | +0.18 | -0.00 | -0.20 | — | +0.35 | -0.01 | +0.32 |
| **`persist`** | +0.19 | +0.12 | +0.04 | +0.17 | +0.35 | — | +0.67 | -0.08 |
| **`proxy_err`** | -0.04 | +0.18 | +0.12 | +0.11 | -0.01 | +0.67 | — | -0.78 |
| **`proxy_skill`** | +0.22 | -0.13 | -0.12 | -0.00 | +0.32 | -0.08 | -0.78 | — |

## Overlap of the selected top 20%

*Random baseline = 20.0%. Higher ⇒ the two axes would fine-tune on the same data; near the baseline ⇒ they pick almost disjoint sets.*

| | `resid_frac` | `zres_mean` | `n_extreme` | `rare_expo` | `near_frac` | `persist` | `proxy_err` | `proxy_skill` |
|---|---|---|---|---|---|---|---|---|
| **`resid_frac`** | — | 21.8% | 20.5% | 6.6% | 61.1% | 30.3% | 18.2% | 6.6% |
| **`zres_mean`** | 21.8% | — | 47.4% | 19.1% | 23.7% | 24.2% | 28.9% | 28.9% |
| **`n_extreme`** | 20.5% | 47.4% | — | 20.4% | 18.7% | 22.2% | 26.1% | 26.0% |
| **`rare_expo`** | 6.6% | 19.1% | 20.4% | — | 9.2% | 14.0% | 24.4% | 31.1% |
| **`near_frac`** | 61.1% | 23.7% | 18.7% | 9.2% | — | 38.3% | 21.4% | 4.2% |
| **`persist`** | 30.3% | 24.2% | 22.2% | 14.0% | 38.3% | — | 59.4% | 11.2% |
| **`proxy_err`** | 18.2% | 28.9% | 26.1% | 24.4% | 21.4% | 59.4% | — | 46.2% |
| **`proxy_skill`** | 6.6% | 28.9% | 26.0% | 31.1% | 4.2% | 11.2% | 46.2% | — |

## Verdict

`proxy_err` vs the latent-geometry axes: `resid_frac` ρ -0.04 / overlap 18.2%, `zres_mean` ρ +0.18 / overlap 28.9%, `near_frac` ρ -0.01 / overlap 21.4%.

**Latent geometry does NOT track model error** (strongest `zres_mean` only ρ +0.18, overlap 28.9% against a 20% baseline). Selecting on encoder-side statistics selects something other than where the model is wrong — the central claim of `docs/ideas/latent_selection.md`, now measured rather than argued.

`persist` vs `proxy_err`: ρ +0.67, overlap 59.4% — how much of the proxy's error is just 'the weather moved a lot' rather than 'the model failed'.
