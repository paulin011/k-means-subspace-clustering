#!/usr/bin/env python3
"""Do the candidate timestamp-selection signals actually rank the record differently?

THE QUESTION `docs/ideas/latent_selection.md` leaves open. Selecting the best ~20% of the
13,021 timestamps to fine-tune on requires a score per timestamp. Several are available and
they are NOT interchangeable -- §3.3 of that doc already measured that the three axes in
`analyze_forecast_error.py`'s `weights.csv` pick near-disjoint sets (`hard` and `rare`
overlap at 6.7% against a 20% random baseline, i.e. anti-correlated). This script puts every
candidate on the same footing and reports the pairwise rank correlation and the top-q overlap.

The one that matters most is **proxy model error** (`runs/proxy/*/err_proxy.npy` from
`proxy_forecast.py`). Every other axis is a statistic of the *encoding*; that one is a
statistic of a *model being wrong*, which is the distinction the selection doc argues is
decisive. If it ranks the record much like latent hardness, then latent geometry is an
adequate stand-in for model error and the cheap axes are fine. If it ranks differently, then
selecting on latent geometry is selecting on the wrong thing, and the conclusion holds.

AXES (all per timestamp, all rank-normalised before comparison):
  resid_frac    `signatures.npz` -- mean subspace residual / total. The current hardness axis.
  zres_mean     per-cell robust z (median/MAD) averaged over the file -- hardness with the
                geographic scale divided out (`docs/LATENT_OUTLIERS.md` §4).
  n_extreme     count of cells above the robust-z event threshold -- the per-cell/event view
                that §3.2 argues should replace per-file means.
  rare_expo     `rare_exposure` -- share of tokens in bottom-quartile-rare clusters.
  near_frac     share of near-tie tokens -- assignment ambiguity.
  persist       mean `err_persist` -- 6 h latent tendency. Model-free difficulty baseline.
  proxy_err     mean `err_proxy` -- the proxy forecaster's squared error. THE model-error axis.
  proxy_skill   1 - proxy_err/persist per file -- where the proxy BEATS persistence, i.e.
                genuinely predictable vs genuinely hard. Sign matters: low skill = hard.

Outputs `<out>/{signal_table.csv, report.md}`.

Usage:
  python3 src/analysis/compare_selection_signals.py --out runs/selection/v1
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`

N_CELLS = 12288


def rankn(v):
    """Rank-normalise to [0,1]; NaNs to the middle so they never win or lose a selection."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    out = np.full(v.shape, 0.5)
    r = np.argsort(np.argsort(v[ok]))
    out[ok] = r / max(len(r) - 1, 1)
    return out


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signatures", default="runs/signatures/v6_d64_margin")
    ap.add_argument("--probe-cache", default="runs/clustering/v6_subspace_big_d64/probe/cache.npz")
    ap.add_argument("--persist", default="runs/persistence/v6/err_persist.npy")
    ap.add_argument("--proxy", default="runs/proxy/v1/err_proxy.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=float, default=0.2, help="selection fraction")
    ap.add_argument("--zthr", type=float, default=5.58, help="robust-z event threshold")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    sig = np.load(f"{args.signatures}/signatures.npz")
    N = len(sig["mean_residual"])
    dts = sig["datetime_hour"]
    df = pd.DataFrame(dict(file_id=np.arange(N), datetime=dts.astype("datetime64[h]"),
                           resid_frac=sig["residual_frac"], rare_expo=sig["rare_exposure"],
                           near_frac=sig["near_frac"]))

    # --- per-cell robust z: hardness with the geographic scale divided out ---
    print("robust-z pass over residual_map ...", flush=True)
    R = np.load(f"{args.signatures}/residual_map.npy", mmap_mode="r")
    c = np.load(args.probe_cache)
    med, mad = c["cell_med"], np.maximum(c["cell_mad"], 1e-6)
    zm = np.empty(N)
    nx = np.empty(N)
    for a in range(0, N, 1000):
        b = min(a + 1000, N)
        Z = (np.asarray(R[a:b]) - med[None]) / (1.4826 * mad[None])
        zm[a:b] = Z.mean(1)
        nx[a:b] = (Z > args.zthr).sum(1)
    df["zres_mean"], df["n_extreme"] = zm, nx

    # --- persistence (rows are transitions t -> t+1, so the last file has none) ---
    P = np.load(args.persist, mmap_mode="r")
    pv = np.full(N, np.nan)
    pv[:P.shape[0]] = np.asarray(P).mean(1)
    df["persist"] = pv

    # --- proxy model error (the axis this whole comparison exists for) ---
    have_proxy = os.path.exists(args.proxy)
    if have_proxy:
        Q = np.load(args.proxy, mmap_mode="r")
        qv = np.full(N, np.nan)
        qv[:Q.shape[0]] = np.asarray(Q).mean(1)
        df["proxy_err"] = qv
        with np.errstate(invalid="ignore", divide="ignore"):
            df["proxy_skill"] = 1.0 - df["proxy_err"] / df["persist"]
        print(f"proxy: global skill vs persistence = "
              f"{1 - np.nansum(qv) / np.nansum(pv):+.4f}", flush=True)
    else:
        print(f"NOTE {args.proxy} absent -- run proxy_forecast.py; comparing without it.")

    axes = [a for a in ("resid_frac", "zres_mean", "n_extreme", "rare_expo", "near_frac",
                        "persist", "proxy_err", "proxy_skill") if a in df]
    df.to_csv(f"{args.out}/signal_table.csv", index=False)

    # --- pairwise rank correlation + top-q overlap ---
    S = pd.DataFrame(index=axes, columns=axes, dtype=float)
    O = pd.DataFrame(index=axes, columns=axes, dtype=float)
    k = int(args.q * N)
    tops = {}
    for a in axes:
        v = df[a].values.copy()
        if a == "proxy_skill":                 # LOW skill = hard, so invert for selection
            v = -v
        tops[a] = set(np.argsort(-np.nan_to_num(v, nan=-np.inf))[:k])
    for a in axes:
        for b in axes:
            S.loc[a, b] = spearman(df[a].values, df[b].values)
            O.loc[a, b] = len(tops[a] & tops[b]) / k

    L = []
    add = L.append
    add(f"# Selection-signal comparison — top {args.q:.0%} of {N:,} timestamps\n")
    add(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `compare_selection_signals.py`. "
        f"Signatures `{args.signatures}`, persistence `{args.persist}`"
        + (f", proxy `{args.proxy}`.*\n" if have_proxy else ", **no proxy error**.*\n"))
    add("## What this answers\n")
    add("Every axis below scores each timestamp for 'should I fine-tune on this'. They are "
        "not interchangeable. The decisive comparison is **`proxy_err` (a model being wrong) "
        "against the latent-geometry axes (`resid_frac`, `zres_mean`, `near_frac`)** — if "
        "they agree, latent geometry is an adequate stand-in for model error and the cheap "
        "axes suffice; if they disagree, selecting on latent geometry selects the wrong "
        "thing. See `docs/ideas/latent_selection.md`.\n")
    add("## Rank correlation (Spearman)\n")
    add("*|ρ| > 0.7 ⇒ the two axes are near-duplicates; near 0 ⇒ they measure different things.*\n")
    add("| | " + " | ".join(f"`{a}`" for a in axes) + " |")
    add("|---" * (len(axes) + 1) + "|")
    for a in axes:
        add(f"| **`{a}`** | " + " | ".join("—" if a == b else f"{S.loc[a, b]:+.2f}"
                                           for b in axes) + " |")
    add(f"\n## Overlap of the selected top {args.q:.0%}\n")
    add(f"*Random baseline = {args.q:.1%}. Higher ⇒ the two axes would fine-tune on the same "
        f"data; near the baseline ⇒ they pick almost disjoint sets.*\n")
    add("| | " + " | ".join(f"`{a}`" for a in axes) + " |")
    add("|---" * (len(axes) + 1) + "|")
    for a in axes:
        add(f"| **`{a}`** | " + " | ".join("—" if a == b else f"{O.loc[a, b]:.1%}"
                                           for b in axes) + " |")

    if have_proxy:
        geo = [a for a in ("resid_frac", "zres_mean", "near_frac") if a in axes]
        rho = {a: S.loc["proxy_err", a] for a in geo}
        ov = {a: O.loc["proxy_err", a] for a in geo}
        best = max(rho, key=lambda a: abs(rho[a]))
        add("\n## Verdict\n")
        add(f"`proxy_err` vs the latent-geometry axes: "
            + ", ".join(f"`{a}` ρ {rho[a]:+.2f} / overlap {ov[a]:.1%}" for a in geo) + ".\n")
        if abs(rho[best]) >= 0.7:
            add(f"**Latent geometry tracks model error** (strongest `{best}`, ρ {rho[best]:+.2f}). "
                f"The cheap latent axes are an adequate stand-in; the checkpoint route buys "
                f"ranking accuracy, not a different answer.")
        else:
            add(f"**Latent geometry does NOT track model error** (strongest `{best}` only "
                f"ρ {rho[best]:+.2f}, overlap {ov[best]:.1%} against a {args.q:.0%} baseline). "
                f"Selecting on encoder-side statistics selects something other than where the "
                f"model is wrong — the central claim of `docs/ideas/latent_selection.md`, now "
                f"measured rather than argued.")
        add(f"\n`persist` vs `proxy_err`: ρ {S.loc['persist', 'proxy_err']:+.2f}, overlap "
            f"{O.loc['persist', 'proxy_err']:.1%} — how much of the proxy's error is just "
            f"'the weather moved a lot' rather than 'the model failed'.")

    open(f"{args.out}/report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[6:]))
    print(f"\nwrote {args.out}/report.md and signal_table.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
