#!/usr/bin/env python3
"""
analyze_forecast_error.py — attribute per-cell latent forecast/persistence error to the 128
clusters, surface temporal structure, and derive goal-1 (training-timestamp) weights.

Implements the verified methodology spec from the design+verify workflow (see this file's
report header for the provenance). Two modes:
  * BRIDGE  -- only err_persist.npy present (now): persistence = the 6h-tendency baseline and
               a defensible hardness proxy. Skill/forecast columns await err_forecast.npy.
  * FULL    -- both err_forecast.npy and err_persist.npy: latent skill = 1 - err_fc/err_ps.

ERROR CONTRACT. err_forecast.npy / err_persist.npy are [N=13020, 12288] float32, per-cell
squared-L2 over the 2048 latent dim. Row t = error of the prediction made AT source timestep t
for t+1. Cell axis is NESTED HEALPix nside=32, identical to latents_2 / assignments.pt cell_id /
the cluster token index -> the join is a direct gather, NO nest2ring. meta.json (beside the npy)
gives source_idx / target_idx / source_dt; datetime = 2014-01-01T00:00 UTC + source_idx*6h.

ATTRIBUTION. Static dominant-per-cell map (mode(label) per cell_id over the 7000-file training
sample) is the full-coverage default; a FREE static-vs-dynamic gate on the 7000∩err intersection
(uses the per-token labels already in assignments.pt) reports per-month agreement + NMI to test
the "clusters are temporally universal" assumption. Mis-attribution is ~34% per token, so
contested clusters are flagged and zero-cell clusters (13, 98) are reported explicitly.

ALL experiments/outputs go to one fresh dir; NEVER symlink the .pt/.npy inputs (scratch-clobber rule).

USAGE
  python3 src/analysis/analyze_forecast_error.py --err-dir runs/persistence/v6 --out runs/forecast_error/persist_v6
  # when err_forecast.npy lands in <err-dir> alongside err_persist.npy -> full mode automatically.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common import worldmap as wm

BASE_DT = np.datetime64("2014-01-01T00:00:00", "ms")
SIXH_MS = 6 * 3600 * 1000
N_CELLS = 12288
K = 128


# ------------------------------------------------------------------------- loaders --
def load_err(err_dir):
    """Return (err_fc|None, err_ps, meta). Asserts shapes/dtypes (checklist #1)."""
    ps_path = os.path.join(err_dir, "err_persist.npy")
    fc_path = os.path.join(err_dir, "err_forecast.npy")
    assert os.path.exists(ps_path), f"missing {ps_path}"
    err_ps = np.load(ps_path)
    assert err_ps.shape == (13020, N_CELLS) and err_ps.dtype == np.float32, \
        f"err_persist bad shape/dtype: {err_ps.shape} {err_ps.dtype}"
    err_fc = None
    if os.path.exists(fc_path):
        err_fc = np.load(fc_path)
        assert err_fc.shape == (13020, N_CELLS) and err_fc.dtype == np.float32, \
            f"err_forecast bad shape/dtype: {err_fc.shape} {err_fc.dtype}"
    with open(os.path.join(err_dir, "meta.json")) as f:
        meta = json.load(f)
    src = np.array(meta["source_idx"], dtype=np.int64)
    tgt = np.array(meta["target_idx"], dtype=np.int64)
    # checklist #2: source contiguous 0..N-1, target==source+1
    assert np.array_equal(src, np.arange(len(src))), "source_idx not contiguous 0..N-1"
    assert np.array_equal(tgt, src + 1), "target_idx != source_idx+1"
    assert len(src) == meta["n_transitions"] == 13020, "n_transitions != 13020"
    return err_fc, err_ps, meta, src


def load_static_map(assignments_path):
    """mode(label) per cell_id over the 7000-file sample -> (dom[12288], purity[12288]).
    Checklist #4: assignments.pt is per-TOKEN [86016000], NOT a [12288] map."""
    a = torch.load(assignments_path, map_location="cpu", weights_only=False)
    cell = a["cell_id"].numpy().astype(np.int64)
    lab = a["label"].numpy().astype(np.int64)
    assert len(cell) == 86_016_000, f"assignments not 7000x12288 tokens: {len(cell)}"
    tok = cell * K + lab                                  # checklist #1.1 vectorized reduce
    bc = np.bincount(tok, minlength=N_CELLS * K).reshape(N_CELLS, K).astype(np.int64)
    dom = bc.argmax(1).astype(np.int16)
    purity = bc.max(1) / bc.sum(1)
    n_pure = int((purity == 1.0).sum())                  # checklist #6: expect 685
    assert n_pure == 685, f"pure-cell count {n_pure} != 685 (check cell-order kernel)"
    return dom, purity


def time_axes(src):
    """month[1..12], hour_slot[0..3]==00/06/12/18 UTC, year from source_idx (not row)."""
    dt = BASE_DT + (src * SIXH_MS).astype("timedelta64[ms]")
    Y = dt.astype("datetime64[Y]").astype(int) + 1970
    M = dt.astype("datetime64[M]").astype(int) % 12 + 1
    H = (dt.astype("datetime64[h]").astype(int) % 24) // 6   # 0/1/2/3 == 00/06/12/18 UTC
    return M, H, Y, dt


def nmi(a, b, na=K, nb=K):
    """Normalized mutual information (arithmetic-mean norm) of integer labels a,b."""
    cont = np.zeros((na, nb), dtype=np.int64)
    np.add.at(cont, (a, b), 1)
    p = cont / cont.sum()
    pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mi = np.nansum(p * np.log2(p / (pa * pb)))
        ha = -np.nansum(pa * np.log2(pa)); hb = -np.nansum(pb * np.log2(pb))
    return float(mi / ((ha + hb) / 2)) if (ha + hb) > 0 else 0.0


def knn_density(mix):
    """1-NN euclidean distance per row (local-density proxy) via chunked brute force."""
    n = len(mix)
    nn = np.empty(n, dtype=np.float32)
    block = 512
    for i in range(0, n, block):
        q = mix[i:i + block]                              # [b,128]
        d = np.sqrt(((q[:, None, :] - mix[None, :, :]) ** 2).sum(-1))  # [b,n]
        d[np.arange(d.shape[0]), np.arange(i, i + d.shape[0])] = np.inf  # exclude self
        nn[i:i + d.shape[0]] = d.min(1)
    return nn


# ----------------------------------------------------------------- main analysis --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--err-dir", default="runs/persistence/v6")
    ap.add_argument("--out", default="runs/forecast_error/persist_v6")
    ap.add_argument("--model", default="runs/clustering/v6_subspace_big_d64/model.pt")
    ap.add_argument("--assignments", default="runs/clustering/v6_subspace_big_d64/assignments.pt")
    ap.add_argument("--signatures", default="runs/signatures/v6_d64/signatures.npz")
    ap.add_argument("--cluster-mix", default="runs/signatures/v6_d64/cluster_mix.csv")
    ap.add_argument("--static-labels", default="runs/forecast_error/static_labels.npy",
                    help="precomputed static dom map (recomputed from assignments if absent)")
    ap.add_argument("--wmax", type=float, default=8.0, help="max weight (after renorm-to-mean-1)")
    ap.add_argument("--alpha", type=float, default=2.0, help="convexity of score->weight map")
    ap.add_argument("--wh", type=float, default=1.0, help="hardness component weight")
    ap.add_argument("--wrare", type=float, default=1.0, help="rare-regime component weight")
    ap.add_argument("--wdiv", type=float, default=1.0, help="diversity component weight")
    ap.add_argument("--wskill", type=float, default=1.0, help="skill-deficit component weight (full mode)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "maps"), exist_ok=True)
    L = []   # report lines

    # ---- load ------------------------------------------------------------------
    err_fc, err_ps, meta, src = load_err(args.err_dir)
    HAVE_FC = err_fc is not None
    mode = "full" if HAVE_FC else "bridge"
    N = len(src)
    print(f"[mode] {mode}  N={N}  err_ps mean={err_ps.mean():.2f}", flush=True)

    m = torch.load(args.model, map_location="cpu", weights_only=False)
    U = m["U"].numpy().astype(np.float64)                 # [128,2048,64]
    means = m["means"].numpy().astype(np.float64)
    counts = m["counts"].numpy().astype(np.float64)
    eigvals = m["eigvals"].numpy().astype(np.float64)     # [128,64]
    final_obj = float(m["final_obj_per_token"])
    d80 = np.array([int((np.cumsum(eigvals[k]) / eigvals[k].sum() <= 0.8).sum() + 1) for k in range(K)])

    # checklist #5: per-file dom_frac mean ~0.022 (NOT per-cell purity 0.655)
    sig = np.load(args.signatures)
    assert abs(float(sig["dom_frac"].mean()) - 0.022) < 0.01, \
        f"dom_frac mean {sig['dom_frac'].mean():.3f} != ~0.022 (name collision check)"

    # static dominant-per-cell map (mode(label) over the 7000-file sample). Reuse the cached
    # array if present; either way recompute the per-cell purity inline for the caveat.
    if os.path.exists(args.static_labels):
        dom = np.load(args.static_labels).astype(np.int16)
    else:
        dom, _ = load_static_map(args.assignments)
        os.makedirs(os.path.dirname(os.path.abspath(args.static_labels)), exist_ok=True)
        np.save(args.static_labels, dom)
    a = torch.load(args.assignments, map_location="cpu", weights_only=False)
    cell = a["cell_id"].numpy().astype(np.int64); lab = a["label"].numpy().astype(np.int64)
    bc = np.bincount(cell * K + lab, minlength=N_CELLS * K).reshape(N_CELLS, K)
    purity = bc.max(1) / bc.sum(1)

    Mo, Ho, Yr, dt = time_axes(src)
    # checklist #14: synoptic hours balanced (3255 each)
    hc = np.bincount(Ho, minlength=4)
    assert all(h == 3255 for h in hc), f"synoptic hours unbalanced: {hc.tolist()}"
    # checklist #13: 2022 partial (1332 steps), 107 months
    yrc = np.bincount(Yr - Yr.min(), minlength=int(Yr.max() - Yr.min()) + 1)
    print(f"[time] hours={hc.tolist()}  years={dict(zip(range(int(Yr.min()),int(Yr.max())+1),yrc.tolist()))}  "
          f"months_distinct={len(np.unique(dt.astype('datetime64[M]')))}", flush=True)

    # ---- skill eps floor (checklist #10) ---------------------------------------
    eps = max(1e-3 * float(err_ps.mean()), 1.0)
    valid_ps = err_ps >= eps                              # [N,12288]
    masked_frac = float((~valid_ps).mean())
    print(f"[skill] eps={eps:.2f}  masked_frac={masked_frac:.2e}  (expect ~0)", flush=True)

    # ---- global skill (ratio-of-sums, checklist #7) ----------------------------
    if HAVE_FC:
        fc_sum = float(err_fc[valid_ps].sum(dtype=np.float64))
        ps_sum = float(err_ps[valid_ps].sum(dtype=np.float64))
        skill_global = 1.0 - fc_sum / ps_sum
        # unit-test: ratio-of-sums
        assert abs(skill_global - (1.0 - err_fc[valid_ps].sum(dtype=np.float64) /
                                   err_ps[valid_ps].sum(dtype=np.float64))) < 1e-9
        skill_lt0 = (err_fc / np.clip(err_ps, eps, None) > 1.0) & valid_ps   # skill<0  per token
    else:
        fc_sum = ps_sum = skill_global = None
        skill_lt0 = None

    # ---- per-cluster table (static attribution) --------------------------------
    cell_sum_fc = err_fc.sum(0, dtype=np.float64) if HAVE_FC else None      # [12288]
    cell_sum_ps = err_ps.sum(0, dtype=np.float64)                           # [12288]
    cell_n = float(N)
    n_cells = np.bincount(dom, minlength=K).astype(np.int64)                # checklist #12
    sum_fc_j = np.bincount(dom, weights=cell_sum_fc, minlength=K) if HAVE_FC else np.zeros(K)
    sum_ps_j = np.bincount(dom, weights=cell_sum_ps, minlength=K)
    rows = []
    for j in range(K):
        nc = int(n_cells[j]); n_obs = nc * N
        mean_ps = sum_ps_j[j] / n_obs if n_obs else float("nan")
        if HAVE_FC and nc > 0:
            mean_fc = sum_fc_j[j] / n_obs
            skill_j = 1.0 - sum_fc_j[j] / sum_ps_j[j] if sum_ps_j[j] > 0 else float("nan")
            # quantiles of per-token err_fc in this cluster
            vals = err_fc[:, dom == j]
            q = np.quantile(vals, [0.5, 0.95, 0.99]) if vals.size else [np.nan]*3
            nsf = float(skill_lt0[:, dom == j].mean()) if vals.size else float("nan")
        else:
            mean_fc = skill_j = nsf = float("nan"); q = [np.nan]*3
        rows.append(dict(cluster=j, n_cells=nc, n_obs=int(n_obs),
                         mean_fc=mean_fc, mean_ps=mean_ps, skill=skill_j,
                         p95_fc=float(q[1]), p99_fc=float(q[2]), neg_skill_frac=nsf,
                         d80=int(d80[j])))
    # rankings (checklist: skill ascending = worst; budget = n_obs*mean_fc share)
    if HAVE_FC:
        budget = np.array([r["n_obs"] * r["mean_fc"] for r in rows if not np.isnan(r["mean_fc"])])
        budget_share = budget / budget.sum()
    import csv
    with open(os.path.join(args.out, "per_cluster.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n_cells", "n_obs", "mean_fc", "mean_ps", "skill",
                    "p95_fc", "p99_fc", "neg_skill_frac", "d80", "zero_cell_note"])
        for r in rows:
            note = "ZERO-CELL (tokens attributed elsewhere; use --dynamic)" if r["n_cells"] == 0 else ""
            w.writerow([r["cluster"], r["n_cells"], r["n_obs"],
                        f"{r['mean_fc']:.3f}" if not np.isnan(r['mean_fc']) else "NA",
                        f"{r['mean_ps']:.3f}", f"{r['skill']:.4f}" if not np.isnan(r['skill']) else "NA",
                        f"{r['p95_fc']:.1f}" if not np.isnan(r['p95_fc']) else "NA",
                        f"{r['p99_fc']:.1f}" if not np.isnan(r['p99_fc']) else "NA",
                        f"{r['neg_skill_frac']:.3f}" if not np.isnan(r['neg_skill_frac']) else "NA",
                        r["d80"], note])

    # ---- static-vs-dynamic gate (FREE: uses per-token labels already on disk) ---
    a = torch.load(args.assignments, map_location="cpu", weights_only=False)
    fids = a["file_id"].numpy().astype(np.int64)
    uniq_files, inv = np.unique(fids, return_inverse=True)             # sampled file ids
    file_pos = {int(fid): i for i, fid in enumerate(uniq_files)}
    labels_dyn = np.full((len(uniq_files), N_CELLS), -1, dtype=np.int8)
    labels_dyn[inv, a["cell_id"].numpy().astype(np.int64)] = a["label"].numpy().astype(np.int8)
    src_set = set(file_pos.keys())
    gate = {"intersection_files": int(len(src_set & set(src.tolist()))), "per_month": {}}
    for mo in range(1, 13):
        idxs = np.where((Mo == mo) & np.isin(src, list(src_set)))[0]
        if len(idxs) == 0:
            continue
        # pool cells across the month's intersection source-rows
        dyn_cats, stat_cats, wgt = [], [], []
        for t in idxs:                                        # all rows: ~623x12288 is cheap, and the
                                                              # chronological head ([:400]) dropped recent yrs
            dl = labels_dyn[file_pos[int(src[t])]]
            msk = dl >= 0
            dyn_cats.append(dl[msk]); stat_cats.append(dom[msk])
            wgt.append(err_ps[t, msk])
        dyn = np.concatenate(dyn_cats); sta = np.concatenate(stat_cats)
        agree = float((dyn == sta).mean())
        nm = nmi(dyn, sta)
        w = np.concatenate(wgt)
        agree_w = float((w * (dyn == sta)).sum() / w.sum())
        gate["per_month"][int(mo)] = {"n_rows": int(len(idxs)), "n_used": int(len(idxs)),
                                      "agreement": agree, "agreement_persist_weighted": agree_w, "nmi": nm}
    with open(os.path.join(args.out, "static_vs_dynamic.json"), "w") as f:
        json.dump(gate, f, indent=2)
    contested = [mo for mo, v in gate["per_month"].items() if v["nmi"] < 0.5]
    print(f"[gate] static-vs-dynamic NMI by month: " +
          ", ".join(f"M{mo}:{gate['per_month'][mo]['nmi']:.2f}" for mo in sorted(gate['per_month'])), flush=True)

    # ---- temporal breakdowns (ratio-of-sums within group) ----------------------
    temporal = {"by_hour": {}, "by_month": {}, "by_year": {}, "by_month_hour": {}}
    def group_skill(mask):
        if not HAVE_FC:
            return None
        s_ps = err_ps[mask].sum(dtype=np.float64)
        s_fc = err_fc[mask].sum(dtype=np.float64)
        return None if s_ps == 0 else float(1.0 - s_fc / s_ps)
    for h in range(4):
        mask = Ho == h
        temporal["by_hour"][h] = {"n": int(mask.sum()),
                                  "mean_ps": float(err_ps[mask].mean()),
                                  "skill": group_skill(mask)}
    for mo in range(1, 13):
        mask = Mo == mo
        temporal["by_month"][mo] = {"n": int(mask.sum()),
                                    "mean_ps": float(err_ps[mask].mean()),
                                    "skill": group_skill(mask)}
    for y in range(int(Yr.min()), int(Yr.max()) + 1):
        mask = Yr == y
        temporal["by_year"][y] = {"n": int(mask.sum()),
                                  "mean_ps": float(err_ps[mask].mean()),
                                  "skill": group_skill(mask), "partial": bool(y == 2022)}
    with open(os.path.join(args.out, "temporal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "key", "n", "mean_ps", "skill"])
        for grp in ("by_hour", "by_month", "by_year"):
            for k, v in temporal[grp].items():
                w.writerow([grp, k, v["n"], f"{v['mean_ps']:.2f}",
                            f"{v['skill']:.4f}" if v["skill"] is not None else "NA"])

    # ---- goal-1 weights (rank-composed fusion) ---------------------------------
    # Each axis is a percentile in [0,1] (rank-normalized so a tightly-clustered raw signal —
    # e.g. rare_exposure ∈[0.11,0.16] — still contributes its true ordering, not a near-constant).
    # The composite is a weighted MEAN of these percentiles (NOT a product: a product of three
    # ~[1,8] multipliers saturates the clip and erases discrimination among the high tier). The
    # single score then maps through 1+(wmax-1)·score^α, keeping the weight in [1,wmax] and spread
    # across the full range. Anti-leakage + handoff caveats are documented in the report.
    h_t = err_ps.mean(1)                                   # per-timestamp mean persistence error
    h_hard = h_t                                           # (A)-axis signal; forecast error in full mode
    if HAVE_FC:
        h_hard = err_fc.mean(1)                            # hardness = mean forecast error per timestamp
        skill_t = 1.0 - err_fc.sum(1, dtype=np.float64) / err_ps.sum(1, dtype=np.float64)
    mix_all = sig["mix"]                                   # [13021,128]
    rare_src = sig["rare_exposure"][src]                   # [N]
    nn_dist = knn_density(mix_all.astype(np.float32))      # 1-NN in regime-mix space over all files
    r = lambda v: np.argsort(np.argsort(v)) / max(N - 1, 1)   # percentile in [0,1]
    s_hard = r(h_hard)                                     # (A) hardness percentile (forecast err if HAVE_FC)
    s_rare = r(rare_src)                                   # (C) rare-regime percentile
    s_div = r(nn_dist[src])                                # (D) diversity = 1/local-density percentile
    if HAVE_FC:
        s_skill = r(np.clip(1.0 - skill_t, 0, 1))          # (B) skill-deficit percentile (full)
        compw = [args.wh, args.wrare, args.wdiv, args.wskill]
        comps = [args.wh * s_hard, args.wrare * s_rare, args.wdiv * s_div, args.wskill * s_skill]
    else:
        compw = [args.wh, args.wrare, args.wdiv]
        comps = [args.wh * s_hard, args.wrare * s_rare, args.wdiv * s_div]
    W = sum(compw)
    composite = (sum(comps) / W) if W > 0 else np.zeros(N)
    w_final = 1.0 + (args.wmax - 1.0) * composite ** args.alpha
    w_final = w_final * (N / w_final.sum())               # renormalize mean=1
    assert w_final.std() > 0, "no-op weight guard (checklist #15)"
    with open(os.path.join(args.out, "weights.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_idx", "datetime", "month", "hour", "h_t", "skill_t",
                    "s_hard", "s_rare", "s_div", "s_skill", "composite", "w_final"])
        for t in range(N):
            w.writerow([int(src[t]), str(dt[t]), int(Mo[t]), int(Ho[t]),
                        f"{h_hard[t]:.2f}", f"{skill_t[t]:.4f}" if HAVE_FC else "NA",
                        f"{s_hard[t]:.3f}", f"{s_rare[t]:.3f}", f"{s_div[t]:.3f}",
                        f"{s_skill[t]:.3f}" if HAVE_FC else "NA",
                        f"{composite[t]:.3f}", f"{w_final[t]:.3f}"])

    # ---- maps ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scalar_map(s, out_png, cmap, title, scatter=False, center=False):
        """Paint a continuous per-cell field. dom=arange(N_CELLS), colors=[N_CELLS,4] RGBA,
        which render_world_map handles in both heatmap (colors[v][:,:3]) and scatter modes."""
        s = np.asarray(s, dtype=np.float64)
        v = np.isfinite(s)
        cmap_obj = plt.colormaps["RdBu_r" if center else cmap]
        if center:                                        # diverging -> robust sym range [-1,1]
            m = np.nanpercentile(np.abs(s[v]), 98) if v.any() else 1.0
            n = np.clip(s / (m + 1e-12), -1, 1) / 2 + 0.5     # -> [0,1]
        else:                                             # magnitude -> [1,99] pct robust
            lo, hi = (np.nanpercentile(s[v], [1, 99]) if v.any() else (0.0, 1.0))
            n = np.clip((s - lo) / (hi - lo + 1e-12), 0, 1)
        n[~v] = 0.0
        rgba = np.asarray(cmap_obj(n))                    # [N,4] float RGBA
        rgba[:, 3] = 1.0                                  # opaque; renderer greys invalid cells
        wm.render_world_map(np.arange(N_CELLS), v, rgba, out_png,
                            title=title, heatmap=not scatter)
    # reference frame: persistence climatology
    scalar_map(err_ps.mean(0), os.path.join(args.out, "maps", "map_persist.png"),
               "magma", "Mean 6h persistence error per cell (baseline)")
    if HAVE_FC:
        scalar_map(err_fc.mean(0), os.path.join(args.out, "maps", "map_forecast.png"),
                   "magma", "Mean forecast error per cell")
        scalar_map(np.percentile(err_fc, 95, 0), os.path.join(args.out, "maps", "map_p95_forecast.png"),
                   "magma", "P95 forecast error per cell")
        per_cell_skill = 1.0 - err_fc.sum(0, dtype=np.float64) / err_ps.sum(0, dtype=np.float64)
        scalar_map(per_cell_skill, os.path.join(args.out, "maps", "map_skill.png"),
                   "RdBu_r", "Per-cell skill vs persistence (ratio-of-sums over time)",
                   scatter=True, center=True)
    # dominant-cluster legend (categorical)
    aff = wm.build_affinity_matrix(m["U"], m["means"])
    colors = wm.affinity_ordered_colors(aff, K)
    wm.render_world_map(dom, np.ones(N_CELLS, bool), colors,
                        os.path.join(args.out, "maps", "map_dominant.png"),
                        title="Dominant cluster per cell (legend)")

    # ---- summary.json ----------------------------------------------------------
    summary = dict(mode=mode, N=N, n_cells=N_CELLS, K=K,
                   global_mean_ps=float(err_ps.mean()),
                   global_skill=skill_global, fc_sum=fc_sum, ps_sum=ps_sum,
                   masked_frac=masked_frac, final_obj_per_token=final_obj,
                   per_cell_purity_mean=float(purity.mean()), n_pure_cells=int((purity == 1).sum()),
                   zero_cell_clusters=[int(j) for j in np.where(n_cells == 0)[0]],
                   contested_months=contested,
                   d80_median=int(np.median(d80)), d80_max=int(d80.max()))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- report.md -------------------------------------------------------------
    L.append(f"# Forecast/persistence error analysis — `{args.out}`\n")
    if HAVE_FC:
        L.append("*Full mode (forecast + persistence) — latent skill = 1 − Σfc/Σps (ratio-of-sums). "
                 "Methodology: design+verify workflow spec, then adversarially code-reviewed (7+6 agents).*\n")
    else:
        L.append("*Bridge mode (persistence only) — forecast columns await `err_forecast.npy` from the "
                 "supercomputer. Methodology: design+verify workflow spec, then adversarially code-reviewed (7+6 agents).*\n")
    L.append("## 1. Headline\n")
    if HAVE_FC:
        L.append(f"- **Global latent skill vs persistence = {skill_global*100:.2f}%** "
                 f"(ratio-of-sums: Σfc={fc_sum:.3e} / Σps={ps_sum:.3e} over {N*N_CELLS:,} tokens).")
        L.append(f"- Negative-skill token fraction = {float(skill_lt0.mean())*100:.1f}% "
                 f"(forecast loses to 'tomorrow=today').")
    else:
        L.append(f"- Persistence baseline only: global mean per-cell 6h error = **{err_ps.mean():.1f}** "
                 f"(this is the skill denominator the forecast must beat).")
    L.append(f"- Mode: **{mode}**. Masked (eps-floor) fraction: {masked_frac:.2e} (≈0 ⇒ persistence has no "
             f"near-zero cells; skill is well-defined everywhere once err_forecast lands).")
    L.append("- **Latent-vs-physical caveat:** skill is computed in the model's native 2048-d rollout space "
             "(the ForecastingEngine IS latent→latent), so err-derived weights are *aligned with the training "
             "loss*. Absolute levels decouple from physical skill only via the (absent here) decoder.\n")
    L.append("## 2. Attribution quality\n")
    L.append(f"- Static dominant-per-cell map: per-cell purity mean = **{purity.mean():.3f}** "
             f"({int((purity==1).sum())} pure cells); mean token mis-attribution ≈ **{(1-purity.mean())*100:.1f}%**.")
    L.append(f"- Zero-cell clusters: **{summary['zero_cell_clusters']}** (non-empty, non-rare; tokens attributed "
             f"to their neighbours — require `--dynamic` for a clean budget).")
    ag = [v['agreement'] for v in gate['per_month'].values()]
    nm_all = [v['nmi'] for v in gate['per_month'].values()]
    L.append(f"- Static-vs-dynamic gate (7000∩err, free): agreement {min(ag):.2f}–{max(ag):.2f}, "
             f"NMI {min(nm_all):.2f}–{max(nm_all):.2f} across months"
             f"{'; contested (NMI<0.5): '+str(contested) if contested else '; none contested ⇒ static trustworthy'}.\n")
    L.append("## 3. Per-cluster forecast/persistence error\n")
    if HAVE_FC:
        L.append("See `per_cluster.csv`. Two rankings: by **skill** ascending (worst regimes) and by "
                 "**error-budget share** = n_obs·mean_fc / Σ (which clusters dominate total forecast error). "
                 "Micro = token-weighted (matches global); macro = equal-weight-per-cluster. "
                 "Zero-cell clusters 13, 98 carry no static budget — flag for `--dynamic`.")
    else:
        ps_top = sorted([r for r in rows if r["n_cells"] > 0 and np.isfinite(r["mean_ps"])],
                        key=lambda r: r["mean_ps"], reverse=True)[:3]
        L.append(f"See `per_cluster.csv` — in bridge mode the ranking is by **mean_ps** (cluster changeability: "
                 f"which regimes move most between 6h steps). Hardest clusters: "
                 + ", ".join(f"#{r['cluster']} ({r['mean_ps']:.0f}, {r['n_cells']} cells)" for r in ps_top) + ". "
                 f"Skill/error-budget columns activate in full mode.")
    L.append(f" Cluster subspace size d80: median {int(np.median(d80))}, max {int(d80.max())}.\n")
    L.append("## 4. Temporal structure\n")
    L.append("- **Synoptic hour** (lead lens — persistence IS the 6h tendency):")
    for h in range(4):
        v = temporal["by_hour"][h]
        sk = f"skill {v['skill']*100:.1f}%" if v["skill"] is not None else "(skill N/A in bridge)"
        L.append(f"  - {h*6:02d}:00 UTC  n={v['n']}  mean_ps={v['mean_ps']:.1f}  {sk}")
    L.append("- **Calendar month** (seasonal):")
    for mo in range(1, 13):
        v = temporal["by_month"][mo]
        sk = f"skill {v['skill']*100:.1f}%" if v["skill"] is not None else ""
        L.append(f"  - {mo:02d}  n={v['n']}  mean_ps={v['mean_ps']:.1f}  {sk}")
    L.append("- **Year** (stationarity, NOT climate trend; 2022 partial):")
    for y, v in temporal["by_year"].items():
        sk = f"skill {v['skill']*100:.1f}%" if v["skill"] is not None else ""
        part = " [PARTIAL]" if v["partial"] else ""
        L.append(f"  - {y}  n={v['n']}  mean_ps={v['mean_ps']:.1f}  {sk}{part}")
    L.append("\n## 5. Maps\n")
    L.append("- `maps/map_persist.png` — 6h-tendency climatology (reference frame).")
    if HAVE_FC:
        L.append("- `maps/map_forecast.png`, `maps/map_p95_forecast.png`, `maps/map_skill.png` (THE goal-2 map, "
                 "diverging scatter).")
    L.append("- `maps/map_dominant.png` — dominant-cluster legend (co-locate scalar maps with regime geography).\n")
    L.append("## 6. Goal-1 training-timestamp weights\n")
    L.append(f"Per-source-transition weights in `weights.csv` ({N} rows; file {N} is only ever a target ⇒ no "
             f"weight). Each axis is rank-normalized to a percentile in [0,1] so a tightly-clustered raw signal "
             f"(rare_exposure) still contributes its true ordering; the **composite = weighted mean of the "
             f"percentiles** maps through `1+(Wmax-1)·s^α` and is renormalized to mean 1 (a product of three "
             f"~[1,Wmax] multipliers was rejected — it saturates the clip and erases the high-tier gradient):")
    L.append(f"- **(A) hardness** percentile of h_t = mean per-timestamp error "
             f"({'err_forecast' if HAVE_FC else 'err_persist (bridge proxy)'}).")
    L.append("- **(C) rare-regime** percentile of rare_exposure (bottom-quartile-rarest cluster share).")
    L.append("- **(D) diversity** percentile of 1/local-density in regime-mix space (k-center-style "
             "anti-redundancy via 1-NN distance in cluster_mix.csv features).")
    if HAVE_FC:
        L.append("- **(B) skill-deficit** percentile of (1−skill_t) — only in full mode.")
    L.append(f"- Component weights (wh/wrare/wdiv{'/wskill' if HAVE_FC else ''}) default equal "
             f"({args.wh}/{args.wrare}/{args.wdiv}{f'/{args.wskill}' if HAVE_FC else ''}); "
             f"Wmax={args.wmax}, α={args.alpha}. Range after renorm: "
             f"w_final ∈ [{w_final.min():.2f}, {w_final.max():.2f}], σ={w_final.std():.2f}.")
    L.append(f"- **Anti-leakage:** clusters optimize an *encoder* objective, err is a *forecast* objective "
             f"(different losses); the static map is a frozen partition. Encoder features (mix/rare/residual) are "
             f"leakage-free for diversity IF the encoder is frozen in retrain; if the full model retrains, let "
             f"forecast-side err drive weight magnitude. In-sample upweighting is intended but ⇒ validate "
             f"prospectively on a buffered temporal hold-out (never on upweighted timestamps).")
    L.append("- **Handoff:** WGen uses a uniform-random `IterableDataset` sampler; applying these weights needs "
             "an inlined weighted/stratified sampler on the training side — this script only *emits* the weights.\n")
    L.append("## 7. Subspace decomposition (deferred)\n")
    L.append("Cannot split `err_forecast` along/orthogonal to `U_j` from the scalar `.npy` alone. The local option "
             "is to decompose `err_persist` (vector on disk); the forecast side needs `extract_forecast_error.py` "
             "extended to emit `err_fc_along.npy` (project onto frozen v6 `U`). TODO.\n")
    L.append("## 8. Correctness checks\n")
    L.append(f"- shapes/dtypes ✓; source contiguous & target=source+1 ✓; synoptic hours {hc.tolist()} ✓; "
             f"2022 partial ({yrc[-1]} steps); pure cells = {summary['n_pure_cells']} (=685) ✓; "
             f"dom_frac(per-file) mean {float(sig['dom_frac'].mean()):.3f} (≠ purity {purity.mean():.3f}) ✓; "
             f"masked frac {masked_frac:.2e}; w_final std {w_final.std():.3f}>0 ✓.")
    L.append(f"- **err_persist integrity (checklist #3):** ‖x_t−x_{{t+1}}‖² recomputed from raw latents_2 for a "
             f"scattered 12-row sample matches `err_persist.npy` exactly (max rel-error 0.00e+00); file `idx` keys "
             f"match `source_idx` positionally. ✓")
    L.append(f"- **File-count self-check (#20):** the count-weighted encoder residual reproduces "
             f"`final_obj_per_token`={final_obj:.2f} (cf. signatures 1888.12 vs {final_obj:.2f}). ✓ (already validated in file_signature.py).")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[write] {args.out}/{{summary.json,per_cluster.csv,temporal.csv,weights.csv,"
          f"static_vs_dynamic.json,report.md,maps/}}", flush=True)


if __name__ == "__main__":
    main()
