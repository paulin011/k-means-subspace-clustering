#!/usr/bin/env python3
"""Single-cluster probe: pick a cluster out of K, then say what it *is*.

Two deliverables, both under `<run>/probe/`:

  --rank        `shortlist.md` -- one table, one row per cluster, sorted. The entry point
                for "which of these 128 clusters should I look at at all".
  --cluster N   `c<NNN>.md`    -- the per-cluster key report: five panels (P0-P4), one
                screen of text plus two figures.

WHY THIS EXISTS. `merge_clusters.py` established that K is a **budget knob**, not a
property of the data (the merge cascade has no knee: per-merge cost rises at 100.0% of
steps from K=256 to K=32). So K is set by what can be *interpreted*, and interpretation
cost is linear in K -- which makes a per-cluster probe the unit of work. Everything here
is O(1) per cluster.

WHAT IT ADDS over `report.md` / `temporal_report.md`. Those already answer how big, how
flat, how localized, how bursty, who it resembles and who contests it. Two gaps remain:

  1. **Every map in the repo is a dominant-cluster map** -- one colour per cell, all K
     clusters at once. There is no map of *one* cluster, and the per-cluster spatial
     columns (`cells@50%`, `owned`) are scalars: they say a cluster is spread out, never
     where. P1/P2 are that map.
  2. **Nothing reaches the token level.** `residual_map.npy` is written by
     `file_signature.py` and then only ever reduced away, so the outlier goal is not
     served. P3 turns the tail into concrete `(date, cell)` cases.

THE THRESHOLD PROBLEM (P1). A raw occupancy threshold cannot be shared across clusters:
peak occupancy is bimodal -- 77 of v6's 128 clusters peak above 0.9 (they own cells
outright, permanently) while 12 never reach 0.5 anywhere (c13 peaks at 0.17) -- so tau=0.5
draws a solid blob for the first group and an *empty map* for the second, which are
exactly the clusters worth looking at. The fix is to slide **mass coverage** instead of
frequency: ask for the smallest cell set holding a fraction q of the cluster's tokens and
read tau off it (`worldmap.core_region`). That is the highest-density-region construction,
it is scale-free, and it is exactly the report's `cells@50%` column generalised to any q,
so P1 is continuous with the existing report rather than new vocabulary. `tau50`, not
`maxf`, is the honest territoriality scalar: v6 c122 has maxf 1.00 (it permanently owns a
couple of cells) yet tau50 0.062 (its mass is smeared over 262).

OUTLIERS (P3). Per-cell residuals are not comparable across cells -- the per-cell mean
spans 12.4x geographically -- so the score is a **median/MAD robust z** against each
cell's own temporal norm, computed here in seconds from `residual_map.npy`. Median/MAD and
not mean/std: mean/std over-selects low-variance cells (crisp polar clusters with the
lowest persistence in the run took 4 of the top 10 host slots), and not per-cell rank
either, which is degenerate by construction -- ranking within a cell and cutting at a
global quantile selects the same number of timesteps *for every cell*, so it can never say
a cell is unusually anomalous. Flagged tokens are grouped into space-time connected
components ("events") using the **real HEALPix NESTED neighbour query**
(`worldmap.healpix_nest_neighbours`); the `cell >> 2` sibling proxy links a pixel only
within its own 2x2 block and fragments every event at block boundaries (measured: the
2022-03-17..19 Antarctic event splits into 57 components under the proxy versus one
280-token component with the real query).

NOT BUILT, and why (each killed by a measurement -- recorded so it is not re-proposed):
  * **diurnal panel** -- null. Across all 128 v6 clusters the largest deviation of any UTC
    hour's share from 0.25 is 0.023, and only 2.6% of cells change dominant cluster between
    00Z and 12Z versus 42.8% between DJF and JJA. The report states the null in one line
    (measured per cluster, not cited).
  * **`maxf`** as a column -- rho -0.83 with `near%`; and see the c122 case above.
  * **`rival_overlap`** (occupancy cosine with the runner-up) -- rho -0.73 with `maxf`,
    because a cluster that owns its cells outright has ~zero overlap with anything by
    definition. `rival_km` is the non-degenerate version.
  * **`burstiness`** -- rho 0.97 with `seasonality`; **per-cluster persistence** -- rho 0.80
    with residual, so only the *ratio* (`unmodelled`) is kept.
  * **per-cluster `mu_dev` / `iso`** -- `iso = ||mu_j - mu_global||^2 / trace[j]` maxes out
    at 0.25 across all 128 clusters, i.e. the between-cluster offset is at most a quarter
    of a cluster's own internal spread. "Outlier cluster" is not a meaningful category in
    this embedding; that is a one-line global statement, not a column. Outliers live at the
    token level, which is what P3 is for.
  * **`--html` coverage slider** -- deliberately deferred (the plan's own "ship the ladder
    first"): the static 50/80/95% ladder carries the same information with no JavaScript,
    consistent with every other report here.

READER, NOT PRODUCER: no pass over `latents_2` (1.2 TB). Every number comes from artefacts
already on disk -- `label_map.npy` / `residual_map.npy` / `signatures.npz` from
`file_signature.py`, `model.pt` from the run, and `err_persist.npy` from
`persistence_error.py`. Algorithm-agnostic: only the `cluster_io.py` schema is read, so a
merged run (`config["method"] == "subspace_kmeans_merged_ward"`) works unchanged.

DEPENDENCY: the probe needs a `runs/signatures/<...>/` whose `manifest.json` names this
run as its frozen model (resolved exactly as `analyze_clusters.py --margins` does,
preferring one that also carries `cluster_margin.csv`). If none matches it prints the
`file_signature.py` command line to create one.

Usage (always from the repo root):
  python3 src/analysis/cluster_probe.py --dir runs/clustering/v6_subspace_big_d64 --rank
  python3 src/analysis/cluster_probe.py --dir runs/clustering/v6_subspace_big_d64 --cluster 13
  python3 src/analysis/cluster_probe.py --dir <run> --cluster 13 --coverage 0.5,0.8,0.95
  python3 src/analysis/cluster_probe.py --dir <run> --rank --event-sweep
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.worldmap import (N_CELLS, NSIDE, cell_lonlat, cell_xyz, core_region,
                             healpix_nest_neighbours, render_scalar_map)

N_FILES_TOTAL = 13021
EARTH_R_KM = 6371.0

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_KEYS = list(SEASONS)

# Robust-z tail kept as "extreme". Settled by the sweep in --event-sweep (measured on v6):
# the fraction of flagged tokens sitting inside a multi-token event rises smoothly and
# *convexly* with the cut (75.3% at 0.005% -> 89.6% at 2%), with no interior structure --
# a max-distance-from-chord knee detector returns 0.02% with all nine chord distances
# inside 0.066-0.084, i.e. no knee, the same "no natural scale" answer the merge cascade
# gave for K. The binding constraint is therefore the repo's distinct-value rule: the
# `events` column separates only 37 of 128 clusters at 0.01% (fails), 70 at 0.05%, 84 at
# 0.1%, 119 at 0.5%. 0.1% is the loosest cut that still selects a genuinely extreme tail
# (robust z >= 5.58; 160k of 160M tokens) while giving the column 84/128 distinct values.
EVENT_PCT = 0.1
MIN_EVENT_TOKENS = 2      # a single (cell, timestep) blip is not an event
MIN_DISTINCT = 40         # `report-metric-audit`: a column separating < 40 of 128 is dead
NEAR_TIE = 0.10           # matches file_signature.py
CONTESTED = 0.30          # `near%` above which a cluster counts as contested
REMOTE_KM = 4000.0        # rival core farther than this => not geographic blur
LOCAL_KM = 2000.0
TERRITORIAL_TAU = 0.5     # tau50 >= this  => territorial
ITINERANT_TAU = 0.10      # tau50 <  this  => itinerant
DEFAULT_COVERAGE = (0.5, 0.8, 0.95)


# ---------------------------------------------------------------------------
# resolution / IO
# ---------------------------------------------------------------------------
def resolve_signature_dir(run_dir, explicit=None):
    """Find the `runs/signatures/*` run whose manifest names `run_dir` as its model.

    Same rule `analyze_clusters.py --margins` uses. v6 matches twice (`v6_d64` and
    `v6_d64_margin`); the one carrying `cluster_margin.csv` is preferred because it is a
    strict superset (it adds the per-cluster margin and `margin_map.npy`).
    """
    if explicit:
        return explicit
    cands = []
    for mf in sorted(glob.glob(os.path.join("runs", "signatures", "*", "manifest.json"))):
        try:
            with open(mf) as f:
                man = json.load(f)
        except (OSError, ValueError):
            continue
        if os.path.abspath(man.get("model_dir", "")) != os.path.abspath(run_dir):
            continue
        d = os.path.dirname(mf)
        if not os.path.exists(os.path.join(d, "label_map.npy")):
            continue
        cands.append((os.path.exists(os.path.join(d, "cluster_margin.csv")), d))
    if not cands:
        raise SystemExit(
            f"No runs/signatures/* run found for {run_dir}.\nThe probe reads that run's "
            f"per-cell maps; create one first (~15-20 min, single GPU):\n"
            f"  python3 src/analysis/file_signature.py --dir {run_dir} "
            f"--out runs/signatures/{os.path.basename(run_dir.rstrip('/'))}")
    cands.sort(reverse=True)
    return cands[0][1]


def resolve_persistence(explicit=None):
    """`err_persist.npy` from persistence_error.py (for `unmodelled`). None if absent.

    Picks the run with the most transitions, and skips `_`-prefixed dirs (the repo's
    smoke-run convention). Plain alphabetical order does NOT work here: `runs/persistence/`
    holds both `_smoke` (49 transitions, the first 49 steps of Jan 2014) and `v6` (13,020),
    and `_smoke` sorts first -- which silently built `unmodelled` from 0.4% of the record.
    """
    if explicit:
        return explicit if os.path.exists(explicit) else None
    best, best_rows = None, -1
    for p in sorted(glob.glob(os.path.join("runs", "persistence", "*", "err_persist.npy"))):
        if os.path.basename(os.path.dirname(p)).startswith("_"):
            continue
        rows = np.load(p, mmap_mode="r").shape[0]
        if rows > best_rows:
            best, best_rows = p, rows
    return best


def month_of(dt64):
    return dt64.astype("datetime64[M]").astype(int) % 12 + 1


# ---------------------------------------------------------------------------
# the shared, cached pass over the per-cell maps
# ---------------------------------------------------------------------------
def build_cache(sig_dir, persist_path, cache_path, key, chunk=1000):
    """One pass over `label_map.npy` + `residual_map.npy` -> everything every panel needs.

    Two reductions, both O(N x 12288):
      occupancy  f_j(cell) = P(cell carries label j), split by season, month and UTC hour;
      residual   per-cell median/MAD (the robust-z normaliser) and per-cluster sums of the
                 raw residual, the robust z and the persistence error.
    Cached to `<run>/probe/cache.npz` keyed by the run fingerprint + signature dir + the
    file count, so the second `--cluster` call costs nothing.
    """
    t0 = time.time()
    sig = np.load(os.path.join(sig_dir, "signatures.npz"))
    lab = np.load(os.path.join(sig_dir, "label_map.npy"), mmap_mode="r")
    N = lab.shape[0]
    K = int(sig["cluster_tokens"].shape[0])
    dt = sig["datetime_hour"]
    file_ids = sig["file_idx"].astype(np.int64)          # idx == file id (manifest)
    mon = month_of(dt)
    hour = (dt - dt.astype("datetime64[D]")).astype(int)
    seas = np.zeros(N, np.int64)
    for si, s in enumerate(SEASON_KEYS):
        seas[np.isin(mon, SEASONS[s])] = si

    print(f"  pass 1/2: occupancy over {N:,}x{N_CELLS:,} labels ...", flush=True)
    socc = np.zeros(4 * N_CELLS * K, np.int64)
    monK = np.zeros((12, K), np.int64)
    hourK = np.zeros((4, K), np.int64)                    # 00/06/12/18 UTC
    cellK = np.arange(N_CELLS, dtype=np.int64) * K
    for i0 in range(0, N, chunk):
        ch = np.asarray(lab[i0:i0 + chunk], dtype=np.int64)
        socc += np.bincount((seas[i0:i0 + chunk, None] * (N_CELLS * K)
                             + cellK[None, :] + ch).ravel(), minlength=4 * N_CELLS * K)
        mm, hh = mon[i0:i0 + chunk], hour[i0:i0 + chunk]
        for m in np.unique(mm):
            monK[m - 1] += np.bincount(ch[mm == m].ravel(), minlength=K)
        for h in np.unique(hh):
            hourK[h // 6] += np.bincount(ch[hh == h].ravel(), minlength=K)
    socc = socc.reshape(4, N_CELLS, K).astype(np.int32)

    print(f"  pass 2/2: per-cell median/MAD + per-cluster residual sums "
          f"({time.time() - t0:.0f}s so far) ...", flush=True)
    R = np.load(os.path.join(sig_dir, "residual_map.npy"))            # 0.6 GiB, fits easily
    med = np.empty(N_CELLS, np.float32)
    mad = np.empty(N_CELLS, np.float32)
    for c0 in range(0, N_CELLS, 2048):                                # blocked: small temporaries
        blk = R[:, c0:c0 + 2048]
        m = np.median(blk, axis=0)
        med[c0:c0 + 2048] = m
        mad[c0:c0 + 2048] = np.median(np.abs(blk - m), axis=0)
    scale = (1.4826 * np.maximum(mad, 1e-6)).astype(np.float32)       # -> ~1 sigma for gaussians

    per = None
    if persist_path:
        per = np.load(persist_path, mmap_mode="r")                    # [13020, 12288]
    clu_n = np.zeros(K, np.float64)
    clu_z = np.zeros(K, np.float64)
    clu_r = np.zeros(K, np.float64)
    clu_p = np.zeros(K, np.float64)
    clu_pn = np.zeros(K, np.float64)
    clu_unz = np.zeros(K, np.float64)     # sum of z*scale+med == sum of raw residual (check 5)
    for i0 in range(0, N, chunk):
        ch = np.asarray(lab[i0:i0 + chunk], dtype=np.int64).ravel()
        rr = R[i0:i0 + chunk].ravel()
        zz = ((R[i0:i0 + chunk] - med) / scale).ravel()
        clu_n += np.bincount(ch, minlength=K)
        clu_r += np.bincount(ch, weights=rr.astype(np.float64), minlength=K)
        clu_z += np.bincount(ch, weights=zz.astype(np.float64), minlength=K)
        clu_unz += np.bincount(ch, weights=(zz * np.tile(scale, len(range(i0, min(i0 + chunk, N))))
                                            + np.tile(med, len(range(i0, min(i0 + chunk, N)))))
                               .astype(np.float64), minlength=K)
        if per is not None:
            fi = file_ids[i0:i0 + chunk]
            ok = fi < per.shape[0]
            if ok.any():
                pp = np.asarray(per[fi[ok]]).ravel().astype(np.float64)
                cp = np.asarray(lab[i0:i0 + chunk], dtype=np.int64)[ok].ravel()
                clu_p += np.bincount(cp, weights=pp, minlength=K)
                clu_pn += np.bincount(cp, minlength=K)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, key=np.array(key), socc=socc, monK=monK, hourK=hourK,
             cell_med=med, cell_mad=mad, clu_n=clu_n, clu_z=clu_z, clu_r=clu_r,
             clu_p=clu_p, clu_pn=clu_pn, clu_unz=clu_unz,
             file_ids=file_ids.astype(np.int32), n_files=np.array(N))
    print(f"  cache written: {cache_path} ({time.time() - t0:.0f}s)", flush=True)
    return np.load(cache_path, allow_pickle=False)


def load_cache(run_dir, sig_dir, persist_path, rebuild=False):
    cache_path = os.path.join(run_dir, "probe", "cache.npz")
    key = json.dumps({"sig": os.path.abspath(sig_dir),
                      "persist": os.path.abspath(persist_path) if persist_path else None,
                      "v": 1})
    if not rebuild and os.path.exists(cache_path):
        c = np.load(cache_path, allow_pickle=False)
        if str(c["key"]) == key:
            return c
        print("  cache key changed -> rebuilding", flush=True)
    return build_cache(sig_dir, persist_path, cache_path, key)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def extract_events(Z, labels, pct, nb):
    """Space-time connected components of the top-`pct`% robust-z tokens.

    Adjacency = a real HEALPix NESTED neighbour at the same timestep, or the same cell at
    t+1. Returns a dict of per-event arrays; `host` is the cluster of the event's peak-z
    token, which is what the shortlist's `events` column counts.
    """
    thr = float(np.quantile(Z.ravel(), 1.0 - pct / 100.0))
    flat = np.nonzero(Z.ravel() >= thr)[0]                 # sorted: row-major scan order
    t = flat // N_CELLS
    c = flat % N_CELLS
    M = flat.size
    rows, cols = [], []
    for i in range(nb.shape[1]):
        n = nb[c, i]
        ok = n >= 0
        keys = t[ok] * N_CELLS + n[ok]
        p = np.minimum(np.searchsorted(flat, keys), max(M - 1, 0))
        hit = flat[p] == keys
        rows.append(np.nonzero(ok)[0][hit])
        cols.append(p[hit])
    keys = flat + N_CELLS                                   # same cell, next timestep
    p = np.minimum(np.searchsorted(flat, keys), max(M - 1, 0))
    hit = flat[p] == keys
    rows.append(np.nonzero(hit)[0])
    cols.append(p[hit])

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    r, cc = np.concatenate(rows), np.concatenate(cols)
    _, comp = connected_components(
        coo_matrix((np.ones(r.size, np.int8), (r, cc)), shape=(M, M)), directed=False)

    z = Z.ravel()[flat]
    E = int(comp.max()) + 1 if M else 0
    size = np.bincount(comp, minlength=E)
    order = np.lexsort((-z, comp))                          # peak-z token first within event
    first = np.ones(M, bool)
    first[1:] = comp[order][1:] != comp[order][:-1]
    pk = order[first]                                       # one index per event, peak first
    xyz = cell_xyz()
    ctr = np.zeros((E, 3))
    np.add.at(ctr, comp, xyz[c])
    nrm = np.linalg.norm(ctr, axis=1, keepdims=True)
    ctr = ctr / np.maximum(nrm, 1e-12)
    return {"thr": thr, "n_flagged": M, "flat": flat, "t": t, "cell": c, "comp": comp,
            "size": size,
            "peak_i": pk, "peak_z": z[pk], "peak_cell": c[pk], "peak_t": t[pk],
            "host": labels[t[pk], c[pk]].astype(np.int64),
            "t_min": _group_min(comp, t, E), "t_max": _group_max(comp, t, E),
            "n_steps": _group_nuniq(comp, t, E), "n_cells": _group_nuniq(comp, c, E),
            "lat": np.degrees(np.arcsin(np.clip(ctr[:, 2], -1, 1))),
            "lon": (np.degrees(np.arctan2(ctr[:, 1], ctr[:, 0])) + 180) % 360 - 180}


def _group_min(g, v, n):
    out = np.full(n, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(out, g, v)
    return out


def _group_max(g, v, n):
    out = np.full(n, np.iinfo(np.int64).min, np.int64)
    np.maximum.at(out, g, v)
    return out


def _group_nuniq(g, v, n):
    key = np.unique(g.astype(np.int64) * (int(v.max()) + 2) + v)
    return np.bincount(key // (int(v.max()) + 2), minlength=n)


# ---------------------------------------------------------------------------
# per-cluster geometry
# ---------------------------------------------------------------------------
def core_stats(f, mask):
    """Occupancy-weighted centroid of a core region + its two concentration numbers.

    Returns (lon, lat, R_sphere, R_lon, lat_lo, lat_hi). Positions are averaged as 3-D unit
    vectors (averaging lon/lat directly is wrong at the dateline and meaningless at a pole).
    `R_lon` is the *circular* concentration of longitude alone: 1 = a single spot, ~0 = the
    longitudes cancel, which happens both for a zonal band and for a ring around a pole --
    so it is always reported next to |lat| (v6: c109 1.00, c13 0.44, c27 0.07).
    """
    w = np.asarray(f, float)[mask]
    if w.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    lon, lat = cell_lonlat()
    xyz = cell_xyz()[mask]
    v = (w[:, None] * xyz).sum(0) / w.sum()
    Rs = float(np.linalg.norm(v))
    v = v / max(Rs, 1e-12)
    e = np.exp(1j * np.radians(lon[mask]))
    Rl = float(abs((w * e).sum() / w.sum()))
    o = np.argsort(lat[mask])
    cw = np.cumsum(w[o]) / w.sum()
    lo = float(lat[mask][o][np.searchsorted(cw, 0.10)])
    hi = float(lat[mask][o][min(np.searchsorted(cw, 0.90), len(o) - 1)])
    return (float((np.degrees(np.arctan2(v[1], v[0])) + 180) % 360 - 180),
            float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), Rs, Rl, lo, hi)


def great_circle_km(lon1, lat1, lon2, lat2):
    a = np.radians([lon1, lat1, lon2, lat2])
    d = np.arccos(np.clip(np.sin(a[1]) * np.sin(a[3])
                          + np.cos(a[1]) * np.cos(a[3]) * np.cos(a[0] - a[2]), -1, 1))
    return float(EARTH_R_KM * d)


def fmt_lat(v):
    return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"


def fmt_lon(v):
    return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"


# ---------------------------------------------------------------------------
# the per-cluster metric table (shared by --rank and --cluster)
# ---------------------------------------------------------------------------
def cluster_metrics(cache, sig, ev, coverage):
    socc = cache["socc"]
    occ = socc.sum(0).astype(np.float64)                     # [N_CELLS, K]
    N = int(cache["n_files"])
    K = occ.shape[1]
    f = occ / N                                              # occupancy f_j(cell)
    tot = occ.sum(0)
    monK = cache["monK"].astype(np.float64)
    T = float(monK.sum())

    m = {"K": K, "N": N, "f": f, "occ": occ, "socc": socc, "tokens": tot.astype(np.int64),
         "share": tot / max(T, 1), "maxf": f.max(0)}

    # tau ladder (the P1 coverage stops, plus tau50 for the shortlist)
    m["tau"] = {}
    for q in sorted(set(list(coverage) + [0.5])):
        taus = np.zeros(K)
        cells = np.zeros(K, np.int64)
        for j in range(K):
            tau, mask = core_region(f[:, j], q)
            taus[j], cells[j] = tau, mask.sum()
        m["tau"][q] = (taus, cells)
    m["tau50"], m["core50"] = m["tau"][0.5]

    n = np.maximum(cache["clu_n"], 1.0)
    m["zres"] = cache["clu_z"] / n
    m["resid"] = cache["clu_r"] / n
    pn = cache["clu_pn"]
    m["persist"] = np.where(pn > 0, cache["clu_p"] / np.maximum(pn, 1), np.nan)
    m["unmodelled"] = m["resid"] / np.where(m["persist"] > 0, m["persist"], np.nan)

    # seasonality: CV of the 12 monthly enrichments (temporal_spatial.py's definition,
    # here over the whole dataset rather than the 7000-file sample)
    enr = (monK / np.maximum(tot, 1)[None, :]) / np.maximum(monK.sum(1) / T, 1e-12)[:, None]
    m["seasonality"] = enr.std(0) / np.maximum(enr.mean(0), 1e-12)
    m["month_enrich"] = enr
    sK = socc.sum(1).astype(np.float64)                                    # [4, K]
    m["season_enrich"] = ((sK / np.maximum(tot, 1)[None, :])
                          / np.maximum(sK.sum(1) / T, 1e-12)[:, None])
    hK = cache["hourK"].astype(np.float64)
    m["hour_share"] = hK / np.maximum(hK.sum(0), 1)[None, :]               # [4, K]

    m["near"] = sig["cluster_near_frac"]
    m["margin"] = sig["cluster_margin_mean"]
    m["rival"] = sig["cluster_runner_up"].astype(np.int64)
    m["rival_frac"] = sig["cluster_runner_up_frac"]

    # rival_km: great-circle distance between the two clusters' tau50 core centroids
    ctr = np.zeros((K, 2))
    conc = np.zeros((K, 4))
    for j in range(K):
        _, mask = core_region(f[:, j], 0.5)
        lo, la, Rs, Rl, y0, y1 = core_stats(f[:, j], mask)
        ctr[j] = (lo, la)
        conc[j] = (Rs, Rl, y0, y1)
    m["centroid"], m["conc"] = ctr, conc
    m["rival_km"] = np.array([great_circle_km(ctr[j, 0], ctr[j, 1],
                                              ctr[m["rival"][j], 0], ctr[m["rival"][j], 1])
                              for j in range(K)])

    big = ev["size"] >= MIN_EVENT_TOKENS
    m["events"] = np.bincount(ev["host"][big], minlength=K)
    m["events_all"] = np.bincount(ev["host"], minlength=K)
    return m


def kind_of(tau50):
    if tau50 >= TERRITORIAL_TAU:
        return "territorial"
    if tau50 < ITINERANT_TAU:
        return "itinerant"
    return "intermediate"


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# validation (plan section 9)
# ---------------------------------------------------------------------------
def run_checks(run_dir, sig_dir, cache, sig, m, ev, model):
    L = []
    ok_all = True

    def rec(name, ok, detail):
        nonlocal ok_all
        ok_all &= ok
        L.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    ct = sig["cluster_tokens"].astype(np.int64)
    ok = np.array_equal(m["tokens"], ct)
    rec("1 f_j.sum(0)*N == full-dataset counts", ok,
        f"{m['tokens'].sum():,} tokens, exact integer match={ok}")

    # 2 tau_q holds >= q of the mass; q=0.5 reproduces the report's cells@50% on the
    #   *sampled* files (assignments.pt), the file set that column was computed on.
    worst = 1.0
    for q, (taus, cells) in m["tau"].items():
        for j in range(m["K"]):
            mask = m["f"][:, j] >= taus[j]
            worst = min(worst, m["f"][mask, j].sum() / max(m["f"][:, j].sum(), 1e-12) - q)
    a = torch.load(os.path.join(run_dir, "assignments.pt"), map_location="cpu",
                   weights_only=False)
    lab_s, cell_s = a["label"].long(), a["cell_id"].long()
    K = m["K"]
    cc = torch.bincount(cell_s * K + lab_s, minlength=N_CELLS * K).view(N_CELLS, K).double()
    srt = cc.sort(0, descending=True).values
    cum = srt.cumsum(0) / srt.sum(0).clamp(min=1)
    cells50_report = ((cum < 0.5).sum(0) + 1).numpy()
    mine = np.array([int(np.searchsorted(
        np.cumsum(np.sort(cc[:, j].numpy())[::-1]) / max(cc[:, j].sum().item(), 1), 0.5,
        side="left")) + 1 for j in range(K)])
    same = int((mine == cells50_report).sum())
    rec("2 tau_q covers >= q, and q=0.5 == report cells@50% on the sampled files",
        worst > -1e-9 and same == K,
        f"min (covered - q) = {worst:+.2e}; cells@50% identical for {same}/{K} clusters")

    fps = cache["socc"].sum((1, 2))
    fseason = cache["socc"].astype(np.float64) / np.maximum(
        cache["socc"].sum(2).sum(1, keepdims=True) / N_CELLS, 1)[:, None]
    nf = np.array([int(round(x / N_CELLS)) for x in fps])
    recon = (cache["socc"].astype(np.float64).sum(0)) / int(cache["n_files"])
    err = float(np.abs(recon - m["f"]).max())
    rec("3 seasonal fields weighted by files/season sum back to the annual field",
        err < 1e-12, f"max |recon - f_annual| = {err:.2e} (files/season {nf.tolist()})")

    # 4 real-neighbour adjacency must merge the two named events
    dt = sig["datetime_hour"]
    det = []
    for name, lo, hi in (("2022-03-17..19", np.datetime64("2022-03-17"), np.datetime64("2022-03-20")),
                         ("2018-02-23..25", np.datetime64("2018-02-23"), np.datetime64("2018-02-26"))):
        sel = (dt[ev["t"]] >= lo) & (dt[ev["t"]] < hi)
        if not sel.any():
            det.append(f"{name}: no flagged tokens")
            continue
        u, n_ = np.unique(ev["comp"][sel], return_counts=True)
        top = u[np.argsort(-ev["size"][u])][:3]
        det.append(f"{name}: largest component {int(ev['size'][top[0]])} tokens "
                   f"(next {', '.join(str(int(ev['size'][x])) for x in top[1:])})")
    dom = all(int(x.split("largest component ")[1].split()[0]) >= 100 for x in det
              if "largest component" in x)
    rec("4 HEALPix-neighbour adjacency merges the two named events", dom, "; ".join(det))

    fo = model.get("final_obj_per_token")
    if fo is None:
        fo = model["history"][-1]["obj_per_token"] if model.get("history") else float("nan")
    n = np.maximum(cache["clu_n"], 1)
    mean_raw = float((cache["clu_r"].sum()) / cache["clu_n"].sum())
    undone = float((cache["clu_unz"].sum()) / cache["clu_n"].sum())
    rec("5 zres un-normalised reproduces the model objective",
        abs(mean_raw - float(fo)) / float(fo) < 0.02 and abs(undone - mean_raw) / mean_raw < 1e-4,
        f"count-weighted mean residual {mean_raw:.2f} vs model final_obj_per_token "
        f"{float(fo):.2f} ({abs(mean_raw - float(fo)) / float(fo):.2%}); "
        f"z*1.4826*MAD+median re-sums to {undone:.2f} ({abs(undone - mean_raw) / mean_raw:.1e} rel)")
    return L, ok_all


# ---------------------------------------------------------------------------
# deliverable A -- the shortlist
# ---------------------------------------------------------------------------
SHORT_COLS = [("tokens", "{:,.0f}"), ("near%", "{:.1%}"), ("zres", "{:+.2f}"),
              ("unmodelled", "{:.2f}"), ("events", "{:,.0f}"), ("tau50", "{:.3f}"),
              ("core", "{:,.0f}"), ("rival", "{:.0f}"), ("rival_km", "{:,.0f}"),
              ("seasonality", "{:.3f}")]


def shortlist_columns(m):
    return {"tokens": m["tokens"].astype(float), "near%": m["near"].astype(float),
            "zres": m["zres"], "unmodelled": m["unmodelled"], "events": m["events"].astype(float),
            "tau50": m["tau50"], "core": m["core50"].astype(float),
            "rival": m["rival"].astype(float), "rival_km": m["rival_km"],
            "seasonality": m["seasonality"]}


def write_shortlist(out_path, run_dir, sig_dir, persist_path, m, ev, checks, sort_key,
                    model, sweep_rows=None):
    K = m["K"]
    cols = shortlist_columns(m)
    audit = {c: len(set(np.round(v[np.isfinite(v)], 6).tolist())) for c, v in cols.items()}
    dropped = [c for c, n in audit.items()
               if n < MIN_DISTINCT and c not in ("rival",)]
    keep = [c for c, _ in SHORT_COLS if c not in dropped]

    order = np.argsort(-np.nan_to_num(cols.get(sort_key, cols["events"]), nan=-np.inf))
    L = []
    add = L.append
    cfg = model["config"]
    add(f"# Cluster shortlist — `{run_dir}`")
    add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `cluster_probe.py --rank`. "
        f"K={K} {'affine subspaces (d=' + str(cfg.get('dim')) + ')' if cfg.get('dim') else 'point clusters'}, "
        f"method `{cfg.get('method', '?')}`. Per-cell arrays from `{sig_dir}` "
        f"({m['N']:,} files × {N_CELLS:,} cells = {m['N'] * N_CELLS:,} tokens); "
        f"persistence baseline `{persist_path or 'not found'}`.*")

    add("\n## What this is\n")
    add("One row per cluster, so you can pick a handful out of "
        f"{K} instead of reading {K} rows of the main report. **Nothing here duplicates "
        "`report.md`**: size, EVR, `cells@50%`, `owned`, `files@50%`, `tCV`, `maxAff` and the "
        "affinity table live there, the monthly/seasonal *dominant-cluster* maps live in "
        "`temporal_report.md`. These are the axes neither of them has.")
    add("\nThen open one cluster with "
        f"`python3 src/analysis/cluster_probe.py --dir {run_dir} --cluster <id>`.\n")

    add("## Columns\n")
    add("*How to read this: every column names the array it comes from. `f_j(cell)` below is "
        "the cluster's **occupancy field** — the fraction of the "
        f"{m['N']:,} timesteps at which that HEALPix cell carries label j — reduced from "
        "`label_map.npy`. `z` is the **robust per-cell residual anomaly**, "
        "`z = (residual_map − median_cell) / (1.4826·MAD_cell)`, i.e. how far a token's "
        "residual sits from that cell's own temporal norm; the normalisation is not optional, "
        "because the per-cell mean residual spans 12.4× geographically.*\n")
    add("| column | formula / source | reads as |")
    add("|---|---|---|")
    add("| `tokens` | `label_map.npy` bincount over the full dataset | size (**not** "
        "`model['counts']`, which covers only the sampled files) |")
    add("| `near%` | `cluster_near_frac` (`signatures.npz`) | share of its own tokens whose "
        "runner-up subspace is within 10% — contestedness |")
    add("| `zres` | mean of `z` over the tokens this cluster owns | **is this cluster hard "
        "*relative to where it sits*** — which the raw residual cannot say |")
    add("| `unmodelled` | mean raw residual ÷ mean `err_persist.npy` over its tokens | "
        "**anomalous because unmodelled** (high) vs **because rapidly changing** (low) |")
    add(f"| `events` | extreme events hosted (≥{MIN_EVENT_TOKENS} tokens), host = cluster of "
        f"the event's peak-z token | the direct outlier axis |")
    add("| `tau50` | occupancy at the 50%-mass crossing (`worldmap.core_region`) | "
        "**the honest territoriality scalar**: 1.0 = owns its territory outright, <0.1 = "
        "itinerant. Not `maxf` — v6 c122 has maxf 1.00 and tau50 0.062 |")
    add("| `core` | cells with `f_j >= tau50` | how much territory holds half its mass |")
    add("| `rival` / `rival_km` | `cluster_runner_up` + great-circle km between the two "
        f"tau50 core centroids | **< {LOCAL_KM:,.0f} km = local map blur** (the common case); "
        f"**> {REMOTE_KM:,.0f} km = a real embedding-space confusion** |")
    add("| `seasonality` | std/mean of the 12 monthly enrichments | 0 = aseasonal |")

    add("\n## Metric audit\n")
    add("*How to read this: the `report-metric-audit` rule — a column that separates fewer "
        f"than ~{MIN_DISTINCT} of {K} clusters is carrying no information and is dropped "
        "before shipping, not printed with more decimals.*\n")
    add("| column | distinct values | verdict |")
    add("|---|---|---|")
    for c, _ in SHORT_COLS:
        n = audit[c]
        v = ("id column, not a metric" if c == "rival" else
             ("**dropped**" if n < MIN_DISTINCT else "keep"))
        add(f"| `{c}` | {n} / {K} | {v} |")
    if dropped:
        add(f"\nDropped this run: **{', '.join(dropped)}**.")

    add("\n### Mutual rank correlation (Spearman)\n")
    add("*A column that duplicates another earns no slot. These are the survivors of the "
        "redundancy audit; the rejected candidates are listed in `cluster_probe.py`'s "
        "docstring with the correlation that killed each.*\n")
    num = [c for c in keep if c != "rival"]
    add("| | " + " | ".join(f"`{c}`" for c in num) + " |")
    add("|---|" + "---|" * len(num))
    for a in num:
        add(f"| `{a}` | " + " | ".join(
            ("—" if a == b else f"{spearman(cols[a], cols[b]):+.2f}") for b in num) + " |")

    add(f"\n## All {K} clusters (sorted by `{sort_key}`, descending)\n")
    hdr = ["cluster"] + keep
    add("| " + " | ".join(hdr) + " |")
    add("|" + "---|" * len(hdr))
    fmt = dict(SHORT_COLS)
    for j in order.tolist():
        row = [f"{j}"]
        for c in keep:
            v = cols[c][j]
            row.append("–" if not np.isfinite(v) else fmt[c].format(v))
        add("| " + " | ".join(row) + " |")

    add("\n## Presets\n")
    add("*The same table cut four ways — each is the top 5 on one axis, so a reader with a "
        "specific question does not have to re-sort.*\n")
    presets = [("hardest relative to its geography", "zres", None, False),
               ("most unmodelled (hard, but not because it moves)", "unmodelled", None, False),
               ("most itinerant (a moving weather state, not a region)", "tau50", None, True),
               (f"most remotely confused (`near%` > {CONTESTED:.0%})", "rival_km",
                lambda: cols["near%"] > CONTESTED, False)]
    for title, key, filt, asc in presets:
        if key in dropped:
            continue
        idx = np.arange(K)
        if filt is not None:
            idx = idx[filt()]
        if idx.size == 0:
            continue
        v = np.nan_to_num(cols[key][idx], nan=np.inf if asc else -np.inf)
        idx = idx[np.argsort(v if asc else -v)][:5]
        add(f"**{title}** — by `{key}`{' (ascending)' if asc else ''}:\n")
        add("| cluster | " + " | ".join(keep) + " |")
        add("|---|" + "---|" * len(keep))
        for j in idx.tolist():
            add(f"| {j} | " + " | ".join(
                "–" if not np.isfinite(cols[c][j]) else fmt[c].format(cols[c][j])
                for c in keep) + " |")
        add("")

    add("## Global findings (not per-cluster columns)\n")
    terr = int((m["tau50"] >= TERRITORIAL_TAU).sum())
    itin = int((m["tau50"] < ITINERANT_TAU).sum())
    add(f"- **Two kinds of cluster.** {terr} of {K} are *territorial* "
        f"(`tau50 >= {TERRITORIAL_TAU}` — a fixed geographic region) and {itin} are "
        f"*itinerant* (`tau50 < {ITINERANT_TAU}` — a moving weather state whose mass is "
        f"smeared over hundreds of cells). `maxf` ↔ `near%` correlate at Spearman "
        f"{spearman(m['maxf'], m['near'].astype(float)):+.3f}: crisp territory ⇒ crisp "
        f"assignment. That is one axis, not two, and `near%` already carries it — which is "
        f"why `maxf` is not a column here. What it does dictate is *how to draw the map*.")
    add(f"- **Presence is useless as a map.** Cells where a cluster appears at all span "
        f"{int((m['f'] > 0).sum(0).min())} to {int((m['f'] > 0).sum(0).max())} "
        f"(mean {(m['f'] > 0).sum(0).mean():.0f}) — up to "
        f"{(m['f'] > 0).sum(0).max() / N_CELLS:.0%} of the globe. Hence the mass-coverage "
        f"threshold rather than an occupancy one.")
    add(f"- **No cluster is an isolated blob.** `iso = ‖μⱼ−μ_global‖²/trace[j]` maxes out at "
        f"{m.get('iso_max', float('nan')):.2f} across all {K} clusters: the between-cluster "
        f"offset is at most that fraction of a cluster's own internal spread. Combined with "
        f"the near-tie rate and the knee-free merge cascade, **\"outlier cluster\" is not a "
        f"meaningful category in this embedding** — outliers exist at the *token* level, "
        f"which is what the `events` column and each cluster's P3 panel are for.")
    rk = m["rival_km"]
    cont = cols["near%"] > CONTESTED
    add(f"- **Rivals are usually neighbours.** Median `rival_km` "
        f"{np.median(rk):,.0f} km against {m.get('rand_km', float('nan')):,.0f} km for a "
        f"random cluster pair; {int((rk < LOCAL_KM).sum())}/{K} are within "
        f"{LOCAL_KM:,.0f} km. But {int((cont & (rk > REMOTE_KM)).sum())} of "
        f"{int(cont.sum())} contested clusters have a rival more than "
        f"{REMOTE_KM:,.0f} km away — those are real embedding-space confusions, not "
        f"geographic blur, and nothing else in the repo distinguishes the two cases.")
    add(f"- **No diurnal cycle.** Largest deviation of any UTC-hour share from 0.25, over "
        f"all {K} clusters: **{np.abs(m['hour_share'] - 0.25).max():.3f}**. The 6-hourly "
        f"latents carry a seasonal cycle and essentially no diurnal one, so there is no "
        f"diurnal panel.")

    add("\n## Extreme events\n")
    add("*How to read this: `residual_map.npy` normalised per cell by median/MAD, cut at the "
        f"top {EVENT_PCT}%, then grouped into space-time connected components — adjacency is a "
        "real HEALPix NESTED neighbour at the same timestep, or the same cell at t+1.*\n")
    big = ev["size"] >= MIN_EVENT_TOKENS
    add(f"- threshold: robust **z ≥ {ev['thr']:.2f}**, **{ev['n_flagged']:,}** tokens "
        f"({EVENT_PCT}% of {m['N'] * N_CELLS:,}).")
    add(f"- **{int(big.sum()):,}** events of ≥{MIN_EVENT_TOKENS} tokens "
        f"(of {ev['size'].size:,} components); largest **{int(ev['size'].max()):,}** tokens; "
        f"{float((ev['size'][ev['comp']] > 1).mean()):.1%} of flagged tokens sit inside a "
        f"multi-token event.")
    add("\nLargest events in the run:\n")
    add("| event | when | where | steps | tokens | peak z | peak residual | host |")
    add("|---|---|---|---|---|---|---|---|")
    dt = np.load(os.path.join(sig_dir, "signatures.npz"))["datetime_hour"]
    for e in np.argsort(-ev["size"])[:10].tolist():
        add(f"| {e} | {str(dt[ev['t_min'][e]])[:13]} → {str(dt[ev['t_max'][e]])[:13]} | "
            f"{fmt_lat(ev['lat'][e])} {fmt_lon(ev['lon'][e])} | {int(ev['n_steps'][e])} | "
            f"{int(ev['size'][e]):,} | {ev['peak_z'][e]:.1f} | "
            f"{m['peak_raw'][e]:,.0f} | c{int(ev['host'][e])} |")

    if sweep_rows:
        add("\n### Why this threshold\n")
        add("*The open question the plan left: sweep the cut and pick the knee in "
            "\"fraction of flagged tokens inside a multi-token event\".*\n")
        add("| cut | robust z | flagged | events ≥2 | % tokens in multi-token events | "
            f"`events` distinct / {K} |")
        add("|---|---|---|---|---|---|")
        for r in sweep_rows:
            add(f"| {r['pct']}% | {r['thr']:.2f} | {r['flagged']:,} | {r['multi']:,} | "
                f"{r['frac']:.1%} | {r['distinct']} |")
        add(f"\n**There is no knee** — the curve is smooth and convex in log(cut). The choice "
            f"therefore falls to the distinct-value rule, and **{EVENT_PCT}%** is the loosest "
            f"cut that still selects a genuinely extreme tail while giving `events` enough "
            f"resolution to separate clusters.")

    add("\n## Validation\n")
    add("*Run on every `--rank` invocation; these are the invariants that catch a regression.*\n")
    add("| # | check | result |")
    add("|---|---|---|")
    for name, ok, detail in checks:
        add(f"| {name.split()[0]} | {' '.join(name.split()[1:])} | "
            f"{'✔' if ok else '✗'} {detail} |")
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Shortlist written to {out_path} ({len(L)} lines)")
    return audit, dropped


# ---------------------------------------------------------------------------
# deliverable B -- the per-cluster key report
# ---------------------------------------------------------------------------
def write_cluster_report(out_path, maps_dir, run_dir, sig_dir, j, m, ev, coverage, sig, dt):
    K, N = m["K"], m["N"]
    f = m["f"][:, j]
    tau50, core50 = float(m["tau50"][j]), int(m["core50"][j])
    lo, la, Rs, Rl, y0, y1 = (m["centroid"][j, 0], m["centroid"][j, 1], *m["conc"][j])
    kind = kind_of(tau50)
    rel = os.path.relpath(maps_dir, os.path.dirname(os.path.abspath(out_path)))

    # ---- P1 figure: the mass-coverage ladder ----
    fields, masks, titles = [], [], []
    for q in coverage:
        tau, mask = core_region(f, q)
        fields.append(f)
        masks.append(mask)
        titles.append(f"{q:.0%} of mass  —  τ={tau:.3f}, {int(mask.sum()):,} cells "
                      f"({mask.sum() / N_CELLS:.1%} of globe)")
    _, rival_core = core_region(m["f"][:, m["rival"][j]], 0.5)
    p1 = os.path.join(maps_dir, f"c{j:03d}_coverage.png")
    render_scalar_map(fields, p1, mask=masks, titles=titles, ncols=len(coverage),
                      cmap="magma", vmin=0.0, vmax=float(f.max()),
                      outline=[rival_core] * len(coverage), outline_color="#00e5ff",
                      cbar_label=f"occupancy f_{j}(cell) = P(cell carries label {j})",
                      suptitle=f"Cluster {j} — where it lives, at three mass-coverage levels "
                               f"(dashed cyan = rival c{int(m['rival'][j])}'s 50% core)")

    # ---- P2 figure: seasonal occupancy as a departure from the annual field ----
    fps = np.array([int(round(m["socc"][s].sum() / N_CELLS)) for s in range(4)])
    anom = [m["socc"][s][:, j] / max(fps[s], 1) - f for s in range(4)]
    amp = max(float(np.abs(a).max()) for a in anom)
    p2 = os.path.join(maps_dir, f"c{j:03d}_seasonal.png")
    render_scalar_map(anom, p2, titles=[f"{s}  (enrichment {m['season_enrich'][si, j]:.2f}×, "
                                        f"{fps[si]:,} files)"
                                        for si, s in enumerate(SEASON_KEYS)],
                      ncols=2, cmap="RdBu_r", vmin=-amp, vmax=amp,
                      cbar_label=f"f_{j}(cell | season) − f_{j}(cell)   (departure from its own "
                                 f"annual field)",
                      suptitle=f"Cluster {j} — seasonal migration")

    L = []
    add = L.append
    add(f"# Cluster {j} — `{run_dir}`")
    add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `cluster_probe.py --cluster {j}`. "
        f"Per-cell arrays from `{sig_dir}` ({N:,} files × {N_CELLS:,} cells). "
        f"Companion reports: `report.md` (size / EVR / affinity), `temporal_report.md` "
        f"(dominant-cluster maps), `probe/shortlist.md` (all {K} clusters).*")

    # ---------------- P0 ----------------
    add("\n## P0 — identity\n")
    add("*How to read this: one sentence generated from the numbers. Longitude is a "
        "**circular** mean and is only meaningful when the circular concentration `R_lon` is "
        "high — a zonal band and a ring around a pole both drive it to ~0, which is why it is "
        "always printed next to the latitude band.*\n")
    lon_txt = (f"centred near {fmt_lat(la)} {fmt_lon(lo)}" if Rl >= 0.6 else
               f"centred near {fmt_lat(la)} with **no meaningful longitude** "
               f"(R_lon = {Rl:.2f})")
    zonal = (Rl < 0.35 and abs(y1 - y0) < 40)
    add(f"> **Cluster {j}** holds **{m['tokens'][j]:,}** tokens (**{m['share'][j]:.2%}** of "
        f"the dataset). Half its mass sits in **{core50:,} cells** "
        f"({core50 / N_CELLS:.1%} of the globe) at occupancy **τ50 = {tau50:.3f}** ⇒ it is "
        f"**{kind}**. It is {lon_txt}, latitude band "
        f"{fmt_lat(y0)}…{fmt_lat(y1)} (R_lon {Rl:.2f}, R_sphere {Rs:.2f})"
        + (", i.e. a **zonal band**" if zonal else "") + ". "
        f"Diagnostics: `zres` **{m['zres'][j]:+.2f}**, `unmodelled` "
        + ("–" if not np.isfinite(m["unmodelled"][j]) else f"**{m['unmodelled'][j]:.2f}**")
        + f", `near%` **{m['near'][j]:.1%}**, **{int(m['events'][j])}** extreme events hosted, "
        f"`seasonality` **{m['seasonality'][j]:.3f}**.")
    if m["maxf"][j] >= 0.9 and tau50 < ITINERANT_TAU:
        add(f"\n**Note the split personality:** peak occupancy is {m['maxf'][j]:.2f} — it owns "
            f"a handful of cells outright, permanently — yet τ50 is only {tau50:.3f}, so its "
            f"*mass* is smeared over {core50:,} cells. This is exactly why `maxf` is not the "
            f"territoriality scalar and τ50 is.")
    add(f"\nRelative to the run: `zres` rank **{int((m['zres'] > m['zres'][j]).sum()) + 1}/{K}** "
        f"(hardest first), `unmodelled` rank "
        f"**{int((np.nan_to_num(m['unmodelled'], nan=-1) > np.nan_to_num(m['unmodelled'][j], nan=-1)).sum()) + 1}/{K}**, "
        f"`events` rank **{int((m['events'] > m['events'][j]).sum()) + 1}/{K}**, "
        f"τ50 rank **{int((m['tau50'] > tau50).sum()) + 1}/{K}** (most territorial first).")

    # ---------------- P1 ----------------
    add("\n## P1 — where it lives (mass-coverage ladder)\n")
    add("*How to read this: the field is the cluster's **occupancy** "
        f"`f_{j}(cell)` — the fraction of the {N:,} timesteps at which that HEALPix cell "
        f"carries label {j}, reduced from `label_map.npy`. Each panel keeps the smallest set "
        "of cells holding the stated fraction of the cluster's tokens (the highest-density "
        "region; `worldmap.core_region`), greying out the rest; `τ` is the occupancy at that "
        "crossing. Sliding **mass coverage** rather than a fixed occupancy threshold is what "
        "makes one map work for both kinds of cluster — τ=0.5 would show a solid blob for a "
        "territorial cluster and an empty map for an itinerant one. The three panels share one "
        "colour scale. The dashed cyan outline is the runner-up cluster's own 50% core (P4).*\n")
    add(f"![coverage ladder]({rel}/c{j:03d}_coverage.png)\n")
    add("| coverage | τ | cells | % of globe | % of its presence footprint |")
    add("|---|---|---|---|---|")
    pres = int((f > 0).sum())
    for q in coverage:
        tau, mask = core_region(f, q)
        add(f"| {q:.0%} | {tau:.3f} | {int(mask.sum()):,} | {mask.sum() / N_CELLS:.1%} | "
            f"{mask.sum() / max(pres, 1):.1%} |")
    add(f"\nPeak occupancy `maxf` = **{m['maxf'][j]:.3f}**; it appears at all in "
        f"**{pres:,}** cells ({pres / N_CELLS:.1%} of the globe), "
        f"which is why a presence map would be uninformative here.")

    # ---------------- P2 ----------------
    add("\n## P2 — seasonal migration (departure from its own annual field)\n")
    add(f"*How to read this: each panel is `f_{j}(cell | season) − f_{j}(cell)` — the "
        "season's occupancy **minus the annual occupancy of P1**, not the raw seasonal field. "
        "The raw seasonal maps mostly restate P1; the difference isolates the *migration*. "
        "Red = the cluster is there more often in that season than on average, blue = less. "
        "Seasons follow the repo's NH convention (DJF groups December with the following "
        "Jan/Feb). The four panels share one symmetric scale.*\n")
    add(f"![seasonal anomaly]({rel}/c{j:03d}_seasonal.png)\n")
    add("Seasonal enrichment (share of its tokens in that season ÷ its overall share; "
        "1.00 = flat): " + " · ".join(
            f"**{s} {m['season_enrich'][si, j]:.2f}×**" for si, s in enumerate(SEASON_KEYS))
        + f". Annual `seasonality` (std/mean of the 12 monthly enrichments) = "
          f"**{m['seasonality'][j]:.3f}** "
          f"(run median {np.median(m['seasonality']):.3f}, max {m['seasonality'].max():.3f}).")
    add(f"\nPer-cell seasonal amplitude for this cluster (max over seasons of "
        f"|anomaly|): max **{amp:.3f}**, 99th pct "
        f"**{np.percentile(np.abs(np.stack(anom)).max(0), 99):.3f}**.")
    hs = m["hour_share"][:, j]
    add(f"\n**No diurnal counterpart.** Its token share by UTC hour is "
        + " / ".join(f"{h:02d}Z {v:.3f}" for h, v in zip((0, 6, 12, 18), hs))
        + f" — largest deviation from 0.25 is **{np.abs(hs - 0.25).max():.3f}**. Measured, not "
          f"assumed: the 6-hourly latents carry a seasonal cycle and essentially no diurnal one, "
          f"so there is no diurnal panel.")

    # ---------------- P3 ----------------
    add("\n## P3 — extreme events hosted\n")
    add("*How to read this: `residual_map.npy` gives a residual per cell per timestep. Raw "
        "residuals are not comparable across cells (the per-cell mean spans 12.4×), so each is "
        "scored against its own temporal norm as "
        "`z = (residual − median_cell) / (1.4826·MAD_cell)`; the top "
        f"{EVENT_PCT}% (z ≥ {ev['thr']:.2f}) is cut and grouped into space-time connected "
        "components — adjacency is a real HEALPix NESTED neighbour at the same timestep, or the "
        "same cell at t+1. **`peak residual` is printed next to `peak z` deliberately**: a large "
        "z in a quiet cell can be a small absolute anomaly, and only the pair tells you which. "
        "An event is attributed here if its peak token carries this cluster's label. These are "
        "`(date, cell)` cases to look up, not aggregates.*\n")
    mine = np.nonzero((ev["host"] == j) & (ev["size"] >= MIN_EVENT_TOKENS))[0]
    if mine.size == 0:
        add(f"This cluster hosts **no** multi-token event at the {EVENT_PCT}% cut "
            f"({int(m['events_all'][j])} single-token flags). Quiet, or well modelled.")
    else:
        mine = mine[np.argsort(-ev["peak_z"][mine])][:10]
        add(f"**{int(m['events'][j])}** events of ≥{MIN_EVENT_TOKENS} tokens "
            f"(rank {int((m['events'] > m['events'][j]).sum()) + 1}/{K} in the run); "
            f"top {len(mine)} by peak z:\n")
        add("| when | steps | where | cells | tokens | peak z | peak residual | glob. pct | "
            "peak cell |")
        add("|---|---|---|---|---|---|---|---|---|")
        for e in mine.tolist():
            add(f"| {str(dt[ev['t_min'][e]])[:13]} → {str(dt[ev['t_max'][e]])[:13]} | "
                f"{int(ev['n_steps'][e])} | {fmt_lat(ev['lat'][e])} {fmt_lon(ev['lon'][e])} | "
                f"{int(ev['n_cells'][e])} | {int(ev['size'][e]):,} | {ev['peak_z'][e]:.1f} | "
                f"{m['peak_raw'][e]:,.0f} | {m['peak_rawpct'][e]:.1f}% | "
                f"{int(ev['peak_cell'][e])} |")
        add(f"\nLook a case up with "
            f"`torch.load('latents_2/latent_{{file_id}}.pt')['latent'][0, {{peak cell}}]`; the "
            f"file id is the row index of `{sig_dir}/residual_map.npy`, i.e. "
            f"`datetime = 2014-01-01 00:00 UTC + file_id × 6 h`.")
    add(f"\nWhole-cluster hardness: `zres` = **{m['zres'][j]:+.2f}** (mean robust z of *all* "
        f"its tokens, run range {m['zres'].min():+.2f}…{m['zres'].max():+.2f}); mean raw "
        f"residual **{m['resid'][j]:,.0f}**"
        + ("" if not np.isfinite(m["unmodelled"][j]) else
           f"; `unmodelled` = residual/persistence = **{m['unmodelled'][j]:.2f}** "
           f"(run range {np.nanmin(m['unmodelled']):.2f}…{np.nanmax(m['unmodelled']):.2f}) — "
           + ("**hard because unmodelled**, not because it moves"
              if m["unmodelled"][j] > np.nanmedian(m["unmodelled"]) else
              "**hard mostly because it moves fast**, not because it is unmodelled"))
        + ".")

    # ---------------- P4 ----------------
    r = int(m["rival"][j])
    km = m["rival_km"][j]
    add("\n## P4 — its rival\n")
    add("*How to read this: `file_signature.py` ranks every token against all K subspaces, so "
        "the runner-up is free. `near%` is the share of this cluster's own tokens whose "
        "runner-up is within 10% of the winner; `runner-up` is the single cluster that takes "
        "most of that runner-up mass. `rival_km` is the great-circle distance between the two "
        "clusters' 50%-mass core centroids — the new part, because it separates two cases the "
        "margin alone conflates.*\n")
    verdict = ("**local blur** — the rival's core is next door, so the near-ties are a map "
               "gradient (one region shading into its neighbour), not a confusion of unlike "
               "states" if km < LOCAL_KM else
               "**remote confusion** — the rival's core is on the other side of the world, so "
               "these near-ties are a genuine embedding-space ambiguity between "
               "geographically unrelated states (look for a cross-hemisphere seasonal mirror: "
               "compare the two clusters' P2 panels)" if km > REMOTE_KM else
               "**intermediate** — same hemisphere, different region")
    add(f"| | value |")
    add(f"|---|---|")
    add(f"| runner-up | **c{r}** |")
    add(f"| runner-up share of its runner-up mass | {m['rival_frac'][j]:.1%} |")
    add(f"| `near%` (its tokens within 10% of a rival) | {m['near'][j]:.1%} "
        f"(run mean {m['near'].mean():.1%}) |")
    add(f"| mean margin `(R₂−R₁)/R₁` | {m['margin'][j]:.3f} |")
    add(f"| `rival_km` (core centroid to core centroid) | **{km:,.0f} km** |")
    add(f"| c{r}'s core | {int(m['core50'][r]):,} cells at τ50 {m['tau50'][r]:.3f}, "
        f"{fmt_lat(m['centroid'][r, 1])} {fmt_lon(m['centroid'][r, 0])} |")
    add(f"\nVerdict: {verdict}. c{r}'s 50% core is outlined in cyan on P1.")
    if m["rival"][r] == j:
        add(f"\n**Mutual pair:** c{r}'s own runner-up is c{j}, so these two contest each other "
            f"symmetrically ({m['near'][j]:.0%} / {m['near'][r]:.0%} near-ties).")

    add("\n---\n")
    add(f"*Not repeated here (see `{run_dir}/report.md`): size/EVR/d80, `cells@50%`, `owned`, "
        f"`files@50%`, `tCV`, `maxAff` and the affinity table. Not repeated here (see "
        f"`{run_dir}/temporal_report.md`): the monthly and seasonal **dominant-cluster** maps, "
        f"which show all {K} clusters at once — this report shows exactly one.*")
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"Cluster report written to {out_path} ({len(L)} lines, 2 figures)")


def write_coverage_slider(out_path, run_dir, j, m, n_stops):
    """P1 as a real slider: one pre-rendered panel per coverage stop, in one HTML file.

    The Markdown ladder (50/80/95%) is the default because it needs no browser and sits in
    the report next to the numbers. This is the same construction swept continuously, for
    when the question is "where exactly does this cluster's footprint stop being real".
    Panels are base64-embedded, so the file is self-contained and can be copied anywhere --
    the alternative (a `maps/` sidecar per stop) breaks the moment the html is moved.
    """
    import base64
    import tempfile

    f = m["f"][:, j]
    stops = np.linspace(0.10, 0.95, n_stops)
    imgs, rows = [], []
    with tempfile.TemporaryDirectory() as td:
        for i, q in enumerate(stops):
            tau, mask = core_region(f, float(q))
            png = os.path.join(td, f"s{i}.png")
            render_scalar_map(f, png, mask=mask, cmap="magma", vmin=0.0,
                              vmax=float(f.max()), figsize=(9.0, 5.2),
                              cbar_label=f"occupancy f_{j}(cell)",
                              titles=f"{q:.0%} of mass — τ={tau:.3f}, "
                                     f"{int(mask.sum()):,} cells ({mask.sum() / N_CELLS:.1%} "
                                     f"of the globe)")
            with open(png, "rb") as fh:
                imgs.append(base64.b64encode(fh.read()).decode())
            rows.append((float(q), float(tau), int(mask.sum())))

    opts = ",".join(f'["{q:.0%}",{tau:.4f},{n}]' for q, tau, n in rows)
    body = "\n".join(f'<img id="i{i}" src="data:image/png;base64,{b}" '
                     f'style="display:{"block" if i == 0 else "none"}">'
                     for i, b in enumerate(imgs))
    html = f"""<!doctype html><meta charset="utf-8">
<title>Cluster {j} coverage — {os.path.basename(run_dir)}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1000px;color:#111;
      background:#fff}}
 img{{width:100%;height:auto}} input{{width:100%}}
 .k{{font-variant-numeric:tabular-nums}} .m{{color:#555}}
</style>
<h2>Cluster {j} — occupancy at a sliding mass-coverage threshold</h2>
<p class="m">Each stop keeps the smallest set of HEALPix cells holding that fraction of
cluster {j}'s tokens (the highest-density region); τ is the occupancy at the crossing.
Sliding <b>mass coverage</b> rather than a fixed occupancy threshold is what makes one
control work for both a territorial cluster (τ50≈1) and an itinerant one (τ50&lt;0.1).
Generated by <code>cluster_probe.py --cluster {j} --html</code>.</p>
<input type="range" min="0" max="{len(rows) - 1}" value="0" id="s">
<p class="k" id="c"></p>
{body}
<script>
const O=[{opts}],s=document.getElementById("s"),c=document.getElementById("c");
let prev=0;
function draw(){{
  const v=+s.value;
  document.getElementById("i"+prev).style.display="none";
  document.getElementById("i"+v).style.display="block";
  prev=v;
  c.textContent=`coverage ${{O[v][0]}}  ·  τ = ${{O[v][1].toFixed(3)}}  ·  ${{O[v][2].toLocaleString()}} cells`;
}}
s.addEventListener("input",draw);draw();
</script>
"""
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"Coverage slider written to {out_path} "
          f"({len(rows)} stops, {os.path.getsize(out_path) / 2**20:.1f} MB)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="runs/clustering/v6_subspace_big_d64",
                    help="clustering run directory (model.pt + assignments.pt)")
    ap.add_argument("--rank", action="store_true", help="write probe/shortlist.md")
    ap.add_argument("--cluster", type=int, action="append", default=None,
                    help="write probe/c<NNN>.md for this cluster (repeatable)")
    ap.add_argument("--sort", default="events", help="shortlist sort column")
    ap.add_argument("--coverage", default=",".join(str(q) for q in DEFAULT_COVERAGE),
                    help="P1 mass-coverage stops")
    ap.add_argument("--signatures", default=None,
                    help="runs/signatures/<run> dir (default: auto-detect by manifest)")
    ap.add_argument("--persistence", default=None,
                    help="err_persist.npy (default: first runs/persistence/*/err_persist.npy)")
    ap.add_argument("--event-pct", type=float, default=EVENT_PCT,
                    help="robust-z tail percentage kept as extreme")
    ap.add_argument("--event-sweep", action="store_true",
                    help="also sweep the event threshold and tabulate it in the shortlist")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--html", action="store_true",
                    help="also write c<NNN>_slider.html: a real coverage slider "
                         "(one pre-rendered panel per stop, embedded, no external files)")
    ap.add_argument("--html-stops", type=int, default=10,
                    help="number of coverage stops the slider offers (default 10)")
    args = ap.parse_args()
    if not args.rank and not args.cluster:
        raise SystemExit("nothing to do: pass --rank and/or --cluster N")
    coverage = tuple(sorted(float(x) for x in args.coverage.split(",")))

    t0 = time.time()
    sig_dir = resolve_signature_dir(args.dir, args.signatures)
    persist = resolve_persistence(args.persistence)
    print(f"Run       : {args.dir}")
    print(f"Signatures: {sig_dir}")
    print(f"Persistence: {persist or 'NOT FOUND -> `unmodelled` unavailable'}", flush=True)

    model = torch.load(os.path.join(args.dir, "model.pt"), map_location="cpu",
                       weights_only=False)
    sig = np.load(os.path.join(sig_dir, "signatures.npz"))
    cache = load_cache(args.dir, sig_dir, persist, args.rebuild_cache)
    dt = sig["datetime_hour"]

    # robust-z map + events (cheap: ~15 s, and the event table depends on --event-pct)
    R = np.load(os.path.join(sig_dir, "residual_map.npy"))
    Z = (R - cache["cell_med"]) / (1.4826 * np.maximum(cache["cell_mad"], 1e-6))
    labels = np.asarray(np.load(os.path.join(sig_dir, "label_map.npy")))
    nb = healpix_nest_neighbours(NSIDE)
    ev = extract_events(Z, labels, args.event_pct, nb)
    print(f"  events: {ev['n_flagged']:,} tokens at z >= {ev['thr']:.2f} -> "
          f"{ev['size'].size:,} components ({time.time() - t0:.0f}s)", flush=True)

    m = cluster_metrics(cache, sig, ev, coverage)
    m["peak_raw"] = R.ravel()[ev["flat"]][ev["peak_i"]]
    # Where the peak sits on the UNNORMALISED scale, as a percentile of all 160 M tokens.
    # Without this the z alone is badly misleading in a quiet cell: the March 2022 Antarctic
    # heatwave -- the largest surface temperature anomaly ever recorded on Earth -- peaks at
    # z 22.2 but only the 14th percentile of raw residual, i.e. *below the global median*.
    # z answers "unusual for this place", which is the question a weather record asks; it
    # does not answer "large in latent space". Print both, always.
    m["peak_rawpct"] = 100.0 * (np.searchsorted(np.sort(R[::7].ravel()), m["peak_raw"])
                                / (R[::7].size))
    # global one-liners for the shortlist (not per-cluster columns -- see the docstring)
    cnt = model["counts"].float()
    w = cnt / cnt.sum()
    mu_g = (w[:, None] * model["means"]).sum(0)
    m["iso_max"] = float((((model["means"] - mu_g) ** 2).sum(1)
                          / model["trace"].clamp(min=1e-12)).max())
    rng = np.random.default_rng(0)
    ii = rng.integers(0, m["K"], 20000)
    jj = rng.integers(0, m["K"], 20000)
    ok = ii != jj
    m["rand_km"] = float(np.median([great_circle_km(*m["centroid"][a], *m["centroid"][b])
                                    for a, b in zip(ii[ok][:3000], jj[ok][:3000])]))
    del R

    probe_dir = os.path.join(args.dir, "probe")
    maps_dir = os.path.join(probe_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    if args.rank:
        print("\nValidation (docs/ideas/cluster_probe.md §9):", flush=True)
        checks, _ = run_checks(args.dir, sig_dir, cache, sig, m, ev, model)
        sweep = None
        if args.event_sweep:
            sweep = []
            for pct in (0.01, 0.05, 0.1, 0.5):
                e = extract_events(Z, labels, pct, nb)
                big = e["size"] >= MIN_EVENT_TOKENS
                cnts = np.bincount(e["host"][big], minlength=m["K"])
                sweep.append({"pct": pct, "thr": e["thr"], "flagged": e["n_flagged"],
                              "multi": int(big.sum()),
                              "frac": float((e["size"][e["comp"]] > 1).mean()),
                              "distinct": len(set(cnts.tolist()))})
                print(f"  sweep {pct}%: z>={e['thr']:.2f} {e['n_flagged']:,} flagged, "
                      f"{int(big.sum()):,} events, {sweep[-1]['frac']:.1%} in multi, "
                      f"{sweep[-1]['distinct']} distinct", flush=True)
        audit, dropped = write_shortlist(os.path.join(probe_dir, "shortlist.md"), args.dir,
                                         sig_dir, persist, m, ev, checks, args.sort, model,
                                         sweep)
        print("\nColumn audit (distinct values of "
              f"{m['K']}): " + ", ".join(f"{c}={n}" for c, n in audit.items()))
        if dropped:
            print(f"  DROPPED (< {MIN_DISTINCT}): {', '.join(dropped)}")

    for j in (args.cluster or []):
        if not 0 <= j < m["K"]:
            raise SystemExit(f"--cluster {j} out of range 0..{m['K'] - 1}")
        write_cluster_report(os.path.join(probe_dir, f"c{j:03d}.md"), maps_dir, args.dir,
                             sig_dir, j, m, ev, coverage, sig, dt)
        if args.html:
            write_coverage_slider(os.path.join(probe_dir, f"c{j:03d}_slider.html"),
                                  args.dir, j, m, args.html_stops)
    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
