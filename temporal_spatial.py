#!/usr/bin/env python3
"""Temporal & spatial report generator.

Reads a run's frozen `model.pt` + `assignments.pt` and writes a dedicated report
(`temporal_report.md`, default) with the spatial and temporal structure at calendar
(monthly) resolution:

  - the ERA5 **time axis** (2014-01-01 00:00, 6-hourly, 4/day; file idx -> datetime),
  - an **annual** dominant-cluster world map (continent outlines + smooth heatmap),
  - **12 monthly** dominant-cluster maps (one per calendar month),
  - a **monthly enrichment** matrix flagging the most seasonal clusters, and
  - a **month-to-month stability** curve (% of cells whose dominant cluster flips).

The maps reuse the run's *existing* assignments: v6 already covers all 12,288 cells in
every calendar month (~540-620 files/month), so there is **no recomputation** -- this is a
rendering/reporting pass (seconds). For a run that under-samples some month, re-cluster
with more `--num-files`; the same script then just reads the denser assignments.

Time axis: the latents store only an integer `idx`, never a timestamp, so the calendar
date is reconstructed positionally as `datetime = 2014-01-01 00:00 UTC + idx × 6h`. Under
this, file 0 = 2014-01-01 00:00 and the last file (13020) = 2022-11-30 00:00 (a full
9-year 2014-2022 ERA5 range would extend to 2022-12-31 18:00, ~127 more steps -- the last
~month of 2022 is not fully covered).

Rendering (continent outlines + smooth heatmap) and HEALPix geometry live in `worldmap.py`;
colors come from `affinity_ordered_colors` (spectral seriation of the subspace-affinity
matrix) and are computed **once** and shared across every map so months are comparable.

Usage:
  python3 temporal_spatial.py --dir subspace_kmeans_runs/v6_subspace_big_d64
"""

import argparse
import calendar
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch

from worldmap import (N_CELLS, build_affinity_matrix, affinity_ordered_colors,
                      render_world_map)

START = datetime(2014, 1, 1, 0, 0)          # ERA5 first step (reconstructed)
STEP = timedelta(hours=6)                   # 6-hourly cadence (4/day)
N_FILES_TOTAL = 13021


def file_index_to_month():
    """Month (1..12) of every file index under START + idx*STEP. Size [N_FILES_TOTAL]."""
    # numpy datetime: add STEP as a timedelta64; one vectorized op, no per-row Python.
    base = np.datetime64(START).astype("datetime64[ms]")
    delta = np.timedelta64(int(STEP.total_seconds() * 1000), "ms")
    idx = np.arange(N_FILES_TOTAL)
    return (base + idx * delta).astype("datetime64[M]").astype(int) % 12 + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="subspace_kmeans_runs/v6_subspace_big_d64",
                    help="directory with model.pt + assignments.pt")
    ap.add_argument("--out", default=None, help="output Markdown (default: <dir>/temporal_report.md)")
    ap.add_argument("--top-seasonal", type=int, default=8,
                    help="most seasonal clusters to profile (12-month enrichment)")
    ap.add_argument("--no-coastlines", action="store_true", help="skip continent outlines")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="use the legacy scatter pixels instead of the smooth heatmap")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.dir, "temporal_report.md")
    coast = not args.no_coastlines
    heat = not args.no_heatmap

    m = torch.load(os.path.join(args.dir, "model.pt"), map_location="cpu", weights_only=False)
    a = torch.load(os.path.join(args.dir, "assignments.pt"), map_location="cpu", weights_only=False)
    U, means = m["U"], m["means"]
    K, D, d = U.shape
    lab = a["label"].long()
    cell = a["cell_id"].long()
    fid = a["file_id"].long()
    cnt = torch.bincount(lab, minlength=K).float()              # from assignments (not model field)
    T = lab.numel()
    sampled_files = sorted(m.get("sampled_files", []))

    mon_of_file = file_index_to_month()                       # [N_FILES_TOTAL] -> month
    mon = torch.from_numpy(mon_of_file)[fid]                  # [T] month per token

    # ---- per (month, cell, cluster) counts in ONE bincount ------------------
    key = ((mon - 1).long() * (N_CELLS * K) + cell * K + lab).to(torch.int64)
    mcc = torch.bincount(key, minlength=12 * N_CELLS * K).view(12, N_CELLS, K).float()
    dominant_m = mcc.argmax(2)                                 # [12, N_CELLS]
    valid_m = mcc.sum(2) > 0                                   # [12, N_CELLS]
    annual_cc = mcc.sum(0)                                     # [N_CELLS, K]
    dominant_annual = annual_cc.argmax(1)
    valid_annual = annual_cc.sum(1) > 0

    monthK = mcc.sum(1)                                        # [12, K]
    month_tot = mcc.sum((1, 2))                               # [12]
    files_per_month = np.bincount(mon_of_file[np.asarray(sampled_files)],
                                  minlength=13)[1:]            # [12]

    # shared colors (one spectral order for every map -> months comparable)
    aff = build_affinity_matrix(U, means)
    colors = affinity_ordered_colors(aff, K)

    maps_dir = os.path.join(args.dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    # report image refs are relative to the report file, so --out outside --dir still resolves
    maps_rel = os.path.relpath(maps_dir, os.path.dirname(os.path.abspath(out_path)))

    print(f"Rendering maps (heatmap={heat}, coastlines={coast})...", flush=True)
    t0 = time.time()
    render_world_map(dominant_annual, valid_annual, colors, os.path.join(maps_dir, "map_annual.png"),
                     title="Dominant cluster per cell — annual", coastlines=coast, heatmap=heat)
    for mm in range(12):
        render_world_map(dominant_m[mm], valid_m[mm], colors,
                         os.path.join(maps_dir, f"map_month_{mm + 1:02d}.png"),
                         title=f"{calendar.month_name[mm + 1]} — dominant cluster per cell "
                               f"({files_per_month[mm]} files, {int(month_tot[mm]):,} tokens)",
                         coastlines=coast, heatmap=heat)
    print(f"  rendered 13 maps in {time.time() - t0:.1f}s", flush=True)

    # ---- monthly enrichment (seasonality) -----------------------------------
    # nan marks a calendar month with no sampled files (or a cluster absent there);
    # seasonality is computed nan-aware so an empty month can't fabricate signal.
    denom = month_tot.unsqueeze(1) / T                          # [12,1]; 0 for empty months
    raw = (monthK / cnt.clamp(min=1).unsqueeze(0)) / denom.clamp(min=1e-12)
    enrich = torch.where(month_tot.unsqueeze(1) > 0, raw,
                         torch.full_like(raw, float("nan")))    # [12,K], 1.0=flat, nan=no data
    enr_np = enrich.numpy()
    seas = torch.tensor(np.nanstd(enr_np, axis=0) /
                        np.nanmean(enr_np, axis=0).clip(1e-12))  # [K] seasonality score

    # ---- month-to-month dominant-flip rate (temporal stability) -------------
    flips = []
    for mm in range(11):
        both = valid_m[mm] & valid_m[mm + 1]
        flips.append(float((dominant_m[mm][both] != dominant_m[mm + 1][both]).float().mean())
                     if both.any() else float("nan"))

    # ---- Jan vs Jul shift ---------------------------------------------------
    both_17 = valid_m[0] & valid_m[6]
    jan_jul_changed = (float((dominant_m[0][both_17] != dominant_m[6][both_17]).float().mean())
                       if both_17.any() else float("nan"))
    owned = lambda dom, v: torch.bincount(dom[v].long(), minlength=K)   # cells per cluster
    delta_jul_jan = (owned(dominant_m[6], valid_m[6]) - owned(dominant_m[0], valid_m[0])).float()

    # ============================= report ====================================
    L = []
    add = L.append
    def cell(v, spec=".2f"):                      # nan (empty month / unseen cluster) -> en-dash
        v = float(v)
        return "–" if v != v else format(v, spec)
    add(f"# Temporal & spatial report — `{args.dir}`")
    add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `temporal_spatial.py`. "
        f"K={K} {'affine subspaces (d=' + str(d) + ')' if d > 0 else 'point clusters'}, "
        f"{T:,} tokens from {len(sampled_files)} files. "
        f"Maps reuse this run's existing assignments (no recomputation).*")

    add("\n## Time axis\n")
    add("The latents are ERA5 atmospheric states at **6-hourly cadence** (00/06/12/18 UTC — "
        "4 per day). Each file stores only an integer sample index (`idx`), **not a "
        "timestamp**, so the calendar date is reconstructed positionally:\n")
    add("> `datetime = 2014-01-01 00:00 UTC + idx × 6 h`\n")
    add(f"The {N_FILES_TOTAL:,} files therefore span **file 0 = 2014-01-01 00:00** through "
        f"**file 13020 = 2022-11-30 00:00**. (A complete 9-year 2014→2022 ERA5 record would "
        f"run to 2022-12-31 18:00 — about 127 more steps — so the last ~month of 2022 is "
        f"treated as not fully covered.) Monthly grouping below uses this reconstructed date.")
    add(f"\nSampled files per calendar month: "
        + ", ".join(f"{calendar.month_abbr[mm + 1]} {int(files_per_month[mm])}"
                    for mm in range(12)) + ".")

    add("\n## Annual map\n")
    add(f"![Annual dominant cluster]({maps_rel}/map_annual.png)\n")
    add("Dominant cluster per HEALPix cell across the full sample (NESTED ordering; grey = no "
        "data). Continent outlines are Natural Earth 110m; the field is the per-cell RGB "
        "(from spectral-ordered cluster colors) interpolated to a smooth heatmap. "
        "**Subspace-similar clusters share hues**, so coherent regions read as gradients.")

    add("\n## How the colors work\n")
    add("Cluster colors are **not random** — they are assigned so similar clusters get similar "
        "colors, which is what makes the maps readable (and the heatmap interpolation "
        "legitimate). The pipeline (`worldmap.py`):\n")
    add("1. **Affinity matrix (K×K)** — for each cluster pair, how similar they are. With "
        "subspaces (d>0) it is the mean squared cosine of the principal angles between the two "
        "bases, `‖UᵢᵀUⱼ‖²_F / d ∈ [0,1]` (1 = same subspace, 0 = orthogonal); for point clusters "
        "(d=0) the cosine of the two centroids. High affinity ⇒ nearby / overlapping structure.")
    add("2. **Spectral seriation** — take the *Fiedler vector* (the 2nd-smallest eigenvector of "
        "the normalized graph Laplacian of that affinity matrix): the classic 1-D embedding "
        "that places similar items next to each other. Ranking clusters by their Fiedler "
        "coordinate gives a single similarity-ordered sequence.")
    add("3. **Colormap** — that rank (0…K−1) is mapped through the smooth perceptual `turbo` "
        "colormap, so the order of colors along the rainbow exactly follows the similarity "
        "order: neighboring hues = affinity-similar clusters.")
    add("4. **Heatmap** — each cell takes its dominant cluster's RGB, and it is the **RGB** "
        "(not the integer cluster id) that is interpolated across the map. Interpolating a "
        "categorical id would be meaningless, but because step 3 already gave similar clusters "
        "similar RGB, blending two neighbors yields a sensible in-between color.")
    add("\nReading the maps:")
    add("- Genuine structure shows up as **smooth gradients** (a region slowly shading into a "
        "neighboring hue); only true salt-and-pepper noise stays speckled. A *random* hue "
        "assignment would put a sharp color jump at every boundary and alias fine sub-regions "
        "as spurious \"scatter,\" especially at large K.")
    add("- The **absolute hue is arbitrary** — blue vs red just reflects a cluster's position "
        "in the Fiedler order, which has no physical meaning; only the *transitions* and "
        "*groupings* carry information. So two months can look similarly hued where the same "
        "cluster family dominates, even if the exact dominant cluster differs.")

    add("\n## Monthly maps\n")
    add("One dominant-cluster map per calendar month (same color scale as the annual map), "
        "revealing the seasonal cycle. Each cell is colored by its most frequent cluster among "
        "that month's tokens.\n")
    for mm in range(12):
        add(f"**{calendar.month_name[mm + 1]}** ({int(files_per_month[mm])} files, "
            f"{int(month_tot[mm]):,} tokens)\n")
        add(f"![{calendar.month_name[mm + 1]}]({maps_rel}/map_month_{mm + 1:02d}.png)\n")

    add("\n## Most seasonal clusters\n")
    add("Enrichment of each cluster per month = `(cluster share in month) / (its average share)`. "
        "**1.0 = present year-round**; values ≫ 1 mark the months where the cluster concentrates "
        "(a seasonal signature). Sorted by seasonality (std/mean across the 12 months).\n")
    add("| cluster | seasonality | " + " | ".join(calendar.month_abbr[mm + 1] for mm in range(12)) + " |")
    add("|---|---|" + "|".join(["---"] * 12) + "|")
    for j in seas.argsort(descending=True)[:args.top_seasonal].tolist():
        row = " | ".join(cell(enrich[mm, j]) for mm in range(12))
        add(f"| {j} | {cell(seas[j])} | {row} |")

    add("\n## Month-to-month stability\n")
    fa = np.array(flips, dtype=float)
    fa_valid = fa[~np.isnan(fa)]
    add("Share of cells whose **dominant cluster changes** between consecutive months "
        "(low = stable geography; peaks mark the seasonal transitions)."
        + (f" Over the {len(fa_valid)} comparable month-pair(s): "
           f"min **{np.nanmin(fa):.1%}**, mean **{np.nanmean(fa):.1%}**, "
           f"max **{np.nanmax(fa):.1%}**.\n" if len(fa_valid)
           else " (no consecutive months both have data.)\n"))
    add("| transition | cells changing dominant cluster |")
    add("|---|---|")
    for mm in range(11):
        add(f"| {calendar.month_abbr[mm + 1]}→{calendar.month_abbr[mm + 2]} | {cell(flips[mm], '.1%')} |")
    add(f"\n**Jan ↔ Jul** (winter vs summer hemisphere): **{cell(jan_jul_changed, '.1%')}** of cells change "
        "dominant cluster. Largest cluster shifts (owned-cell count, Jul − Jan): "
        + ", ".join(f"{int(j)}: {delta_jul_jan[j]:+.0f}"
                    for j in delta_jul_jan.argsort(descending=True)[:5].tolist())
        + " ….")

    add("\n## Interpretation notes\n")
    add("- **Stable geography + month-to-month flips near the minimum** ⇒ clusters are "
        "**geographic regimes** (region/surface type) that hold their territory year-round.")
    add("- **Clusters with high seasonality and a single summer/winter peak** ⇒ **seasonal "
        "regimes** (e.g. monsoon, sea-ice, snow cover); find them in the table above.")
    add("- **Jan↔Jul changes concentrate in one hemisphere** ⇒ a hemispheric seasonal cycle "
        "(opposite phases north/south).")
    add("- Monthly maps share one color scale, so a hue *appearing* in a region month-to-month "
        "is a real shift, not a recoloring.")
    add(f"\n*See the main clustering report (`{args.dir}/report.md`) for convergence, variance "
        f"decomposition, per-cluster spatial/temporal columns, and subspace affinity.*")

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print(f"Temporal report written to {out_path} ({len(L)} lines)")


if __name__ == "__main__":
    main()
