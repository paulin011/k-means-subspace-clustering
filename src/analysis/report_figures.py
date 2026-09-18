#!/usr/bin/env python3
"""Appendix figures for `report/report.tex` (page 3).

Two deliverables, both written into `--out` (default `report/`) so `pdflatex`
finds them next to `report.tex`:

1. **`fig_territories.png`** — the "clusters have territories" figure. Three
   Mollweide panels, one per cluster, painting the **occupancy field**
   `f_j(cell) = P(cell carries label j)` reduced from a signature run's
   `label_map.npy`, with the cluster's **50%-mass core** (`worldmap.core_region`,
   the `tau50` of `cluster_probe.py`) outlined dashed on top. Panels carry
   **their own colour scale**: peak occupancy is bimodal across the run (77 of
   128 v6 clusters peak above 0.9, 12 never reach 0.5), so one shared scale
   would saturate the territorial clusters and flatten the itinerant ones into
   the background — the very effect the figure exists to show. This is why the panels are
   rendered here rather than through `worldmap.render_scalar_map`, which shares
   `vmin/vmax` across panels by design; everything else (griddata + Gaussian
   smooth + `pcolormesh` + Natural Earth coastlines) is the same path, so the
   figure sits next to the other maps without looking foreign.

   The three defaults are the clusters the report's prose already names, chosen
   to span the territoriality range: **c27** (tau50 1.000, the Arctic ring
   75-86 N), **c105** (tau50 0.625, the Nino 3.4 box's dominant cluster) and
   **c13** (tau50 0.042, the run's most itinerant cluster: present in 36% of
   cells, half its mass in 4.4% of the globe). Override with `--clusters`.

2. **`fig_seasonality.png`** — the winter/summer pair, side by side in one
   image. `temporal_spatial.py` already writes `map_season_DJF.png` and
   `map_season_JJA.png`, but each is a standalone 13x6.2in figure with its own
   baked-in title, so stacking the two files costs most of an A4 page and the
   titles are unreadable once shrunk to fit. Here the same two maps are
   re-rendered **title-less** through the same `worldmap.render_world_map` with
   the same `affinity_ordered_colors` palette as the report's Fig. 1 (so a
   region keeping its colour keeps its regime across all three maps), then
   composed into one 7in-wide figure whose panel titles are set at the size
   they will print at.

Every number the caption quotes is printed to stdout, so the caption can be
checked against the run instead of trusted.

Run:  python3 src/analysis/report_figures.py
"""

import argparse
import datetime
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.worldmap import (NSIDE, affinity_ordered_colors, build_affinity_matrix, core_region,
                             get_coastlines, healpix_nest2ring, healpix_ring_lonlat,
                             render_world_map)

# Northern-Hemisphere convention, December grouped with the FOLLOWING Jan/Feb --
# the same grouping `temporal_spatial.SEASONS` uses. Redefined here rather than
# imported: only `common/` is importable across role directories (see
# `src/common/__init__.py`), the same reason `partition_nmi.py` copies its `nmi()`.
SEASONS = {"DJF": ((12, 1, 2), "Winter (Dec-Jan-Feb)"),
           "JJA": ((6, 7, 8), "Summer (Jun-Jul-Aug)")}
START = datetime.datetime(2014, 1, 1)                        # file 0, 6-hourly cadence


def occupancy(label_map, clusters, n_cells):
    """Occupancy field f_j(cell) for each requested cluster, streamed over timesteps."""
    occ = np.zeros((len(clusters), n_cells), dtype=np.int64)
    for i in range(0, label_map.shape[0], 1000):
        block = np.asarray(label_map[i:i + 1000])
        for r, k in enumerate(clusters):
            occ[r] += (block == k).sum(0)
    return occ / label_map.shape[0]


def render_territories(fields, cores, titles, out_png):
    """One Mollweide panel per cluster, each on its OWN colour scale (see module docstring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    lon, lat = healpix_ring_lonlat(NSIDE, healpix_nest2ring(NSIDE, np.arange(fields.shape[1])))
    glon, glat = np.arange(-180.0, 180.0, 1.0), np.arange(-90.0, 90.0 + 1e-9, 1.0)
    GX, GY = np.meshgrid(glon, glat)
    gpts, src = np.column_stack([GX.ravel(), GY.ravel()]), np.column_stack([lon, lat])

    def interp(v):
        g = griddata(src, np.asarray(v, dtype=float), gpts, method="linear")
        bad = np.isnan(g)
        if bad.any():
            g[bad] = griddata(src, np.asarray(v, dtype=float), gpts[bad], method="nearest")
        return gaussian_filter(g.reshape(GX.shape), sigma=0.9, mode="nearest")

    coast = get_coastlines()
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("0.90")
    n = len(fields)
    fig = plt.figure(figsize=(5.2 * n, 3.5))
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1, projection="mollweide")
        # Cells the cluster never visits are painted the neutral `bad` grey rather than the
        # colormap's zero end, so an itinerant cluster's faint territory reads against a
        # background instead of disappearing into it.
        G = np.where(interp((fields[i] > 0).astype(float)) >= 0.5, interp(fields[i]), np.nan)
        mesh = ax.pcolormesh(np.radians(GX), np.radians(GY), np.ma.masked_invalid(G),
                             cmap=cmap, vmin=0.0, vmax=float(fields[i].max()),
                             shading="nearest")
        ax.contour(np.radians(GX), np.radians(GY), interp(cores[i].astype(float)),
                   levels=[0.5], colors="#0b6fd4", linewidths=1.2, linestyles="--")
        for poly in coast:
            ax.plot(np.radians(poly[:, 0]), np.radians(poly[:, 1]), color="0.45", lw=0.35)
        ax.set_title(titles[i], fontsize=10, pad=10)
        ax.grid(alpha=0.25)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _autocrop(img, thresh=0.96):
    """Trim the white margin matplotlib leaves around a saved figure."""
    ink = (img[:, :, :3] < thresh).any(2)
    rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    return img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def render_seasonality(run_dir, out_png):
    """Winter and summer dominant-cluster maps, side by side, at print size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    asg = torch.load(os.path.join(run_dir, "assignments.pt"), map_location="cpu", weights_only=False)
    K, n_cells = int(model["config"]["clusters"]), NSIDE * NSIDE * 12
    colors = affinity_ordered_colors(build_affinity_matrix(model["U"], model["means"]), K)

    file_id, cell, label = asg["file_id"].long(), asg["cell_id"].long(), asg["label"].long()
    # `file_id` is the latent file's own id (0..13020), not a row into `sampled_files`,
    # so the calendar month is reconstructed straight from it: file i is START + i*6h.
    month = torch.tensor([(START + datetime.timedelta(hours=6 * i)).month
                          for i in range(int(file_id.max()) + 1)])
    tok_month = month[file_id]

    panels = []
    with tempfile.TemporaryDirectory() as tmp:
        for key, (months, title) in SEASONS.items():
            sel = torch.isin(tok_month, torch.tensor(months))
            cnt = torch.bincount(cell[sel] * K + label[sel], minlength=n_cells * K).view(n_cells, K)
            png = os.path.join(tmp, f"{key}.png")
            render_world_map(cnt.argmax(1), cnt.sum(1) > 0, colors, png, title="")
            panels.append((_autocrop(plt.imread(png)), title, int(sel.sum())))

    # 7.0in wide is the report's \textwidth, so the figure prints 1:1 and a 9pt
    # panel title is a 9pt panel title on the page.
    fig = plt.figure(figsize=(7.0, 2.3))
    for i, (img, title, ntok) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(img)
        ax.set_title(f"{title}, {ntok / 1e6:.1f}M tokens", fontsize=8.5, pad=4)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.03)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return [(t, n) for _, t, n in panels]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sig-dir", default="runs/signatures/v6_d64_margin",
                   help="signature run holding label_map.npy")
    p.add_argument("--run-dir", default="runs/clustering/v6_subspace_big_d64",
                   help="clustering run whose maps/ holds the seasonal maps")
    p.add_argument("--clusters", type=int, nargs="+", default=[27, 105, 13],
                   help="clusters to draw, most territorial first")
    p.add_argument("--out", default="report", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lm = np.load(os.path.join(args.sig_dir, "label_map.npy"), mmap_mode="r")
    n_cells = lm.shape[1]
    lon, lat = healpix_ring_lonlat(NSIDE, healpix_nest2ring(NSIDE, np.arange(n_cells)))

    f = occupancy(lm, args.clusters, n_cells)
    cores, titles = [], []
    print(f"{args.sig_dir}: {lm.shape[0]:,} timesteps x {n_cells:,} cells")
    for i, k in enumerate(args.clusters):
        tau, core = core_region(f[i], 0.5)
        cores.append(core)
        pres, frac = float((f[i] > 0).mean()), float(core.sum()) / n_cells
        titles.append(f"cluster {k}:  50% of its mass in {core.sum():,} cells "
                      f"({frac * 100:.1f}% of globe), $\\tau_{{50}}$={tau:.3f}")
        print(f"  c{k:<4d} tau50={tau:.3f}  core={core.sum():>5,} cells ({frac * 100:5.2f}% of globe)"
              f"  present in {pres * 100:5.1f}% of cells  peak f={f[i].max():.3f}"
              f"  core lat {lat[core].min():+.1f}..{lat[core].max():+.1f}")

    out_png = os.path.join(args.out, "fig_territories.png")
    render_territories(f, cores, titles, out_png)
    print(f"wrote {out_png}")

    season_png = os.path.join(args.out, "fig_seasonality.png")
    for title, ntok in render_seasonality(args.run_dir, season_png):
        print(f"  {title}: {ntok:,} tokens")
    print(f"wrote {season_png}")


if __name__ == "__main__":
    main()
