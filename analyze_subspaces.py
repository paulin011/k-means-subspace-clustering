#!/usr/bin/env python3
"""Analyze K-subspaces clustering output and write a self-contained Markdown report.

Reads model.pt + assignments.pt produced by subspace_kmeans.py and reports:
  - run configuration and convergence history
  - global variance decomposition (between-cluster / within-cluster / residual)
  - per-cluster table: size, explained variance, effective dimensionality,
    spatial concentration over HEALPix cells, temporal coverage/variation
  - subspace affinity between clusters (mean squared cosine of principal
    angles) -- high-affinity pairs indicate clusters that could be merged
  - temporal profiles of the most time-varying clusters
  - a Mollweide world map (dominant_cluster_map.png) of the dominant cluster per
    cell under NESTED ordering (confirmed: coherent continent-scale regions under
    NESTED, incoherent stripes under RING; pure-numpy HEALPix geometry, no healpy)

All tabular spatial statistics are ordering-independent.

Usage:
  python3 analyze_subspaces.py --dir subspace_out --out report.md
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

N_CELLS = 12288
N_FILES_TOTAL = 13021
NSIDE = 32


def healpix_ring_lonlat(nside, p):
    """RING-ordered HEALPix pixel centers -> (lon_deg, lat_deg). No healpy needed."""
    p = np.asarray(p, dtype=np.int64)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    z = np.empty(p.shape)
    phi = np.empty(p.shape)
    m = p < ncap                                            # north polar cap
    pp = p[m] + 1.0
    i = (np.floor(np.sqrt(pp / 2 - np.sqrt(np.floor(pp / 2))))).astype(np.int64) + 1
    j = pp - 2 * i * (i - 1)
    z[m] = 1 - i ** 2 / (3.0 * nside ** 2)
    phi[m] = np.pi / (2 * i) * (j - 0.5)
    m = (p >= ncap) & (p < npix - ncap)                     # equatorial belt
    pp = p[m] - ncap
    i = pp // (4 * nside) + nside
    j = pp % (4 * nside) + 1
    s = (i - nside + 1) % 2
    z[m] = 4.0 / 3 - 2 * i / (3.0 * nside)
    phi[m] = np.pi / (2 * nside) * (j - s / 2)
    m = p >= npix - ncap                                    # south polar cap (mirror)
    pp = (npix - p[m]).astype(np.float64)
    i = (np.floor(np.sqrt(pp / 2 - np.sqrt(np.floor(pp / 2))))).astype(np.int64) + 1
    j = 4 * i + 1 - (pp - 2 * i * (i - 1))
    z[m] = i ** 2 / (3.0 * nside ** 2) - 1
    phi[m] = np.pi / (2 * i) * (j - 0.5)
    return np.degrees(phi), np.degrees(np.arcsin(np.clip(z, -1, 1)))


def healpix_nest2ring(nside, p):
    """Convert NESTED pixel indices to RING indices (standard HEALPix algorithm)."""
    p = np.asarray(p, dtype=np.int64)
    jrll = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    jpll = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])
    face, pp = p // (nside * nside), p % (nside * nside)
    x = np.zeros_like(pp)
    y = np.zeros_like(pp)
    for b in range(nside.bit_length()):                     # de-interleave bits
        x |= ((pp >> (2 * b)) & 1) << b
        y |= ((pp >> (2 * b + 1)) & 1) << b
    jr = jrll[face] * nside - x - y - 1                     # ring number 1..4nside-1
    npix, ncap = 12 * nside * nside, 2 * nside * (nside - 1)
    nr = np.where(jr < nside, jr, np.where(jr > 3 * nside, 4 * nside - jr, nside))
    n_before = np.where(jr < nside, 2 * jr * (jr - 1),
                        np.where(jr > 3 * nside, npix - 2 * nr * (nr + 1),
                                 ncap + (jr - nside) * 4 * nside))
    kshift = np.where((jr >= nside) & (jr <= 3 * nside), (jr - nside) & 1, 0)
    jp = (jpll[face] * nr + x - y + 1 + kshift) // 2
    jp = np.where(jp > 4 * nr, jp - 4 * nr, np.where(jp < 1, jp + 4 * nr, jp))
    return n_before + jp - 1


def affinity_ordered_colors(U):
    """Color clusters so subspace-similar ones get similar colors.

    Builds the K×K subspace-affinity matrix (mean squared principal-angle cosine),
    orders clusters along the Fiedler vector of its normalized graph Laplacian (the
    classic 1-D spectral seriation that places similar items adjacent), and maps that
    order through the smooth perceptual `turbo` colormap. With this coloring genuine
    spatial structure shows up as smooth gradients; only true salt-and-pepper noise
    stays speckled — so the map's legibility no longer rides on a random hue shuffle.
    """
    import matplotlib.pyplot as plt
    K, D, d = U.shape
    Ucat = U.permute(1, 0, 2).reshape(D, K * d)
    A = ((Ucat.T @ Ucat).view(K, d, K, d) ** 2).sum((1, 3)).numpy() / d
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(K) - dinv[:, None] * A * dinv[None, :]       # normalized Laplacian
    fiedler = np.linalg.eigh(L)[1][:, 1]                    # 2nd-smallest eigenvector
    rank = np.empty(K)
    rank[np.argsort(fiedler)] = np.arange(K)
    return plt.cm.turbo(rank / max(K - 1, 1))


def render_world_map(dominant, valid, K, out_png, colors):
    """Mollweide map of the dominant cluster per cell (NESTED ordering, confirmed).

    `colors` is a [K,4] RGBA array (see affinity_ordered_colors): subspace-similar
    clusters receive similar colors so real regions read as gradients, not speckle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pix = np.arange(N_CELLS)
    c = colors[dominant.numpy()].copy()
    c[~valid.numpy()] = (0.85, 0.85, 0.85, 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(13, 6),
                           subplot_kw={"projection": "mollweide"})
    rp = healpix_nest2ring(NSIDE, pix)                       # NESTED cell ids → RING for lon/lat
    lon, lat = healpix_ring_lonlat(NSIDE, rp)
    lon = (lon + 180.0) % 360.0 - 180.0
    ax.scatter(np.radians(lon), np.radians(lat), c=c, s=7, marker="s", lw=0)
    ax.set_title("Dominant cluster per HEALPix cell (NESTED ordering)")
    ax.grid(alpha=0.3)
    ax.set_xticklabels([])
    fig.suptitle("Cells colored by most frequent cluster; subspace-similar clusters "
                 "share similar colors (grey = no data)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="subspace_out", help="directory with model.pt + assignments.pt")
    ap.add_argument("--out", default=None, help="output Markdown file (default: <dir>/report.md)")
    ap.add_argument("--top-pairs", type=int, default=12, help="most similar cluster pairs to list")
    ap.add_argument("--top-temporal", type=int, default=8, help="most time-varying clusters to profile")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.dir, "report.md")

    m = torch.load(os.path.join(args.dir, "model.pt"), map_location="cpu", weights_only=False)
    a = torch.load(os.path.join(args.dir, "assignments.pt"), map_location="cpu", weights_only=False)
    U, means, eig = m["U"], m["means"], m["eigvals"]
    tr, cnt, evr = m["trace"], m["counts"].float(), m["explained_var_ratio"]
    cfg, hist = m["config"], m["history"]
    K, D, d = U.shape
    lab, cell, fid = a["label"].long(), a["cell_id"].long(), a["file_id"].long()
    T = lab.numel()
    w = cnt / cnt.sum()

    L = []
    add = L.append
    add(f"# Subspace clustering report — `{args.dir}`")
    add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `analyze_subspaces.py`. "
        f"K={K} affine subspaces of dim {d} in {D}-dim token space, {T:,} tokens.*")

    add("\n## Configuration\n")
    add("| parameter | value |")
    add("|---|---|")
    for key in ["src", "num_files", "tokens_per_file", "clusters", "dim",
                "iters", "tol", "linear", "seed", "chunk_size", "gpus"]:
        if key in cfg:
            add(f"| {key} | {cfg[key]} |")
    add(f"| tokens analyzed | {T:,} |")

    # ---- Token sample (reproducibility / cross-run comparison) --------------
    add("\n## Token sample\n")
    sampled_files = list(m.get("sampled_files", []))
    fp = m.get("sample_fingerprint")
    if fp is None and sampled_files:                       # older runs predate the field
        h = hashlib.sha1()
        h.update(f"{cfg.get('tokens_per_file', N_CELLS)}|{cfg.get('seed', 0)}|".encode())
        h.update(",".join(map(str, sorted(sampled_files))).encode())
        fp = h.hexdigest()[:12]
    # Backfill sample.json for runs that predate the manifest, so --files-from works.
    sj = os.path.join(args.dir, "sample.json")
    if sampled_files and not os.path.exists(sj):
        with open(sj, "w") as f:
            json.dump({"fingerprint": fp, "num_files": len(sampled_files),
                       "tokens_per_file": cfg.get("tokens_per_file", N_CELLS),
                       "seed": cfg.get("seed", 0), "src": cfg.get("src", "latents_2"),
                       "files": list(sampled_files)}, f)             # load order preserved
        print(f"  backfilled {sj}")
    add(f"- **Sample fingerprint:** `{fp}` — runs sharing this fingerprint were "
        f"clustered on the identical token set and are directly comparable.")
    add(f"- **Files:** {len(sampled_files)} latent files, "
        f"{cfg.get('tokens_per_file', N_CELLS)} tokens each, seed {cfg.get('seed', 0)}.")
    add(f"- **Reproduce this exact sample** for a new run (e.g. to vary K or d):\n")
    add(f"  ```bash\n  python3 subspace_kmeans.py --files-from {args.dir}/sample.json "
        f"--seed {cfg.get('seed', 0)} --tokens-per-file {cfg.get('tokens_per_file', N_CELLS)} \\\n"
        f"      --clusters <K> --dim <d> --out <new_dir>\n  ```")
    if sampled_files:
        preview = ", ".join(map(str, sorted(sampled_files)[:20]))
        add(f"- File ids (first 20 of {len(sampled_files)}, full list in "
            f"`{args.dir}/sample.json`): {preview}{' …' if len(sampled_files) > 20 else ''}")

    add("\n## Convergence\n")
    add("| iter | objective/token | labels changed | min size | max size |")
    add("|---|---|---|---|---|")
    for h in hist:
        add(f"| {h['iter']} | {h['obj_per_token']:.2f} | {h['frac_changed']:.2%} "
            f"| {h['size_min']:,} | {h['size_max']:,} |")

    # ---- Global variance decomposition --------------------------------------
    add("\n## Global variance decomposition\n")
    mu_g = (w[:, None] * means).sum(0)
    between = float((w * ((means - mu_g) ** 2).sum(1)).sum())
    within = float((w * tr).sum())
    total = between + within
    captured = float((w * eig.sum(1)).sum())
    resid = within - captured
    add(f"Total token variance E‖x−μ_global‖² = **{total:.0f}**, split into:\n")
    add(f"- **{between / total:.1%}** between clusters (the means alone — how much "
        f"cluster identity explains)")
    add(f"- **{captured / total:.1%}** within clusters, captured by the top-{d} subspace directions")
    add(f"- **{resid / total:.1%}** residual (unexplained by the model)")
    frac = eig.cumsum(1) / eig.sum(1, keepdim=True).clamp(min=1e-12)
    d80 = (frac < 0.8).sum(1) + 1
    add(f"\nCount-weighted within-cluster EVR(top-{d}): **{float((w * evr).sum()):.3f}**. "
        f"Dimensions needed for 80% of captured variance: "
        f"min {int(d80.min())} / median {int(d80.median())} / max {int(d80.max())} "
        f"(close to {d} ⇒ flat spectrum, consider larger --dim).")

    # ---- Spatial / temporal statistics --------------------------------------
    cell_counts = torch.bincount(cell * K + lab, minlength=N_CELLS * K).view(N_CELLS, K)
    cells_with_data = cell_counts.sum(1) > 0
    dominant = cell_counts.argmax(1)
    owned = torch.bincount(dominant[cells_with_data], minlength=K)

    sorted_cells = cell_counts.float().sort(0, descending=True).values
    cum = sorted_cells.cumsum(0) / sorted_cells.sum(0).clamp(min=1)
    cells50 = (cum < 0.5).sum(0) + 1                       # cells holding 50% of each cluster

    file_present = torch.bincount(fid * K + lab, minlength=N_FILES_TOTAL * K) \
                        .view(N_FILES_TOTAL, K) > 0
    n_sampled_files = int((torch.bincount(fid, minlength=N_FILES_TOTAL) > 0).sum())
    files_pct = file_present.sum(0).float() / n_sampled_files

    dec = fid * 10 // N_FILES_TOTAL                        # time decile by file index
    decK = torch.bincount(dec * K + lab, minlength=10 * K).view(10, K).float()
    dec_tot = decK.sum(1, keepdim=True)
    enrich = (decK / cnt) / (dec_tot / T)                  # 1.0 = temporally flat
    temp_cv = enrich.std(0) / enrich.mean(0).clamp(min=1e-12)

    add("\n## Clusters (sorted by size)\n")
    add(f"Spatial columns are over the {int(cells_with_data.sum())} HEALPix cells with data; "
        f"`cells@50%` = number of cells holding half the cluster's tokens (low = localized); "
        f"`owned` = cells where this cluster is the most common label; "
        f"`files` = share of the {n_sampled_files} sampled time steps where the cluster appears; "
        f"`tCV` = coefficient of variation of its share across time deciles (0 = constant in time).\n")
    add(f"| cluster | tokens | share | EVR(top-{d}) | d80 | cells@50% | owned | files | tCV |")
    add("|---|---|---|---|---|---|---|---|---|")
    for j in cnt.argsort(descending=True).tolist():
        add(f"| {j} | {int(cnt[j]):,} | {float(w[j]):.1%} | {float(evr[j]):.3f} | {int(d80[j])} "
            f"| {int(cells50[j])} | {int(owned[j])} | {float(files_pct[j]):.0%} "
            f"| {float(temp_cv[j]):.2f} |")

    # ---- Subspace affinity ---------------------------------------------------
    add("\n## Subspace affinity between clusters\n")
    add(f"Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / {d} ∈ [0,1]: mean squared cosine of the principal "
        "angles between the two subspaces (1 = identical span, 0 = orthogonal). "
        "High-affinity pairs are candidates for merging (K may be too large); "
        "uniformly low values mean genuinely distinct regimes.\n")
    Ucat = U.permute(1, 0, 2).reshape(D, K * d)
    G = (Ucat.T @ Ucat).view(K, d, K, d)
    aff = (G ** 2).sum((1, 3)) / d
    mu_n = means / means.norm(dim=1, keepdim=True).clamp(min=1e-12)
    mcos = mu_n @ mu_n.T
    iu = torch.triu_indices(K, K, offset=1)
    offdiag = aff[iu[0], iu[1]]
    add(f"Off-diagonal affinity: median {float(offdiag.median()):.3f}, "
        f"mean {float(offdiag.mean()):.3f}, max {float(offdiag.max()):.3f}.\n")
    add("| pair | subspace affinity | mean-vector cosine |")
    add("|---|---|---|")
    top = offdiag.argsort(descending=True)[:args.top_pairs]
    for t in top.tolist():
        i, j = int(iu[0, t]), int(iu[1, t])
        add(f"| {i} ↔ {j} | {float(aff[i, j]):.3f} | {float(mcos[i, j]):.3f} |")

    # ---- Temporal profiles ---------------------------------------------------
    add("\n## Most time-varying clusters\n")
    add("Enrichment of each cluster per time decile of the dataset (file index 0…13020; "
        "1.00 = the cluster's average rate). Values ≫1 mark the periods where the "
        "cluster concentrates — a strong seasonal/temporal signature.\n")
    add("| cluster | tCV | " + " | ".join(f"D{t}" for t in range(10)) + " |")
    add("|---|---|" + "---|" * 10)
    for j in temp_cv.argsort(descending=True)[:args.top_temporal].tolist():
        row = " | ".join(f"{float(enrich[t, j]):.2f}" for t in range(10))
        add(f"| {j} | {float(temp_cv[j]):.2f} | {row} |")

    # ---- World map -------------------------------------------------------
    add("\n## World map\n")
    try:
        png = "dominant_cluster_map.png"
        colors = affinity_ordered_colors(U)
        render_world_map(dominant, cells_with_data, K, os.path.join(args.dir, png), colors)
        add(f"![Dominant cluster per HEALPix cell]({png})\n")
        add("Each of the 12,288 HEALPix cells is colored by its most frequent cluster "
            "(grey = no data). Cell indices use **NESTED HEALPix ordering** (confirmed: "
            "geographically coherent continent-scale regions appear under NESTED, "
            "incoherent stripes under RING). Colors are assigned by spectral ordering of "
            "the subspace-affinity matrix, so subspace-similar clusters share similar "
            "hues — real regions read as smooth gradients, genuine noise stays speckled.")
    except Exception as e:                      # report must still be written
        add(f"_Map rendering failed: {e}_")

    add("\n## Interpretation notes\n")
    add("- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the "
        "cluster is a **geographic regime** (region/surface type), stable in time.")
    add("- *High `tCV` with smooth decile profile* ⇒ **seasonal or trend** behaviour; check the "
        "decile table above.")
    add("- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the "
        "spectrum; re-run with larger `--dim` to capture more structure.")
    add("- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, "
        "descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.")
    add("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print(f"Report written to {out_path} ({len(L)} lines)")


if __name__ == "__main__":
    main()
