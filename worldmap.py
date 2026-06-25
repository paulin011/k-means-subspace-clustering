"""Shared HEALPix geometry, cluster coloring, and world-map rendering.

Used by both `analyze_clusters.py` (affinity table) and `temporal_spatial.py` (the
dedicated spatial/temporal report), so the two reports share one geometry + one
cluster-color assignment + one map renderer.

Map rendering has two readability features (see `render_world_map`):
  - **continent outlines** drawn from a cached Natural Earth 110m coastline GeoJSON
    (fetched once, parsed with stdlib `json` -- no cartopy/shapely needed), and
  - a smooth **heatmap** mode that interpolates the per-cell RGB field onto a regular
    grid and paints it as a dense, per-point-projected Mollweide field -- replacing the
    old 12,288 speckled scatter pixels (with white gaps) by a continuous, coast-aligned
    fill. RGB (not cluster id) is interpolated, which is valid because
    `affinity_ordered_colors` already makes subspace-similar clusters share colors, so a
    blended color between two neighbors is still meaningful.
"""

import json
import os
import subprocess

import numpy as np
import torch

NSIDE = 32
N_CELLS = 12 * NSIDE * NSIDE          # 12,288

DEFAULT_COASTLINE = "ne_110m_coastline.geojson"
COASTLINE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                 "master/geojson/ne_110m_coastline.geojson")


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


def build_affinity_matrix(U, means):
    """K×K cluster affinity as a torch tensor (computed once, reused).

    Subspace affinity (mean squared principal-angle cosine) when a basis exists
    (d>0): ‖Uᵢᵀ·Uⱼ‖²_F / d ∈ [0,1]. Centroid cosine mapped to [0,1] via (1+cos)/2
    for point clusters (d=0, e.g. k-means/k-center) -- raw signed cosine would make
    the graph Laplacian indefinite (non-PSD). The d>0 path concatenates the
    per-cluster bases into [DIM, K*d] and does one [K*d, K*d] matmul -- computed
    once here so the world-map coloring and the affinity table share it instead of
    each rebuilding it (at d=128 that matmul is ~1 GB).
    """
    K, D, d = U.shape
    if d > 0:
        Ucat = U.permute(1, 0, 2).reshape(D, K * d)
        return ((Ucat.T @ Ucat).view(K, d, K, d) ** 2).sum((1, 3)) / d
    mu_n = means / means.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (1.0 + mu_n @ mu_n.T) / 2.0     # signed centroid cosine -> [0,1] affinity


def affinity_ordered_colors(aff, K):
    """Color clusters so similar ones get similar colors.

    Orders clusters along the Fiedler vector of the normalized graph Laplacian of
    the precomputed affinity matrix `aff` (the classic 1-D spectral seriation that
    places similar items adjacent), and maps that order through the smooth
    perceptual `turbo` colormap. With this coloring genuine spatial structure
    shows up as smooth gradients; only true salt-and-pepper noise stays speckled
    -- so the map's legibility no longer rides on a random hue shuffle.
    """
    import matplotlib.pyplot as plt
    if K <= 1:                                              # single cluster: no ordering
        return plt.cm.turbo(np.array([0.5]))
    A = aff.detach().cpu().numpy() if torch.is_tensor(aff) else np.asarray(aff)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(K) - dinv[:, None] * A * dinv[None, :]       # normalized Laplacian
    fiedler = np.linalg.eigh(L)[1][:, 1]                    # 2nd-smallest eigenvector
    rank = np.empty(K)
    rank[np.argsort(fiedler)] = np.arange(K)
    return plt.cm.turbo(rank / max(K - 1, 1))


def get_coastlines(cache_path=DEFAULT_COASTLINE, url=COASTLINE_URL):
    """List of Nx2 float arrays (lon_deg, lat_deg), one per coastline polyline.

    Fetches + caches the Natural Earth 110m coastline GeoJSON on first use (so later
    runs work offline), then parses it with stdlib `json` -- no shapely/cartopy.
    """
    if not os.path.exists(cache_path):
        print(f"  fetching coastlines -> {cache_path}", flush=True)
        subprocess.run(["curl", "-sSL", "-o", cache_path, url], check=True)
    with open(cache_path) as f:
        gj = json.load(f)
    polys = []
    for ft in gj["features"]:
        g = ft["geometry"]
        if g["type"] == "LineString":
            polys.append(np.asarray(g["coordinates"], dtype=float))
        elif g["type"] == "MultiLineString":
            polys.extend(np.asarray(part, dtype=float) for part in g["coordinates"])
    return polys


def _cell_lonlat(nside=NSIDE):
    """Per-cell (lon_deg, lat_deg) in [-180,180)×[-90,90] under NESTED ordering."""
    pix = np.arange(12 * nside * nside)
    rp = healpix_nest2ring(nside, pix)                       # NESTED cell ids -> RING
    lon, lat = healpix_ring_lonlat(nside, rp)
    return (lon + 180.0) % 360.0 - 180.0, lat


def render_world_map(dominant, valid, colors, out_png, *, title="", suptitle="",
                     coastlines=True, heatmap=True, nside=NSIDE,
                     grid_deg=1.0, smooth_sigma=0.9):
    """Mollweide map of the dominant cluster per cell (NESTED ordering).

    `colors` is a [K,4] RGBA array (see `affinity_ordered_colors`): subspace-similar
    clusters share colors so real regions read as gradients, and so interpolating the
    RGB field (rather than the categorical cluster id) is meaningful.

    `heatmap=True` (default) paints a smooth continuous field: per-cell RGB is
    interpolated (`scipy.interpolate.griddata`, linear + nearest-fill) onto a regular
    grid and lightly Gaussian-smoothed, then drawn with `pcolormesh` -- native quads
    that matplotlib projects per-corner in the Mollweide axes, so there is no edge
    distortion, no white gaps between cells, and coastlines (drawn the same way) align
    exactly. The RGB is quantized to a `ListedColormap` since `pcolormesh` is single-
    channel; at 8-bit this is visually lossless for a smooth field. `heatmap=False`
    keeps the legacy 12,288-pixel scatter for comparison. `coastlines=True` overlays
    cached Natural Earth 110m coastlines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, NoNorm
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    dom = dominant.numpy() if torch.is_tensor(dominant) else np.asarray(dominant)
    v = (valid.numpy() if torch.is_tensor(valid) else np.asarray(valid)).astype(bool)
    lon, lat = _cell_lonlat(nside)

    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")

    if heatmap and v.any():          # need >=1 valid cell to triangulate; else grey frame
        src = np.column_stack([lon[v], lat[v]])
        vals = colors[dom[v]][:, :3]                         # RGB at data cells
        glon = np.arange(-180.0, 180.0, grid_deg)
        glat = np.arange(-90.0, 90.0 + 1e-9, grid_deg)
        GX, GY = np.meshgrid(glon, glat)
        gpts = np.column_stack([GX.ravel(), GY.ravel()])
        # linear where possible, nearest-fill the rest -> whole globe covered, no gaps
        Grgb = griddata(src, vals, gpts, method="linear")
        Gfill = griddata(src, vals, gpts, method="nearest")
        bad = np.isnan(Grgb[:, 0])
        Grgb[bad] = Gfill[bad]
        Grgb = Grgb.reshape(*GX.shape, 3)
        for c in range(3):                                   # light per-channel smooth
            Grgb[:, :, c] = gaussian_filter(Grgb[:, :, c], sigma=smooth_sigma, mode="nearest")
        Grgb = np.clip(Grgb, 0, 1)
        # quantize to a palette so pcolormesh (single-channel) can paint the RGB field
        rgb8 = (Grgb * 255).astype(np.uint8).reshape(-1, 3)
        uniq, idx = np.unique(rgb8, axis=0, return_inverse=True)
        cmap = ListedColormap(uniq.astype(float) / 255.0)
        ax.pcolormesh(np.radians(GX), np.radians(GY), idx.reshape(GX.shape),
                      cmap=cmap, shading="nearest", norm=NoNorm())
    else:
        rgb = colors[dom].copy()
        rgb[~v, :3] = 0.85                                   # grey where no data
        ax.scatter(np.radians(lon), np.radians(lat), c=rgb, s=7, marker="s", lw=0)

    if coastlines:
        for poly in get_coastlines():
            ax.plot(np.radians(poly[:, 0]), np.radians(poly[:, 1]), color="0.25", lw=0.4)

    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_xticklabels([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
