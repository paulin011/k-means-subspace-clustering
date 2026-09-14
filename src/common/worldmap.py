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

Also here, for `cluster_probe.py` (single-cluster reports):
  - `render_scalar_map` -- the continuous-field twin of `render_world_map`: one or more
    Mollweide panels of a per-cell scalar through a colormap, same griddata + coastline
    path, with an optional interpolated mask and a second region's contour on top.
  - `core_region(f, q)` -- the highest-density region: the smallest cell set holding a
    fraction `q` of a field's mass, i.e. the report's `cells@50%` generalised to any `q`.
  - `healpix_nest_neighbours` -- the real NESTED 8-neighbour query (reference face-crossing
    tables), replacing the `cell >> 2` sibling proxy that fragments connected regions at
    every HEALPix block boundary.
  - `cell_lonlat` / `cell_xyz` -- per-cell geometry, the latter for spherical averaging.
"""

import json
import os
import subprocess

import numpy as np
import torch

NSIDE = 32
N_CELLS = 12 * NSIDE * NSIDE          # 12,288

DEFAULT_COASTLINE = "assets/ne_110m_coastline.geojson"
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
    # HEALPix staggers alternate belt rings by half a pixel. The reference offset is
    #   fodd = 1 if (iring + nside) is odd else 1/2
    # (healpix_base.cc pix2ang, equatorial branch). An earlier version of this line used
    # `(i - nside + 1) % 2 / 2`, i.e. an offset of 0 or 1/2, which is short by a full pixel
    # spacing on every ring where (iring+nside) is odd -- a rigid 2.8125 deg longitude
    # rotation of half the belt rings at nside=32. Caught 2026-08-13 by the neighbour query
    # below (neighbouring pixel centres came out up to 2.4x the pixel scale apart) and
    # confirmed against the nside=1 ground truth, where the four equatorial base pixels must
    # sit at lon 0/90/180/270: the old line returned 90/180/270/0. Fine-scale only -- the
    # smoothed heatmaps in the existing reports are visually unaffected -- but it matters for
    # anything measuring distance on the sphere (see cluster_probe.py's `rival_km`).
    fodd = np.where(((i + nside) & 1) == 1, 1.0, 0.5)
    z[m] = 4.0 / 3 - 2 * i / (3.0 * nside)
    phi[m] = np.pi / (2 * nside) * (j - fodd)
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


# ---------------------------------------------------------------------------
# NESTED neighbour query (the reference HEALPix face-crossing tables)
# ---------------------------------------------------------------------------
# A NESTED pixel id de-interleaves into (face, x, y). Eight of the nine 3x3 offsets
# leave the face; `_NB_FACE[nbnum, face]` says which face you land on (-1 = no pixel
# there at all) and `_NB_SWAP[nbnum, face >> 2]` says how (x, y) must be reflected /
# transposed to express the landing point in the new face's own axes.
# `nbnum` = 4 + (x underflow -1 / overflow +1) + 3*(y underflow -1 / overflow +1), so
# the nine rows are S, SE, E, SW, centre, NE, W, NW, N in that order.
_NB_XOFF = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
_NB_YOFF = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_NB_FACE = np.array([
    [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],        # 0  S
    [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],            # 1  SE
    [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],        # 2  E
    [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],            # 3  SW
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],              # 4  centre
    [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],                # 5  NE
    [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],        # 6  W
    [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],                # 7  NW
    [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3]])           # 8  N  (mirror image of S)
_NB_SWAP = np.array([[0, 0, 3], [0, 0, 6], [0, 0, 0], [0, 0, 5], [0, 0, 0],
                     [5, 0, 0], [0, 0, 0], [6, 0, 0], [3, 0, 0]])


def _nest2xyf(nside, p):
    """NESTED pixel id -> (x, y, face). x takes the even bits, y the odd ones."""
    p = np.asarray(p, dtype=np.int64)
    face, pp = p // (nside * nside), p % (nside * nside)
    x = np.zeros_like(pp)
    y = np.zeros_like(pp)
    for b in range(nside.bit_length()):
        x |= ((pp >> (2 * b)) & 1) << b
        y |= ((pp >> (2 * b + 1)) & 1) << b
    return x, y, face


def _xyf2nest(nside, x, y, face):
    """(x, y, face) -> NESTED pixel id (inverse of `_nest2xyf`)."""
    pp = np.zeros_like(x)
    for b in range(nside.bit_length()):
        pp |= ((x >> b) & 1) << (2 * b)
        pp |= ((y >> b) & 1) << (2 * b + 1)
    return face * (nside * nside) + pp


def healpix_nest_neighbours(nside, p=None):
    """The up-to-8 NESTED neighbours of each pixel, as an `[N, 8]` int64 array.

    Column order is the `_NB_XOFF`/`_NB_YOFF` ring around the pixel; **-1 marks a
    direction with no pixel in it**. Exactly 24 of the 12,288 nside=32 pixels have 7
    rather than 8 neighbours: the eight HEALPix corner vertices (lon 0/90/180/270 at
    lat +/-41.81 deg) are shared by only *three* pixels instead of four.

    Why this and not "same HEALPix parent block" (`cell >> 2`): the parent-block test is
    a cheap proxy that links a pixel only to its 3 siblings inside one 2x2 block and
    never across a block boundary, so a spatially connected patch fragments at every
    block edge. That proxy split single physical events in two while this plan was being
    measured (`docs/ideas/cluster_probe.md` §3.3); this is the real query.

    Validated data-free at nside=32: the relation is symmetric (0 asymmetric pairs of
    49,140 undirected edges), has no self-loops and no duplicate entries per row, and
    every neighbour's centre lies within 2.04x the mean pixel scale (1.83 deg) -- the
    check that caught a wrong `N` row in an earlier transcription of the face table.
    """
    p = np.arange(12 * nside * nside) if p is None else np.asarray(p, dtype=np.int64)
    ix, iy, face = _nest2xyf(nside, p)
    out = np.empty((p.size, 8), dtype=np.int64)
    for i in range(8):
        x, y = ix + _NB_XOFF[i], iy + _NB_YOFF[i]
        nb = np.full(p.shape, 4, dtype=np.int64)
        nb = np.where(x < 0, nb - 1, np.where(x >= nside, nb + 1, nb))
        x = np.where(x < 0, x + nside, np.where(x >= nside, x - nside, x))
        nb = np.where(y < 0, nb - 3, np.where(y >= nside, nb + 3, nb))
        y = np.where(y < 0, y + nside, np.where(y >= nside, y - nside, y))
        f = _NB_FACE[nb, face]
        bits = _NB_SWAP[nb, face >> 2]
        xs = np.where(bits & 1, nside - x - 1, x)
        ys = np.where(bits & 2, nside - y - 1, y)
        xf = np.where(bits & 4, ys, xs)                      # bits & 4 -> transpose
        yf = np.where(bits & 4, xs, ys)
        out[:, i] = np.where(f >= 0, _xyf2nest(nside, xf, yf, np.maximum(f, 0)), -1)
    return out


def core_region(f, q):
    """Highest-density region: `(tau, mask)` for the smallest cell set holding mass `q`.

    `f` is a per-cell non-negative field (in `cluster_probe.py`: the cluster's occupancy
    `f_j(cell) = P(cell carries label j)`). Sorting `f` descending and walking the
    cumulative share until it reaches `q` gives the threshold `tau = f` at the crossing;
    the region is `f >= tau`.

    This is the fix for the fact that **no fixed occupancy threshold can be shared across
    clusters**: peak occupancy is bimodal (77 of v6's 128 clusters peak above 0.9 while 12
    never reach 0.5), so tau=0.5 draws a solid blob for one kind and an empty map for the
    other. Sliding *mass coverage* instead is scale-free, and it is exactly the report's
    `cells@50%` column generalised to arbitrary q.

    `int(mask.sum())` is the drawn cell count. It can exceed the *minimal* count when
    cells tie at `tau` (v6 c27 has 106 cells at f=1.000, all of which must be drawn),
    which is why the count is taken as `f >= tau` rather than as the crossing index.
    """
    f = np.asarray(f, dtype=np.float64)
    tot = f.sum()
    if tot <= 0:
        return 0.0, np.zeros(f.shape, dtype=bool)
    order = np.argsort(f)[::-1]
    cum = np.cumsum(f[order]) / tot
    k = min(int(np.searchsorted(cum, q, side="left")), f.size - 1)
    tau = float(f[order[k]])
    return tau, f >= tau


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


cell_lonlat = _cell_lonlat        # public alias (the leading underscore predates outside use)


def cell_xyz(nside=NSIDE):
    """Per-cell unit vectors `[N_CELLS, 3]` under NESTED ordering.

    Averaging positions on a sphere has to be done in 3-D and re-normalised; averaging
    lon/lat directly is wrong at the dateline and meaningless at the poles. Every
    centroid / great-circle distance in `cluster_probe.py` goes through this.
    """
    lon, lat = _cell_lonlat(nside)
    rl, rb = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(rb) * np.cos(rl), np.cos(rb) * np.sin(rl), np.sin(rb)], 1)


def render_scalar_map(fields, out_png, *, titles=None, suptitle="", cmap="viridis",
                      vmin=None, vmax=None, mask=None, outline=None, outline_color="k",
                      cbar_label="", ncols=None, coastlines=True, nside=NSIDE,
                      grid_deg=1.0, smooth_sigma=0.9, bad_color="0.88", figsize=None):
    """Continuous-field twin of `render_world_map`: one or more Mollweide panels.

    `render_world_map` paints a *categorical* dominant-cluster field by interpolating its
    RGB. This paints a *scalar* field (occupancy, an occupancy anomaly, a residual) through
    a colormap, sharing the same griddata + Gaussian-smooth + `pcolormesh` + Natural Earth
    coastline path so the new maps sit next to the old ones without looking foreign.

      fields   [N_CELLS] array, or a list of them -> one panel each (a shared colorbar).
      mask     [N_CELLS] bool (or a list, one per panel). Cells outside the mask are
               painted `bad_color`; the mask is interpolated as a 0/1 field and cut at 0.5,
               so the boundary follows the same smoothing as the data instead of showing
               the raw 12,288-cell staircase.
      outline  [N_CELLS] bool (or a list): drawn as a single contour line, for putting a
               second region's outline on top of a panel (the rival's core in P1).
      vmin/vmax  shared across panels (default: the min/max over all panels), so panels
               are comparable -- the whole point of a ladder or a seasonal quartet.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    def _aslist(v, n):
        if v is None:
            return [None] * n
        return list(v) if isinstance(v, (list, tuple)) else [v] * n

    fields = [fields] if np.ndim(fields[0]) == 0 else list(fields)
    n = len(fields)
    masks, outlines = _aslist(mask, n), _aslist(outline, n)
    titles = _aslist(titles, n)
    lon, lat = _cell_lonlat(nside)
    glon = np.arange(-180.0, 180.0, grid_deg)
    glat = np.arange(-90.0, 90.0 + 1e-9, grid_deg)
    GX, GY = np.meshgrid(glon, glat)
    gpts = np.column_stack([GX.ravel(), GY.ravel()])
    src = np.column_stack([lon, lat])

    def _interp(v):
        """Interpolate a per-cell field onto the regular grid (linear, nearest-filled)."""
        g = griddata(src, np.asarray(v, dtype=float), gpts, method="linear")
        bad = np.isnan(g)
        if bad.any():
            g[bad] = griddata(src, np.asarray(v, dtype=float), gpts[bad], method="nearest")
        return g.reshape(GX.shape)

    vals = [np.asarray(f, dtype=float) for f in fields]
    shown = [v if masks[i] is None else v[masks[i]] for i, v in enumerate(vals)]
    shown = [s for s in shown if s.size]                      # an empty mask contributes nothing
    if vmin is None:
        vmin = float(min(s.min() for s in shown)) if shown else 0.0
    if vmax is None:
        vmax = float(max(s.max() for s in shown)) if shown else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    ncols = ncols or (1 if n == 1 else (2 if n <= 4 else 3))
    nrows = int(np.ceil(n / ncols))
    # A Mollweide panel draws a 2:1 ellipse inside its allocated box, so a row needs
    # width/2 for the map itself plus headroom for its title; with too little, the next
    # row's title lands in the previous row's whitespace.
    pw = 7.0                                                  # panel width, inches
    fig = plt.figure(figsize=figsize or (pw * ncols, (pw / 2 + 0.85) * nrows + 0.7))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(bad_color)
    coast = get_coastlines() if coastlines else []
    mesh = None
    for i, v in enumerate(vals):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="mollweide")
        G = gaussian_filter(_interp(v), sigma=smooth_sigma, mode="nearest")
        if masks[i] is not None:
            G = np.where(_interp(masks[i].astype(float)) >= 0.5, G, np.nan)
        mesh = ax.pcolormesh(np.radians(GX), np.radians(GY), np.ma.masked_invalid(G),
                             cmap=cm, vmin=vmin, vmax=vmax, shading="nearest")
        if outlines[i] is not None and outlines[i].any():
            ax.contour(np.radians(GX), np.radians(GY),
                       gaussian_filter(_interp(outlines[i].astype(float)), sigma=smooth_sigma,
                                       mode="nearest"),
                       levels=[0.5], colors=outline_color, linewidths=0.9, linestyles="--")
        for poly in coast:
            ax.plot(np.radians(poly[:, 0]), np.radians(poly[:, 1]), color="0.25", lw=0.4)
        if titles[i]:
            ax.set_title(titles[i], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    # Reserve the colorbar strip in INCHES, not as a fraction of the figure: the figure
    # grows with the row count, so a fixed fraction leaves a 1-row plot roomy and clips the
    # label off the bottom of a 2-row one.
    H = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0.80 / H, 1, 1))
    cax = fig.add_axes((0.25, 0.42 / H, 0.5, 0.16 / H))
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


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

def render_change_map(dom_a, valid_a, dom_b, valid_b, colors, out_png, *,
                      mode="dest", top_n=8, title="", coastlines=True, heatmap=True,
                      grey=(0.85, 0.85, 0.87), nside=NSIDE):
    """Map the cells whose dominant cluster changed between state A and state B.

    Only cells valid in BOTH states are comparable; among those, cells whose dominant
    cluster differs are 'changed'. Changed cells are colored, everything else (held, or
    not comparable) is painted neutral grey. The field is made DENSE (every cell gets a
    value, held -> an appended grey palette row) so the smooth `render_world_map` heatmap
    path interpolates honestly instead of extrapolating from sparse points.

    mode:
      "dest"  -> color each changed cell by its state-B (destination) cluster, using the
                 shared `colors` palette (so hues match the per-state maps).
      "src"   -> color each changed cell by its state-A (source) cluster, same palette.
      "trans" -> color by transition: the `top_n` most common (A,B) flows each get a
                 distinct qualitative color; rarer flips fall back to grey. Returns a
                 legend so the caller can render a table.

    Returns (n_changed, n_comparable, legend). `legend` is [] for dest/src, or a list of
    (slot, a, b, n_cells) rows for trans (slot indexes the qualitative palette order).
    """
    import matplotlib.pyplot as plt

    a = dom_a.long() if torch.is_tensor(dom_a) else torch.as_tensor(dom_a).long()
    b = dom_b.long() if torch.is_tensor(dom_b) else torch.as_tensor(dom_b).long()
    va = valid_a if torch.is_tensor(valid_a) else torch.as_tensor(valid_a)
    vb = valid_b if torch.is_tensor(valid_b) else torch.as_tensor(valid_b)
    both = va.bool() & vb.bool()                              # comparable cells
    changed = both & (a != b)                                # flipped cells
    n_changed, n_comp = int(changed.sum()), int(both.sum())

    K = colors.shape[0]
    grey_row = np.asarray([[grey[0], grey[1], grey[2], 1.0]])
    legend = []

    if mode in ("dest", "src"):
        src_field = b if mode == "dest" else a               # which cluster names the color
        grey_slot = K
        field = torch.full((N_CELLS,), grey_slot, dtype=torch.long)
        field[changed] = src_field[changed]
        palette = np.vstack([colors, grey_row])              # [K+1, 4]
    elif mode == "trans":
        pair_id = a * K + b                                  # unique id per (A,B) pair
        cp = pair_id[changed]
        uniq, counts = torch.unique(cp, return_counts=True)
        top = uniq[torch.argsort(counts, descending=True)][:top_n]
        n_top = len(top)
        # qualitative, maximally-distinct colors; held/rare -> last (grey) slot
        base = plt.cm.tab10 if n_top <= 10 else plt.cm.tab20
        qual = base(np.linspace(0, 1, 10 if n_top <= 10 else 20))[:n_top]
        palette = np.vstack([qual, grey_row])               # [n_top+1, 4]
        grey_slot = n_top
        field = torch.full((N_CELLS,), grey_slot, dtype=torch.long)
        flip_idx = changed.nonzero(as_tuple=True)[0]
        for slot, p in enumerate(top.tolist()):
            sel = flip_idx[(cp == p)]
            field[sel] = slot
            pa, pb = divmod(p, K)
            legend.append((slot, pa, pb, int((cp == p).sum())))
    else:
        raise ValueError(f"mode must be 'dest', 'src', or 'trans'; got {mode!r}")

    valid_all = torch.ones(N_CELLS, dtype=torch.bool)        # dense: every cell has a value
    render_world_map(field, valid_all, palette, out_png,
                     title=title, coastlines=coastlines, heatmap=heatmap, nside=nside)
    return n_changed, n_comp, legend
