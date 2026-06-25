#!/usr/bin/env python3
"""Analyze clustering output and write a self-contained Markdown report.

Algorithm-agnostic: reads any model.pt + assignments.pt following the shared schema
in cluster_io.py (produced by subspace_kmeans.py, or by k-means/k-center, which are the
d=0 "point cluster" case -- no subspace basis, just a centroid). Reports:
  - run configuration and convergence history
  - global variance decomposition (between-cluster / within-cluster / residual)
  - per-cluster table: size, spatial concentration over HEALPix cells, temporal
    coverage/variation, plus (when d>0) explained variance / effective dimensionality,
    and (when present) k-center's radius -- the max distance from centroid to any
    member, its native minimax objective (distinct from `trace`, the mean squared
    distance that k-means/subspace_kmeans optimize)
  - subspace affinity between clusters (d>0 only): mean squared cosine of principal
    angles -- high-affinity pairs indicate clusters that could be merged

The **world map and the temporal/seasonal analysis live in the dedicated temporal &
spatial report** (`temporal_spatial.py` -> `temporal_report.md`): 12 monthly
dominant-cluster maps, an annual map with continent outlines + smooth heatmap, monthly
enrichment and month-to-month stability. This report keeps only the compact per-cluster
spatial/temporal columns and points to that report. Map rendering and HEALPix geometry
live in `worldmap.py`.

Usage:
  python3 analyze_clusters.py --dir subspace_out --out report.md
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from cluster_io import OVERFIT_GAP_THRESHOLD
from worldmap import build_affinity_matrix

N_CELLS = 12288
N_FILES_TOTAL = 13021


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="subspace_out", help="directory with model.pt + assignments.pt")
    ap.add_argument("--out", default=None, help="output Markdown file (default: <dir>/report.md)")
    ap.add_argument("--top-pairs", type=int, default=12, help="most similar cluster pairs to list")
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
    sampled_files = list(m.get("sampled_files", []))   # actual files used (ground truth)

    method = cfg.get("method", "subspace_kmeans")
    L = []
    add = L.append
    add(f"# Clustering report ({method}) — `{args.dir}`")
    if d > 0:
        add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `analyze_clusters.py`. "
            f"K={K} affine subspaces of dim {d} in {D}-dim token space, {T:,} tokens.*")
    else:
        add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `analyze_clusters.py`. "
            f"K={K} point clusters in {D}-dim token space, {T:,} tokens.*")

    add("\n## Configuration\n")
    add("| parameter | value |")
    add("|---|---|")
    for key in ["src", "num_files", "tokens_per_file", "clusters", "dim",
                "iters", "tol", "linear", "seed", "chunk_size", "gpus"]:
        if key == "num_files":
            # cfg["num_files"] is the --num-files ARG, which --files-from ignores, so for
            # runs that reuse a previous sample it can be a stale default (e.g. 1500).
            # Report the actual file count from sampled_files and flag the override.
            arg = cfg.get("num_files")
            note = (f" *(--files-from reused the sample; --num-files={arg} ignored)*"
                    if cfg.get("files_from") else "")
            add(f"| num_files | {len(sampled_files)}{note} |")
        elif key in cfg:
            add(f"| {key} | {cfg[key]} |")
    add(f"| tokens analyzed | {T:,} |")

    # ---- Token sample (reproducibility / cross-run comparison) --------------
    add("\n## Token sample\n")
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
    add(f"- **Reproduce this exact sample** for a new run, with this or any other "
        f"`cluster_io.py`-based script (e.g. to vary K, d, or the algorithm itself):\n")
    add(f"  ```bash\n  python3 {method}.py --files-from {args.dir}/sample.json "
        f"--seed {cfg.get('seed', 0)} --tokens-per-file {cfg.get('tokens_per_file', N_CELLS)} \\\n"
        f"      --clusters <K> --out <new_dir>\n  ```")
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
    captured = float((w * eig.sum(1)).sum()) if d > 0 else 0.0
    resid = within - captured
    add(f"Total token variance E‖x−μ_global‖² = **{total:.0f}**, split into:\n")
    add(f"- **{between / total:.1%}** between clusters (the means alone — how much "
        f"cluster identity explains)")
    if d > 0:
        add(f"- **{captured / total:.1%}** captured within clusters by the top-{d} subspace directions")
        add(f"- **{resid / total:.1%}** residual (unexplained by the model)")
        frac = eig.cumsum(1) / eig.sum(1, keepdim=True).clamp(min=1e-12)
        d80 = (frac < 0.8).sum(1) + 1
        add(f"\nCount-weighted within-cluster EVR(top-{d}): **{float((w * evr).sum()):.3f}**. "
            f"Dimensions needed for 80% of captured variance: "
            f"min {int(d80.min())} / median {int(d80.median())} / max {int(d80.max())} "
            f"(d80 is capped at d+1={d + 1}: a cluster reaching that value never hits 80% "
            f"even using all {d} kept directions).")
    else:
        add(f"- **{resid / total:.1%}** residual, i.e. within-cluster (point clusters: "
            f"no subspace basis, so nothing beyond the centroid is captured)")

    # ---- Held-out generalization (only if holdout_eval.py has been run) ------
    hj = os.path.join(args.dir, "holdout.json")
    if os.path.exists(hj):
        with open(hj) as f:
            ho = json.load(f)
        add("\n## Held-out generalization\n")
        add(f"The variance split above is measured on the tokens the model was *fit* on. "
            f"`holdout_eval.py` froze the trained means and bases and replayed the assignment "
            f"rule on **{ho['num_holdout_tokens']:,} tokens from {ho['num_holdout_files']} "
            f"latent files the run never saw**. A held-out residual close to the in-sample "
            f"residual means the subspaces capture reusable structure rather than memorising "
            f"the training tokens.\n")
        g = ho["holdout"]["residual_frac"] - ho["in_sample"]["residual_frac"]
        verdict = ("**overfitting** — the bases are too expensive for the signal; lower `--dim` "
                   "or add tokens" if g > OVERFIT_GAP_THRESHOLD else
                   "the subspaces **generalise**; the in-sample residual is trustworthy")
        add("| residual (unexplained variance) | fraction |")
        add("|---|---|")
        add(f"| in-sample | {ho['in_sample']['residual_frac']:.1%} |")
        add(f"| held-out | {ho['holdout']['residual_frac']:.1%} |")
        add(f"| generalization gap | {g:+.1%} |")
        add(f"\nObjective/token (the minimised orthogonal residual): "
            f"in-sample **{ho['in_sample']['final_obj_per_token']:.2f}** vs held-out "
            f"**{ho['holdout']['residual']:.2f}**. Verdict: {verdict}.")

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
    dec_tot = decK.sum(1, keepdim=True)                    # 0 for an empty (unsampled) decile
    # nan marks an empty decile; temp_cv is taken over populated deciles only, so a sparse
    # sample can't fabricate a temporal signal. (cnt.clamp guards an empty cluster; the
    # decile divisor is clamped separately so an empty decile -> 0, then masked to nan.)
    dden = (dec_tot / T).clamp(min=1e-12)
    raw = (decK / cnt.clamp(min=1)) / dden
    enrich = torch.where(dec_tot > 0, raw, torch.full_like(raw, float("nan")))
    enr_np = enrich.numpy()
    temp_cv = torch.tensor(np.nanstd(enr_np, axis=0) /
                           np.nanmean(enr_np, axis=0).clip(1e-12))

    radius = m.get("radius")

    add("\n## Clusters (sorted by size)\n")
    add(f"Spatial columns are over the {int(cells_with_data.sum())} HEALPix cells with data; "
        f"`cells@50%` = number of cells holding half the cluster's tokens (low = localized); "
        f"`owned` = cells where this cluster is the most common label; "
        f"`files` = share of the {n_sampled_files} sampled time steps where the cluster appears; "
        f"`tCV` = coefficient of variation of its share across time deciles (0 = constant in time)."
        + (" `radius` = k-center's native objective, the max distance from centroid to any "
           "member (vs. `tCV`-adjacent `trace`, the mean squared distance k-means/subspace "
           "optimize)." if radius is not None else "") + "\n")
    cols = ["cluster", "tokens", "share"]
    if d > 0:
        cols += [f"EVR(top-{d})", "d80"]
    cols += ["cells@50%", "owned", "files", "tCV"]
    if radius is not None:
        cols += ["radius"]
    add("| " + " | ".join(cols) + " |")
    add("|" + "---|" * len(cols))
    for j in cnt.argsort(descending=True).tolist():
        row = [f"{j}", f"{int(cnt[j]):,}", f"{float(w[j]):.1%}"]
        if d > 0:
            row += [f"{float(evr[j]):.3f}", f"{int(d80[j])}"]
        tcv = float(temp_cv[j])
        row += [f"{int(cells50[j])}", f"{int(owned[j])}", f"{float(files_pct[j]):.0%}",
                ("–" if tcv != tcv else f"{tcv:.2f}")]
        if radius is not None:
            row += [f"{float(radius[j]):.2f}"]
        add("| " + " | ".join(row) + " |")

    # ---- Temporal & spatial analysis (pointer) ------------------------------
    ts_report = os.path.join(args.dir, "temporal_report.md")
    add("\n## Temporal & spatial analysis\n")
    add("The world map, 12 monthly dominant-cluster maps, and seasonal profiles live in the "
        "dedicated **temporal & spatial report** (`temporal_spatial.py`), read at calendar "
        "(monthly) resolution with continent outlines and a smooth heatmap — clearer than a "
        "single 12,288-pixel map. Generate it from this run's frozen model + assignments:\n")
    add(f"  ```bash\n  python3 temporal_spatial.py --dir {args.dir} "
        f"--out {args.dir}/temporal_report.md\n  ```")
    add(f"\nThe per-cluster `cells@50%` / `owned` / `files` / `tCV` columns above are the "
        f"compact in-report summary of that same spatial/temporal structure"
        f"{f'; see `{ts_report}`' if os.path.exists(ts_report) else ''}.")

    # ---- Subspace affinity (only meaningful with an actual basis, d>0) ------
    aff = build_affinity_matrix(U, means)                  # [K,K]
    if d > 0:
        add("\n## Subspace affinity between clusters\n")
        add(f"Affinity(i,j) = ‖Uᵢᵀ·Uⱼ‖²_F / {d} ∈ [0,1]: mean squared cosine of the principal "
            "angles between the two subspaces (1 = identical span, 0 = orthogonal). "
            "High-affinity pairs are candidates for merging (K may be too large); "
            "uniformly low values mean genuinely distinct regimes.\n")
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
    else:
        add("\n## Subspace affinity between clusters\n")
        add("_Skipped: point clusters (d=0) have no subspace basis to compare._")

    add("\n## Interpretation notes\n")
    add("- *Localized + present in ~100% of files* (low `cells@50%`, `files` ≈ 100%) ⇒ the "
        "cluster is a **geographic regime** (region/surface type), stable in time.")
    add("- *High `tCV`* ⇒ **seasonal or trend** behaviour; see the monthly profiles in the "
        "temporal & spatial report (`temporal_spatial.py`).")
    if d > 0:
        add("- *EVR near the global average with d80 ≈ d* ⇒ the subspace dimension truncates the "
            "spectrum; re-run with larger `--dim` to capture more structure.")
        add("- Subspace bases live in `model.pt['U']` `[K, 2048, d]` (orthonormal columns, "
            "descending eigenvalue order); project tokens with `(x-μ_j) @ U_j`.")
    if radius is not None:
        add("- *Large `radius` relative to `trace`* ⇒ an outlier-driven cluster; k-center "
            "minimizes the worst case, so a few far points can still leave `radius` high "
            "even with low average (`trace`) variance.")
    add("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print(f"Report written to {out_path} ({len(L)} lines)")


if __name__ == "__main__":
    main()
