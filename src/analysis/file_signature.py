#!/usr/bin/env python3
"""Per-file regime signature + subspace residual for *every* latent file.

Given a frozen clustering run (a trained `model.pt` with means mu_j and per-cluster
bases U_j), this replays the exact assignment rule the algorithm optimises --

    residual_j(x) = ||x - mu_j||^2 - ||U_j^T (x - mu_j)||^2     (affine, the default)
                  = ||x - mu_j||^2                              (point cluster, d=0)

-- on **all 12,288 tokens of every one of the 13,021 latent files**, and reduces the
result to a compact per-FILE (per-timestep) summary. That summary is the coordinate
system the "which timestamps to train on" question (goal 1) is built on:

  * `mix`            [K]  -- fraction of the file's 12,288 cells in each cluster
                              (the file's regime-mix signature; low-dim fingerprint of
                              the snapshot, stable because clusters are geographic).
  * `mean_residual`  scalar -- mean orthogonal residual over the file's tokens, i.e. the
                              part of the state NOT explained by "which regime + its
                              ~d-dim within-regime variation". The model's global
                              objective/token is the count-weighted mean of this.
  * `residual_frac`  scalar -- mean_residual / mean_total  (== the file's share of
                              variance left unexplained). Globally ~31.5% for v6;
                              per-file outliers are the "hard-to-encode / anomalous"
                              timestamps -- candidates for upweighting.
  * `rare_exposure`  scalar -- share of the file's tokens in the globally-rarest clusters
                              (bottom quartile by token share). High => the snapshot is
                              rich in rare regimes -- candidates for class-balancing.

The math is identical to `holdout_eval.py` (same frozen-assignment kernel); the only
difference is the reduction axis: holdout averages over a held-out *set* to test
generalisation, this averages *per file* over the whole dataset. The global mean of
`mean_residual` therefore reproduces the model's `final_obj_per_token` (~1888 for v6)
and `residual_frac` reproduces the in-sample 31.5% -- a built-in correctness check.

STREAMING (why this is not just load_tokens + assign): the full dataset is ~656 GB as
fp16 in RAM > 512 GB, so we load files in batches of `--batch-files`, assign each batch
on a single GPU, scatter the per-token results into [N_FILES]-wide accumulators, free
the buffer, and continue. I/O dominates (~17 files/s => ~13 min to read 1.3 TB); GPU
compute is negligible, so one GPU suffices and the broken inter-GPU P2P never enters.

Outputs (under <out>/):
  residual_map.npy   [N, 12288] float32 -- the per-CELL residual, i.e. the same quantity
                     as `mean_residual` but WITHOUT the reduction over cells. This is the
                     per-cell-per-timestep outlier score (goal: outliers for fine-tuning).
                     Same [T, 12288] NESTED grid as persistence/forecast error, so the maps
                     gather onto each other directly. Skip with --no-cell-maps.
  label_map.npy      [N, 12288] int16 -- assigned cluster per cell (the full-dataset
                     dynamic label map; assignments.pt only covers the sampled files).
  signatures.npz     canonical: mix[N,K], mean_residual/total/within/captured[N],
                     residual_frac[N], dominant[N], rare_exposure[N], n_tokens[N],
                     cell_mean_residual[12288], cell_std_residual[12288] (the per-cell
                     climatology that makes residual_map comparable across cells),
                     file_idx[N], datetime_hour[N] (numpy datetime64[h]).
  file_summary.csv   one row per file: id, idx, date, y/m/d/h, dom_cluster, dom_frac,
                     rare_exposure, mean_residual, residual_frac, mean_total.
  cluster_mix.csv    wide form: file_id, c000..c{K-1} (the mix matrix; load into pandas
                     / sklearn for k-center / diversity timestamp selection).
  manifest.json      provenance (model path, K, d, fingerprint, src, START, formulas).
  report.md          self-describing headline findings + how to use for goal 1.

Usage:
  # smoke-test correctness on the exact 200 held-out files (residual must match holdout.json):
  python3 src/analysis/file_signature.py --files-from runs/clustering/v6_subspace_big_d64/holdout/files.json \
      --out runs/signatures/_smoke
  # full run over all 13,021 files:
  python3 src/analysis/file_signature.py --out runs/signatures/v6_d64
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.cluster_io import DIM, N_CELLS, N_FILES_TOTAL, load_file_list

START = datetime(2014, 1, 1, 0, 0)          # ERA5 first step (reconstructed, UTC)
STEP_H = 6                                  # 6-hourly cadence (4/day)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="runs/clustering/v6_subspace_big_d64",
                   help="run directory with model.pt (the frozen model)")
    p.add_argument("--src", default=None, help="latent dir (default: the run's config src)")
    p.add_argument("--out", required=True, help="output directory for signatures/*")
    p.add_argument("--files-from", default=None,
                   help="json file-list or model.pt to assign (default: all 13,021 files)")
    p.add_argument("--limit", type=int, default=None,
                   help="assign only the first N files (default: all in --files-from / all)")
    p.add_argument("--batch-files", type=int, default=500,
                   help="files loaded per streaming batch (~25 GB fp16 each at 500)")
    p.add_argument("--chunk-size", type=int, default=262144, help="tokens per GPU chunk")
    p.add_argument("--load-workers", type=int, default=24)
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (single GPU)")
    p.add_argument("--no-cell-maps", action="store_true",
                   help="skip residual_map.npy / label_map.npy (the per-CELL outlier "
                        "arrays, ~640 MB + ~320 MB at 13,021 files)")
    return p.parse_args()


def load_batch(files, src, workers):
    """Load a batch of files in parallel -> (data fp16 [B*12288, DIM], fid int32, idx int64)."""
    B = len(files)
    tpf = N_CELLS
    data = torch.empty(B * tpf, DIM, dtype=torch.float16)
    fid = torch.empty(B * tpf, dtype=torch.int32)
    idxv = torch.empty(B, dtype=torch.int64)

    def one(pos, fidnum):
        d = torch.load(os.path.join(src, f"latent_{fidnum}.pt"),
                       map_location="cpu", weights_only=False)
        lat = d["latent"]
        if lat.dim() == 3:
            lat = lat.squeeze(0)
        if lat.shape != (N_CELLS, DIM):
            raise ValueError(f"latent_{fidnum}.pt has shape {tuple(lat.shape)}")
        o = pos * tpf
        data[o:o + tpf] = lat.to(torch.float16)
        fid[o:o + tpf] = fidnum
        idxv[pos] = int(d["idx"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, pos, f) for pos, f in enumerate(files)]
        for fut in as_completed(futs):
            fut.result()
    return data, fid, idxv


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{args.gpu}"
    t_start = time.time()

    # ---- frozen model ---------------------------------------------------------
    m = torch.load(os.path.join(args.dir, "model.pt"), map_location="cpu", weights_only=False)
    U, means, cnt = m["U"], m["means"], m["counts"].float()
    cfg = m["config"]
    K, D, d = U.shape
    affine = not cfg.get("linear", False)
    src = args.src or cfg.get("src", "latents_2")
    fp_model = m.get("sample_fingerprint", "?")
    final_obj = (m.get("final_obj_per_token") if m.get("final_obj_per_token") is not None
                 else (m["history"][-1]["obj_per_token"] if m.get("history") else float("nan")))
    print(f"Model: {args.dir}  K={K} d={d} affine={affine}  fp={fp_model}", flush=True)
    print(f"  model final_obj_per_token = {final_obj:.2f} (full-pass mean must match)", flush=True)

    # ---- file list to assign --------------------------------------------------
    if args.files_from:
        files = sorted(load_file_list(args.files_from))
    else:
        files = list(range(N_FILES_TOTAL))
    if args.limit:
        files = files[:args.limit]
    N = len(files)
    fmax = int(max(files)) if files else 0
    print(f"Assigning {N:,} files ({files[0]}..{files[-1] if files else '-'}); "
          f"max id {fmax}", flush=True)

    # ---- move model to GPU (identical prep to holdout_eval.py) ----------------
    U = U.to(device)
    U_cat = U.permute(1, 0, 2).reshape(D, K * d).contiguous()      # [D, K*d]
    w = cnt / cnt.sum()
    mu_g = (w[:, None] * means).sum(0).to(device)                 # frozen global mean [D]
    if affine:
        mu = means.to(device)                                     # [K, D]
        c = torch.bmm(U.transpose(1, 2), mu.unsqueeze(-1)).squeeze(-1)   # [K, d]
        mnorm = (mu * mu).sum(1)
        cnorm = (c * c).sum(1)

    # ---- per-FILE accumulators (on GPU; small: N scalars + N*K counts) --------
    W = fmax + 1                                                   # width = max file id + 1
    counts = torch.zeros(W, K, dtype=torch.float64, device=device)
    resid_sum = torch.zeros(W, dtype=torch.float64, device=device)
    within_sum = torch.zeros(W, dtype=torch.float64, device=device)
    cap_sum = torch.zeros(W, dtype=torch.float64, device=device)
    total_sum = torch.zeros(W, dtype=torch.float64, device=device)
    stored_idx = np.full(W, -1, dtype=np.int64)

    # ---- per-CELL outlier maps (the per-token residual, kept instead of reduced) --
    # The assignment kernel below already computes a residual for every one of the
    # 12288 cells; the per-file accumulators throw that resolution away. Keeping it
    # costs one extra write per batch and turns outlier detection from per-timestep
    # into per-cell-per-timestep. Rows follow `files` order (row i <-> file_ids[i],
    # the same order as signatures.npz / file_summary.csv), NOT file id, so a
    # --limit/--files-from subset stays dense. Written as .npy memmaps so the full
    # 13,021 x 12,288 arrays never have to sit in RAM, and so downstream analysis can
    # np.load(..., mmap_mode="r") a single row/column without reading 640 MB.
    cell_maps = not args.no_cell_maps
    os.makedirs(args.out, exist_ok=True)
    if cell_maps:
        resid_map = np.lib.format.open_memmap(
            os.path.join(args.out, "residual_map.npy"), mode="w+",
            dtype=np.float32, shape=(N, N_CELLS))
        label_map = np.lib.format.open_memmap(
            os.path.join(args.out, "label_map.npy"), mode="w+",
            dtype=np.int16, shape=(N, N_CELLS))
        # staging buffers, allocated once at max batch size and sliced to [:M]
        buf_resid = torch.empty(args.batch_files * N_CELLS, dtype=torch.float32, device=device)
        buf_lab = torch.empty(args.batch_files * N_CELLS, dtype=torch.int16, device=device)
        cell_sum = torch.zeros(N_CELLS, dtype=torch.float64, device=device)
        cell_sqsum = torch.zeros(N_CELLS, dtype=torch.float64, device=device)
        print(f"  writing per-cell maps: residual_map.npy [{N},{N_CELLS}] float32 "
              f"({N * N_CELLS * 4 / 2**30:.2f} GiB) + label_map.npy int16", flush=True)

    # ---- streaming assignment -------------------------------------------------
    nb = (N + args.batch_files - 1) // args.batch_files
    for bi, i0 in enumerate(range(0, N, args.batch_files)):
        batch = files[i0:i0 + args.batch_files]
        data, fid, idxv = load_batch(batch, src, args.load_workers)
        for fpos, fidnum in enumerate(batch):
            stored_idx[fidnum] = int(idxv[fpos])
        M = data.shape[0]
        for c0 in range(0, M, args.chunk_size):
            X = data[c0:c0 + args.chunk_size].to(device, non_blocking=True).float()
            fc = fid[c0:c0 + args.chunk_size].to(device).long()    # [b] file id per token
            xnorm = (X * X).sum(1, keepdim=True)                   # [b,1]
            P = (X @ U_cat).view(X.shape[0], K, d)
            pe = (P * P).sum(-1)                                   # [b,K]  ||U_j^T x||^2
            if affine:
                xm = X @ mu.T                                      # [b,K]
                pc = torch.einsum("bkd,kd->bk", P, c)
                dist2 = xnorm - 2 * xm + mnorm                    # [b,K]  ||x-mu_j||^2
                proj = pe - 2 * pc + cnorm                        # [b,K]  ||U_j^T(x-mu_j)||^2
            else:
                dist2 = xnorm - 2 * (X @ means.to(device).T) + (means.to(device) ** 2).sum(1)
                proj = pe
            R = dist2 - proj                                       # [b,K]  residual to each subspace
            vals, a = R.min(1)                                     # assign by min residual
            if cell_maps:
                # keep the per-token result before it is reduced away; c0 indexes the
                # batch's token array, which is laid out file-major then cell-ordered
                # (load_batch writes data[pos*12288 : (pos+1)*12288] = latent rows), so
                # a [len(batch), 12288] view is exactly [file, HEALPix cell].
                buf_resid[c0:c0 + X.shape[0]] = vals.clamp_min(0).float()
                buf_lab[c0:c0 + X.shape[0]] = a.to(torch.int16)
            ad = dist2.gather(1, a.unsqueeze(1)).squeeze(1)       # within (= dist to assigned mu)
            ap = proj.gather(1, a.unsqueeze(1)).squeeze(1)        # captured by assigned subspace
            tot = ((X - mu_g) ** 2).sum(1)                        # total to frozen global mean
            ones = torch.ones(a.shape[0], device=device, dtype=torch.float64)
            counts.view(-1).index_add_(0, fc * K + a, ones)
            resid_sum.index_add_(0, fc, vals.clamp_min(0).double())
            within_sum.index_add_(0, fc, ad.clamp_min(0).double())
            cap_sum.index_add_(0, fc, ap.clamp_min(0).double())
            total_sum.index_add_(0, fc, tot.double())
        if cell_maps:
            nbf = len(batch)
            rm = buf_resid[:M].view(nbf, N_CELLS)
            cell_sum += rm.sum(0).double()                          # per-cell climatology
            cell_sqsum += (rm.double() ** 2).sum(0)
            resid_map[i0:i0 + nbf] = rm.cpu().numpy()
            label_map[i0:i0 + nbf] = buf_lab[:M].view(nbf, N_CELLS).cpu().numpy()
        del data, fid
        el = time.time() - t_start
        done = i0 + len(batch)
        print(f"  batch {bi + 1}/{nb}: {done}/{N} files "
              f"({done / max(el, 1):.1f} files/s, {el / 60:.1f} min, "
              f"ETA {(N - done) / max(done / max(el, 1), 1e-9) / 60:.1f} min)", flush=True)

    # ---- reduce to per-file quantities ----------------------------------------
    sel = torch.tensor(files, device=device, dtype=torch.long)
    n_tok = counts[sel].sum(1)                                    # [N] (==12288 each)
    mix = (counts[sel] / n_tok.clamp(min=1).unsqueeze(1)).cpu().numpy()
    mean_resid = (resid_sum[sel] / n_tok.clamp(min=1)).cpu().numpy()
    mean_within = (within_sum[sel] / n_tok.clamp(min=1)).cpu().numpy()
    mean_cap = (cap_sum[sel] / n_tok.clamp(min=1)).cpu().numpy()
    mean_total = (total_sum[sel] / n_tok.clamp(min=1)).cpu().numpy()
    resid_frac = mean_resid / np.clip(mean_total, 1e-12, None)
    dominant = mix.argmax(1)
    dom_frac = mix.max(1)
    n_tok = n_tok.cpu().numpy()
    idx_arr = stored_idx[sel.cpu().numpy()]

    # rare-regime exposure: share of a file's tokens in the globally-rarest
    # clusters (bottom quartile by token share).
    gfreq = mix.mean(0)                                           # [K] global share
    rare_thresh = float(np.quantile(gfreq, 0.25))
    rare_mask = gfreq <= rare_thresh
    rare_exposure = mix[:, rare_mask].sum(1)

    # ---- reconstructed calendar date (START + idx*6h) -------------------------
    base = np.datetime64(START).astype("datetime64[ms]")
    delta = np.timedelta64(STEP_H * 3600 * 1000, "ms")
    dt = base + idx_arr.astype(np.int64) * delta
    dt_h = dt.astype("datetime64[h]")                            # '2014-01-01T00'
    years = dt.astype("datetime64[Y]").astype(int) + 1970
    months = (dt.astype("datetime64[M]").astype(int) % 12) + 1
    days = (dt.astype("datetime64[D]") - dt.astype("datetime64[M]")).astype(int) + 1
    hours = (dt.astype("datetime64[h]") - dt.astype("datetime64[D]")).astype(int)

    # ============================= write outputs ===============================
    # Per-cell climatology: the raw residual is not comparable across cells (a
    # storm-track cell is intrinsically harder than a subtropical one), so the useful
    # outlier score is the anomaly against each cell's OWN temporal norm:
    #     z[t, cell] = (residual_map[t, cell] - cell_mean[cell]) / cell_std[cell]
    # These two vectors are what makes that normalisation possible without a second
    # pass over the 640 MB map.
    cell_extra = {}
    if cell_maps:
        resid_map.flush()
        label_map.flush()
        cell_mean = (cell_sum / max(N, 1)).cpu().numpy()
        cell_std = np.sqrt(np.clip((cell_sqsum / max(N, 1)).cpu().numpy() - cell_mean ** 2,
                                   0.0, None))
        cell_extra = {"cell_mean_residual": cell_mean.astype(np.float32),
                      "cell_std_residual": cell_std.astype(np.float32)}
    np.savez(os.path.join(args.out, "signatures.npz"), **cell_extra,
             mix=mix.astype(np.float32), mean_residual=mean_resid.astype(np.float32),
             mean_total=mean_total.astype(np.float32), mean_within=mean_within.astype(np.float32),
             mean_captured=mean_cap.astype(np.float32), residual_frac=resid_frac.astype(np.float32),
             dominant=dominant.astype(np.int32), dom_frac=dom_frac.astype(np.float32),
             rare_exposure=rare_exposure.astype(np.float32), n_tokens=n_tok.astype(np.int32),
             file_idx=idx_arr.astype(np.int32), datetime_hour=dt_h,
             rare_clusters=np.where(rare_mask)[0].astype(np.int32),
             global_freq=gfreq.astype(np.float32))

    file_ids = np.asarray(files)
    with open(os.path.join(args.out, "file_summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file_id", "idx", "datetime_utc", "year", "month", "day", "hour_utc",
                     "dom_cluster", "dom_frac", "rare_exposure",
                     "mean_residual", "residual_frac", "mean_total"])
        for i in range(N):
            wr.writerow([int(file_ids[i]), int(idx_arr[i]), str(dt_h[i]),
                         int(years[i]), int(months[i]), int(days[i]), int(hours[i]),
                         int(dominant[i]), f"{dom_frac[i]:.4f}", f"{rare_exposure[i]:.4f}",
                         f"{mean_resid[i]:.2f}", f"{resid_frac[i]:.4f}",
                         f"{mean_total[i]:.2f}"])

    np.savetxt(os.path.join(args.out, "cluster_mix.csv"),
               np.column_stack([file_ids, mix]), delimiter=",",
               header="file_id," + ",".join(f"c{k:03d}" for k in range(K)),
               comments="", fmt=["%d"] + ["%.6f"] * K)

    manifest = {
        "model_dir": os.path.abspath(args.dir),
        "model_fingerprint": fp_model, "K": int(K), "d": int(d), "affine": bool(affine),
        "src": src, "n_files": int(N), "dim": DIM, "n_cells": N_CELLS,
        "model_final_obj_per_token": float(final_obj),
        "fullpass_mean_residual": float(mean_resid.mean()),
        "fullpass_mean_residual_frac": float(resid_frac.mean()),
        "datetime_convention": "datetime_utc = 2014-01-01 00:00 UTC + idx * 6h (idx == file id)",
        "rare_definition": "rare_exposure = sum of mix over clusters whose global token share "
                           "(mix.mean(0)) is in the bottom quartile (<= %.4f)" % rare_thresh,
        "rare_clusters": np.where(rare_mask)[0].astype(int).tolist(),
        "formulas": {
            "residual_j(x)": "||x-mu_j||^2 - ||U_j^T(x-mu_j)||^2  (affine)",
            "mean_residual": "mean over the file's 12288 tokens of residual_(assigned)(x)",
            "residual_frac": "mean_residual / mean_total  (total = E||x-mu_global||^2, "
                             "mu_global frozen from the trained cluster means)",
            "mix_k": "fraction of the file's 12288 cells assigned to cluster k"},
        "cell_maps": ({
            "residual_map.npy": f"[{N}, {N_CELLS}] float32 -- residual_(assigned)(x) for "
                                "every HEALPix cell of every file (NESTED ordering); row i "
                                "is file_ids[i], NOT file id i",
            "label_map.npy": f"[{N}, {N_CELLS}] int16 -- assigned cluster per cell, same "
                             "row convention",
            "row_axis": "rows follow the sorted file list, identical to signatures.npz / "
                        "file_summary.csv row order (file_ids)",
            "normalisation": "z[t,cell] = (residual_map[t,cell] - cell_mean_residual[cell]) "
                             "/ cell_std_residual[cell]  -- raw residuals are not comparable "
                             "across cells",
            "aligns_with": "same [T, 12288] NESTED grid as runs/persistence/*/err_persist.npy "
                           "(one row shorter there: transitions, not states)",
        } if cell_maps else None),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_min": round((time.time() - t_start) / 60, 1),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ============================= report ======================================
    gmean_resid = float(mean_resid.mean())
    gmean_rf = float(resid_frac.mean())
    # monthly mean residual-fraction (anomaly seasonality)
    mon_rf = np.array([resid_frac[months == mm].mean() if (months == mm).any() else np.nan
                       for mm in range(1, 13)])
    top_anom = np.argsort(resid_frac)[::-1][:12]
    top_rare = np.argsort(rare_exposure)[::-1][:12]
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mon_rf_val = [0.0 if np.isnan(mon_rf[i]) else float(mon_rf[i]) for i in range(12)]
    mon_rf_str = ", ".join(f"{MONTHS[i]} {mon_rf_val[i]:.1%}" for i in range(12))

    L = []
    add = L.append
    add(f"# Per-file regime signature + residual — `{args.out}`")
    add(f"\n*Generated {time.strftime('%Y-%m-%d %H:%M')} by `file_signature.py`. "
        f"Frozen model `{args.dir}` (K={K}, d={d}, affine={affine}, fingerprint `{fp_model}`). "
        f"{N:,} files × {N_CELLS} cells = {N * N_CELLS:,} tokens assigned on a single GPU.*")
    add("\n## What this is\n")
    add("For every latent timestep the frozen v6 model assigns each of its 12,288 cells to the "
        "nearest affine subspace (`residual_j = ‖x−μ_j‖²−‖Uⱼᵀ(x−μ_j)‖²`) and reduces the result "
        "to a compact per-file summary. This is the coordinate system for **goal 1 (which "
        "timestamps to train on)**: `mix` is a low-dim fingerprint of each snapshot, "
        "`residual_frac` flags the anomalous / hard-to-encode ones, and `rare_exposure` flags "
        "snapshots rich in rare regimes.\n")
    add("## Correctness check\n")
    add(f"The count-weighted mean of `mean_residual` over all {N:,} files is "
        f"**{gmean_resid:.2f}** (model `final_obj_per_token` = {final_obj:.2f}) and the mean "
        f"`residual_frac` is **{gmean_rf:.1%}** (in-sample / held-out = 31.5% in the run report). "
        "A full-dataset pass that reproduces the trained objective confirms the assignment kernel "
        f"is correct.{'  ✔ matches' if abs(gmean_resid - final_obj) / final_obj < 0.02 else '  ✗ MISMATCH'}\n")
    add("## Per-file residual fraction (anomaly / hardness)\n")
    add("`residual_frac` = the share of a snapshot's variance the regime+subspace model leaves "
        "unexplained (globally ~31.5%). High outliers are the **anomalous timestamps** — "
        "candidates for hard-example upweighting.\n")
    qs = np.quantile(resid_frac, [0, .01, .25, .5, .75, .99, 1])
    add(f"- distribution (min/p1/p25/median/p75/p99/max): "
        + " / ".join(f"{q:.1%}" for q in qs))
    add("- monthly mean `residual_frac` (does anomaly concentrate seasonally?): " + mon_rf_str)
    add("\n| rank | file | datetime | resid_frac | rare_exp | dom_cluster |\n|---|---|---|---|---|---|")
    for r, i in enumerate(top_anom, 1):
        add(f"| {r} | {int(file_ids[i])} | {dt_h[i]} | {resid_frac[i]:.1%} | "
            f"{rare_exposure[i]:.1%} | {int(dominant[i])} |")
    add("\n## Rare-regime exposure\n")
    add(f"`rare_exposure` = share of a snapshot's tokens in the {int(rare_mask.sum())} globally-"
        f"rarest clusters (bottom quartile, share ≤ {rare_thresh:.3%}: "
        + ", ".join(str(int(c)) for c in np.where(rare_mask)[0]) + "). "
        "High values mark snapshots rich in rare regimes — candidates for class-balancing.\n")
    add("| rank | file | datetime | rare_exposure | resid_frac | dom_cluster |\n|---|---|---|---|---|---|")
    for r, i in enumerate(top_rare, 1):
        add(f"| {r} | {int(file_ids[i])} | {dt_h[i]} | {rare_exposure[i]:.1%} | "
            f"{resid_frac[i]:.1%} | {int(dominant[i])} |")
    if cell_maps:
        add("\n## Per-cell outlier map\n")
        add("*How to read this: `residual_frac` above scores a whole snapshot. "
            "`residual_map.npy` `[N, 12288]` keeps the **same residual per HEALPix cell**, so an "
            "outlier can be localised to where on the globe it happened instead of only when. "
            "It is the identical quantity — averaging a row reproduces that row's "
            "`mean_residual` exactly — just not reduced over cells.*\n")
        add("Raw residuals are **not comparable between cells**: an intrinsically hard cell "
            "(storm track) outscores an easy one (subtropical ocean) in every snapshot. Score "
            "outliers against each cell's own temporal norm, using the climatology vectors "
            "stored in `signatures.npz`:\n")
        add("```python\n"
            "import numpy as np\n"
            f"R = np.load('{args.out}/residual_map.npy', mmap_mode='r')   # [N, 12288]\n"
            f"s = np.load('{args.out}/signatures.npz')\n"
            "z = (R - s['cell_mean_residual']) / np.maximum(s['cell_std_residual'], 1e-6)\n"
            "# z[t, cell] > 4  ->  this cell is far outside its own normal range at step t\n"
            "```\n")
        cm, cs = cell_extra["cell_mean_residual"], cell_extra["cell_std_residual"]
        add(f"- per-cell mean residual across the {N:,} steps: min **{cm.min():.0f}** / "
            f"median **{np.median(cm):.0f}** / max **{cm.max():.0f}** "
            f"({cm.max() / max(cm.min(), 1e-9):.1f}× spread — this is exactly why the "
            "normalisation is needed).")
        add(f"- per-cell temporal std: median **{np.median(cs):.0f}** "
            f"(median std/mean = {np.median(cs / np.maximum(cm, 1e-9)):.1%}).")
        add(f"- hardest cells (highest mean residual, NESTED ids): "
            + ", ".join(str(int(i)) for i in np.argsort(cm)[::-1][:10]) + ".")
        add(f"- easiest cells: " + ", ".join(str(int(i)) for i in np.argsort(cm)[:10]) + ".\n")
        add("`label_map.npy` `[N, 12288]` int16 carries the assigned cluster per cell over the "
            "**whole dataset** — the dynamic counterpart to `assignments.pt`, which only covers "
            "the sampled files. Both arrays sit on the same NESTED `[T, 12288]` grid as "
            "`err_persist.npy`, so they gather onto each other with no remapping: intersect them "
            "to separate *anomalous because rapidly changing* (high persistence error) from "
            "*anomalous because unmodelled* (high residual, low persistence error).\n")

    add("\n## How to use for goal 1 (timestamp selection)\n")
    add("- **Diversity / coverage sampling** (anti-redundancy): `cluster_mix.csv` is the feature "
        "matrix; run k-center / farthest-point in mix-space to pick a minimal covering subset — "
        "consecutive timesteps are near-duplicates in mix (clusters are geographic & time-stable).")
    add("- **Seasonal stratification**: the `month` column + the seasonal clusters in "
        "`temporal_report.md` give per-month upweighting (shoulder months Apr→May, Oct→Nov).")
    add("- **Rare-regime balancing**: upweight files with high `rare_exposure`.")
    add("- **Hard-example upweighting**: upweight high-`residual_frac` rows.")
    add("- **The closed loop (most principled)**: re-run this script with the model's *forecast "
        "errors* as input instead of embeddings — the resulting signatures flag the regimes where "
        "the model fails, i.e. the timestamps that most deserve training weight.")
    add("\n## Files\n")
    add("- `signatures.npz` — canonical arrays (mix[N,K], all per-file stats, datetime).")
    add("- `file_summary.csv` — one row per file (human-readable, sorted by file_id).")
    add("- `cluster_mix.csv` — wide mix matrix for pandas / sklearn timestamp selection.")
    add("- `manifest.json` — provenance, formulas, rare-cluster list.")
    if cell_maps:
        add(f"- `residual_map.npy` — `[{N}, {N_CELLS}]` float32 per-cell residual "
            f"({N * N_CELLS * 4 / 2**30:.2f} GiB; load with `mmap_mode='r'`).")
        add(f"- `label_map.npy` — `[{N}, {N_CELLS}]` int16 per-cell assigned cluster.")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(L))

    # ============================= console summary =============================
    print("\n================ Per-file signatures done ================", flush=True)
    print(f"  files assigned            : {N:,}")
    print(f"  mean mean_residual        : {gmean_resid:.2f}  (model obj {final_obj:.2f})")
    print(f"  mean residual_frac        : {gmean_rf:.1%}  (report in/held-out 31.5%)")
    print(f"  residual_frac range       : {resid_frac.min():.1%} .. {resid_frac.max():.1%}")
    print(f"  rare clusters (bottom ¼)  : {int(rare_mask.sum())} of {K}")
    ta = int(top_anom[0])
    print(f"  most anomalous file       : {int(file_ids[ta])}  dt={dt_h[ta]}  resid_frac={resid_frac[ta]:.1%}")
    print(f"  wall time                 : {(time.time() - t_start) / 60:.1f} min")
    print(f"  wrote                     : {args.out}/{{signatures.npz,file_summary.csv,"
          f"cluster_mix.csv,manifest.json,report.md}}")


if __name__ == "__main__":
    main()
