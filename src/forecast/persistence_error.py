#!/usr/bin/env python3
"""
persistence_error.py — per-CELL latent PERSISTENCE error over latents_2 (model-free, local).

The persistence baseline: how far does "predict t+1 = t" land from the truth, per cell.
This is the skill-score denominator for the forecast error (extract_forecast_error.py) and is
computable on the analysis box RIGHT NOW with no checkpoint -- the latents are already on disk.

  For each source t (0 .. N-2), with x_t = latents_2[t][0] ([12288,2048]):
      err_persist[t, cell] = ||x_t[cell] - x_{t+1}[cell]||^2     # [N,12288] float32

Same output contract as extract_forecast_error.py's err_persist.npy, so analyze_forecast_error.py
consumes either. When the forecast err_forecast.npy arrives on this box, the latent skill score
      skill[t,cell] = 1 - err_forecast[t,cell] / err_persist[t,cell]
falls out for free (persistence baseline is exactly what the WGen repo lacks; wgen_architecture.md
decision f). Persistence error itself is meaningful: it measures how *changeable* each cell /
cluster / timestamp is between 6h steps (dynamic/stormy vs quiescent).

Reads the LOCAL copy (key `idx`). The supercomputer copy uses key `idx_num`; --key overrides.

USAGE
  python3 src/forecast/persistence_error.py --out runs/persistence/v6            # full run, ~10-20 min
  python3 src/forecast/persistence_error.py --limit 200 --out runs/persistence/_smoke   # quick sanity check

OUTPUTS  <out>/
  err_persist.npy  [N,12288] float32   (N = n_files-1)
  meta.json        {source_idx, target_idx, source_dt, norm, ...}
"""

import argparse
import json
import os
import sys
import time
from queue import Queue
from threading import Thread

import numpy as np
import torch

BASE_DT = np.datetime64("2014-01-01T00:00").astype("datetime64[ms]")
SIXH = np.timedelta64(6 * 3600 * 1000, "ms")
N_CELLS, DIM = 12288, 2048


def idx_to_dt(idx):
    return BASE_DT + int(idx) * SIXH


def list_latent_files(latents_dir):
    out = []
    for fn in os.listdir(latents_dir):
        if fn.startswith("latent_") and fn.endswith(".pt"):
            try:
                out.append((int(fn[len("latent_"):-len(".pt")]), os.path.join(latents_dir, fn)))
            except ValueError:
                pass
    out.sort(key=lambda x: x[0])
    return out


def load_latent(path, key):
    d = torch.load(path, map_location="cpu", weights_only=False)
    idx = int(d[key]) if key in d else int(d.get("idx_num", d.get("idx")))
    x = d["latent"]
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    assert x.shape == (N_CELLS, DIM), f"unexpected shape {tuple(x.shape)} in {path}"
    return x.float().contiguous(), idx


class PrefetchReader:
    def __init__(self, files, key, depth=8):
        self.files, self.key = files, key
        self.q = Queue(maxsize=depth)
        Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for _idx, path in self.files:
                self.q.put(load_latent(path, self.key))
        except Exception as e:  # noqa: BLE001
            self.q.put(e)
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents-dir", default="latents_2")
    ap.add_argument("--out", default="runs/persistence/v6")
    ap.add_argument("--key", default="idx", help="dict key for the index (local=idx, supercomputer=idx_num)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = list_latent_files(args.latents_dir)
    if args.limit:
        files = files[: args.limit]
    n_files = len(files)
    if n_files < 2:
        raise SystemExit("need >= 2 files")
    n_src = n_files - 1
    print(f"[io] {n_files} files -> {n_src} transitions (key={args.key})", flush=True)

    err_persist = np.zeros((n_src, N_CELLS), dtype=np.float32)
    src_idx = np.zeros(n_src, dtype=np.int64)
    tgt_idx = np.zeros(n_src, dtype=np.int64)

    it = iter(PrefetchReader(files, args.key))
    prev_x, prev_i = next(it)
    t = 0
    t0 = time.time()
    running_sum = 0.0

    while t < n_src:
        got = next(it, None)
        if got is None:
            break
        cur_x, cur_i = got
        diff = (prev_x - cur_x).pow(2).sum(dim=-1).numpy()    # [12288]
        err_persist[t] = diff
        src_idx[t] = prev_i
        tgt_idx[t] = cur_i
        running_sum += float(diff.mean())
        prev_x, prev_i = cur_x, cur_i
        t += 1
        if t % 500 == 0:
            el = time.time() - t0
            print(f"  [{t}/{n_src}] {t/max(el,1):.1f} trans/s  running mean={running_sum/t:.1f}",
                  flush=True)

    el = time.time() - t0
    mean_err = float(err_persist.mean())
    print(f"[done] {t} transitions in {el/60:.1f} min; mean per-cell persistence err={mean_err:.3f}",
          flush=True)

    np.save(os.path.join(args.out, "err_persist.npy"), err_persist)
    meta = {
        "type": "persistence", "n_cells": N_CELLS, "dim": DIM, "n_transitions": int(t),
        "source_idx": src_idx.tolist(), "target_idx": tgt_idx.tolist(),
        "source_dt": [str(idx_to_dt(i)) for i in src_idx],
        "norm": "per-cell squared L2 over the 2048 latent dim",
        "err_persist": "||x_t - x_{t+1}||^2",
        "mean": mean_err,
        "latents_dir": os.path.abspath(args.latents_dir),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] {args.out}/{{err_persist.npy,meta.json}}", flush=True)


if __name__ == "__main__":
    main()
