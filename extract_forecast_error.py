#!/usr/bin/env python3
"""
extract_forecast_error.py — per-CELL latent forecast error over the latents_2 stream.

Runs ON THE SUPERCOMPUTER, inside the WeatherGenerator environment, where the trained
checkpoint + production Config + dataset live. NOT runnable on the analysis box (no
checkpoint there). See /home/psaher/latents/wgen_architecture.md (§4) for the full plan.

WHAT IT COMPUTES (single lead, t -> t+1)
  For each source timestep t (0 .. N-2), with x_t = latents_2[t][0]  ([12288,2048]):
      pred   = forecast_engine(x_t, step=0, rope_coords)        # model's latent t->t+1
      truth  = latents_2[t+1][0]
      err_forecast[t, cell] = ||pred[cell] - truth[cell]||^2    # [N,12288] float32
      err_persist [t, cell] = ||x_t[cell] - truth[cell]||^2     # persistence baseline (free)

  forecast_engine is latent->latent (engines.py:623-636; model.py:703 "roll-out in latent
  space"). The only op between the encoder output (= latents_2) and forecast_engine is a
  single-input-step sum (model.py:689-691, identity when num_input_steps=1), so feeding
  latents_2[t] straight in IS the model's own forward path. The engine is time-homogeneous
  (`fstep` unused inside ForecastingEngine.forward), so step=0 is the t->t+1 map.

THE SELF-CHECK (answers the blocking open question, wgen_architecture.md §6.1)
  A real forecaster must beat persistence. On the first batch the script prints
      mean_forecast_err, mean_persist_err, skill = 1 - mean_forecast/mean_persist
  skill>0 => latents_2[t] is the correct step-0 forecast input. skill<=0 => wrong
  input/rope_coords/config => ABORT before the full sweep. No ERA5 needed. --selfcheck-only
  runs just this on one batch.

JOIN TO CLUSTERS
  err_*[t,cell] uses the SAME NESTED HEALPix cell index as assignments.pt (both come from
  latents_2). The join on the analysis box is a direct gather -- no remap. See
  analyze_forecast_error.py.

USAGE (supercomputer)
  # 1) cheap verification (loads model, one batch, prints skill vs persistence):
  python3 extract_forecast_error.py --config <prod.yml> --run-id <id> --selfcheck-only \\
      --latents-dir /p/scratch/weatherai/slurm/slurm_weathergen_atmosfo2_copy_dir/WeatherGenerator/latents_2
  # 2) full sweep (detached):
  setsid nohup python3 extract_forecast_error.py --config <prod.yml> --run-id <id> \\
      --out err_forecast/run0 > err_forecast.log 2>&1 &

OUTPUTS  <out>/
  err_forecast.npy  [N,12288] float32    (N = n_files-1)
  err_persist.npy   [N,12288] float32    (same shape; the persistence baseline)
  meta.json         {run_id, config, source_idx, target_idx, source_dt, ...}

TWO THINGS TO ADAPT ON THE SUPERCOMPUTER (marked TODO below):
  (1) build_dataset()  -- construct the production Anemoi dataset exactly as the trainer does.
  (2) the Config loader -- use the WGen config system / merge used for the production run.
Multi-lead rollout (t -> t+k) is intentionally not implemented; extend the reader to a
lookahead window of k if needed.
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

# ---- WGen imports (available on the supercomputer inside the WGen env) --------------
# These will NOT import on the analysis box; that is expected -- this script runs remotely.
from weathergen.common.config import Config  # noqa: E402
from weathergen.model.model_interface import init_model_and_shard  # noqa: E402

BASE_DT = np.datetime64("2014-01-01T00:00").astype("datetime64[ms]")
SIXH = np.timedelta64(6 * 3600 * 1000, "ms")
N_CELLS, DIM = 12288, 2048


def idx_to_dt(idx):
    return BASE_DT + int(idx) * SIXH


def list_latent_files(latents_dir):
    out = []
    for fn in os.listdir(latents_dir):
        if not (fn.startswith("latent_") and fn.endswith(".pt")):
            continue
        try:
            idx = int(fn[len("latent_"):-len(".pt")])
        except ValueError:
            continue
        out.append((idx, os.path.join(latents_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def load_latent(path, key="idx_num"):
    """Return ([12288,2048] cpu fp32, idx). Accepts idx_num (supercomputer) or idx (local copy)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "idx_num" in d:
        idx = int(d["idx_num"])
    elif "idx" in d:
        idx = int(d["idx"])
    else:
        raise KeyError(f"no idx/idx_num key in {path}: {list(d.keys())}")
    x = d["latent"]
    if x.dim() == 3 and x.shape[0] == 1:          # squeeze leading batch dim
        x = x[0]
    assert x.shape == (N_CELLS, DIM), f"unexpected latent shape {tuple(x.shape)} in {path}"
    return x.float().contiguous(), idx


class PrefetchReader:
    """Load each file exactly once into a queue so I/O overlaps with the GPU forward."""

    def __init__(self, files, depth=6):
        self.files = files
        self.q = Queue(maxsize=depth)
        self.t = Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            for _idx, path in self.files:
                self.q.put(load_latent(path))
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


# ---------------------------------------------------------------- model loading -----
def build_dataset(cf):
    """Construct the production Anemoi dataset exactly as the WGen trainer does for this run.

    init_model_and_shard -> get_model needs sources_size / targets_num_channels /
    targets_coords_size, which come from the dataset. TODO(confirm): match the run that
    produced latents_2. Mirror train/trainer.py's dataset construction.
    """
    raise NotImplementedError(
        "Construct the production Anemoi dataset here as the WGen trainer does for this run; "
        "it must expose get_sources_size()/get_targets_num_channels()/get_targets_coords_size()."
    )


def load_forecast_model(config_path, run_id, device):
    """Build model from config, load checkpoint, return (forecast_engine, rope_coords)."""
    cf = Config(config_path)  # TODO(confirm): use the WGen loader/merge used in training
    cf["local_rank"] = 0
    dataset = build_dataset(cf)
    training_mode = cf.get("training_mode", "forecasting")
    model, model_params = init_model_and_shard(
        cf, dataset,
        run_id_contd=run_id, mini_epoch_contd=-1,
        training_mode=training_mode, device=device,
        with_ddp=False, with_fsdp=False,
    )
    model.eval()
    fe = model.forecast_engine
    from weathergen.model.engines import IdentityEngine
    if isinstance(fe, IdentityEngine):
        raise RuntimeError(
            "model.forecast_engine is IdentityEngine -- config has no forecast engine. "
            "Use the forecasting config / a checkpoint trained with policy='forecast'."
        )
    rope_coords = model_params.rope_coords          # None unless cf.rope_2D
    if rope_coords is not None:
        rope_coords = rope_coords.to(device)
    return fe, rope_coords


@torch.no_grad()
def forecast_step(fe, x, rope_coords, device):
    """x: [B,12288,2048] cpu/fp32 -> predicted-next latent [B,12288,2048] cpu/fp32."""
    xt = x.to(device, non_blocking=True).float()
    out = fe(xt, 0, rope_coords)        # fstep unused inside (time-homogeneous)
    return out.float().cpu()


def per_cell_sq_residual(pred, truth):
    """pred,truth: [B,12288,2048] -> [B,12288] float32 (||.||^2 over the 2048 dim)."""
    return (pred - truth).float().pow(2).sum(dim=-1)


# ---------------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="production WGen config .yml")
    ap.add_argument("--run-id", required=True, help="trained model run_id (checkpoint to load)")
    ap.add_argument("--latents-dir", default="/p/scratch/weatherai/slurm/slurm_weathergen_atmosfo2_copy_dir/WeatherGenerator/latents_2")
    ap.add_argument("--out", default="err_forecast/run0")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap source files (0=all)")
    ap.add_argument("--batch", type=int, default=4, help="samples per GPU forward")
    ap.add_argument("--selfcheck-only", action="store_true", help="one batch, print skill, exit")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] config={args.config} run_id={args.run_id} device={device}", flush=True)
    fe, rope_coords = load_forecast_model(args.config, args.run_id, device)
    nparams = sum(p.numel() for p in fe.parameters())
    print(f"[load] forecast_engine: {nparams/1e6:.1f}M params, "
          f"rope_coords={'set' if rope_coords is not None else 'None (rope_2D=False)'}", flush=True)

    files = list_latent_files(args.latents_dir)
    if args.limit:
        files = files[: args.limit]
    n_files = len(files)
    if n_files < 2:
        raise SystemExit("need >= 2 files to form a transition")
    n_src = n_files - 1
    print(f"[io] {n_files} files -> {n_src} source transitions", flush=True)

    err_forecast = np.zeros((n_src, N_CELLS), dtype=np.float32)
    err_persist = np.zeros((n_src, N_CELLS), dtype=np.float32)
    src_idx = np.zeros(n_src, dtype=np.int64)
    tgt_idx = np.zeros(n_src, dtype=np.int64)

    it = iter(PrefetchReader(files))
    prev_x, prev_i = next(it)                # file 0
    t = 0                                     # completed source transitions
    bp, bn, bsi, bti = [], [], [], []         # current batch buffers

    def flush():
        nonlocal t
        B = len(bp)
        if B == 0:
            return
        sx = torch.stack(bp, 0); nx = torch.stack(bn, 0)
        err_ps = per_cell_sq_residual(sx, nx).numpy()
        pred = forecast_step(fe, sx, rope_coords, device)
        err_fc = per_cell_sq_residual(pred, nx).numpy()
        err_forecast[t:t+B] = err_fc
        err_persist[t:t+B] = err_ps
        src_idx[t:t+B] = np.fromiter(bsi, dtype=np.int64)
        tgt_idx[t:t+B] = np.fromiter(bti, dtype=np.int64)
        bp.clear(); bn.clear(); bsi.clear(); bti.clear()
        return err_fc, err_ps, B

    selfcheck_done = False
    t0 = time.time()

    while t < n_src:
        got = next(it, None)
        if got is None:
            break
        cur_x, cur_i = got
        bp.append(prev_x); bn.append(cur_x); bsi.append(prev_i); bti.append(cur_i)
        prev_x, prev_i = cur_x, cur_i
        if len(bp) >= args.batch:
            err_fc, err_ps, B = flush()
            t += B
            if not selfcheck_done:
                mf, mp = float(err_fc.mean()), float(err_ps.mean())
                skill = 1.0 - mf / mp if mp > 0 else float("nan")
                print("=" * 64, flush=True)
                print(f"[SELF-CHECK] mean_forecast={mf:.3f}  mean_persist={mp:.3f}", flush=True)
                print(f"[SELF-CHECK] latent skill vs persistence = {skill*100:.2f}%", flush=True)
                print("  skill>0  => forecast beats 'tomorrow=today' => latents_2[t] IS the", flush=True)
                print("              correct step-0 forecast input (blocking Q resolved).", flush=True)
                print("  skill<=0 => WRONG input/rope_coords/config -- do not trust this run.", flush=True)
                print("=" * 64, flush=True)
                selfcheck_done = True
                if not (skill > 0):
                    print("[ABORT] skill<=0; aborting before full sweep.", flush=True)
                    sys.exit(2)
                if args.selfcheck_only:
                    print("[selfcheck-only] OK, exiting.", flush=True)
                    sys.exit(0)
            if t % 500 < args.batch:
                el = time.time() - t0
                print(f"  [{t}/{n_src}] {t/max(el,1):.2f} trans/s", flush=True)

    if bp:                                    # flush remainder
        err_fc, err_ps, B = flush()
        t += B

    el = time.time() - t0
    print(f"[done] {t} transitions in {el/60:.1f} min ({t/max(el,1):.2f} trans/s)", flush=True)

    np.save(os.path.join(args.out, "err_forecast.npy"), err_forecast)
    np.save(os.path.join(args.out, "err_persist.npy"), err_persist)
    meta = {
        "run_id": args.run_id, "config": args.config,
        "n_cells": N_CELLS, "dim": DIM, "n_transitions": int(t),
        "source_idx": src_idx.tolist(), "target_idx": tgt_idx.tolist(),
        "source_dt": [str(idx_to_dt(i)) for i in src_idx],
        "norm": "per-cell squared L2 over the 2048 latent dim",
        "err_forecast": "||forecast_engine(x_t,0,coords) - x_{t+1}||^2",
        "err_persist": "||x_t - x_{t+1}||^2 (persistence baseline)",
        "skill": "1 - err_forecast/err_persist (latent skill vs persistence)",
        "rope_coords": "set" if rope_coords is not None else "None",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] {args.out}/{{err_forecast.npy,err_persist.npy,meta.json}}", flush=True)


if __name__ == "__main__":
    main()