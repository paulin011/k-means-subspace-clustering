#!/usr/bin/env python3
"""Local latent->latent PROXY forecaster: a model error we can compute without a checkpoint.

WHY THIS EXISTS. `docs/ideas/latent_selection.md` concludes that selecting fine-tuning
timestamps by *latent geometry* (how unusual the encoder's output is) is the wrong signal --
what matters is where the *model* is wrong. The real answer needs the WeatherGenerator
checkpoint, which is not on this box, and even with it the naive latent error is confounded
because the forecast engine's input and output are not the same space
(`docs/wgen_architecture.md` §5b).

This script sidesteps both problems. It trains a small latent->latent forecaster on
`latents_2` here, so:

  * it is a **model** error, not an encoding statistic -- the categorical upgrade the
    selection doc argues for;
  * it is trained AND evaluated in one consistent space, so §5b's mismatch cannot arise
    by construction;
  * it needs no checkpoint, no decoder, no ERA5, no Anemoi reader.

Justification for a proxy at all: Selection via Proxy (Coleman et al., arXiv 1906.11829)
and Small-to-Large Generalization (arXiv 2505.16260) both find that data-selection signals
from a much smaller model transfer to the target model; TAROT's own AutoBots->Wayformer
transfer is the same phenomenon. The proxy is not a forecast product -- it is a *ranking*
device for selection.

WHAT IT PREDICTS. The 6-hour tendency, not the state:

    Delta_t = x_{t+1} - x_t          (per cell, 2048-dim)

so the trivial baseline "predict zero" IS persistence, and skill is measured directly
against `runs/persistence/v6/err_persist.npy`:

    skill = 1 - sum ||pred - truth||^2 / sum ||x_t - x_{t+1}||^2      (ratio of sums)

A proxy with skill <= 0 has learned nothing and its error field is not a usable signal; the
script says so rather than shipping a number that looks like a forecast error.

ARCHITECTURE (deliberately small). Per cell: its own token plus its 8 HEALPix NESTED
neighbours (`worldmap.healpix_nest_neighbours`), each projected 2048->`--dim` by a shared
linear, concatenated, then a few residual MLP blocks, then a head back to 2048. Neighbours
matter because 6 h of advection at ~20 m/s is ~430 km, and an nside=32 cell is ~200 km, so
the tendency is not a per-cell function. ~5-10 M parameters; single GPU (the broken
inter-GPU P2P on this machine never enters).

DATA. Consecutive pairs are needed, so the loader reads `--chunks` contiguous blocks of
`--chunk-len` files spread evenly over the full 2014-2022 range (sequential reads, and every
season represented), holds them as fp16 in RAM, and samples random (pair, cell) minibatches.
Default 20 x 150 = 3000 files ~ 151 GB fp16.

OUTPUT. `runs/proxy/<name>/{model.pt, err_proxy.npy [13020,12288] float32, meta.json,
report.md}` -- `err_proxy` has the same shape/contract/NESTED ordering as `err_persist.npy`
and `err_forecast.npy`, so `analyze_forecast_error.py` and the cluster join work unchanged.

Usage:
  python3 src/forecast/proxy_forecast.py --out runs/proxy/v1 --steps 4000
  python3 src/forecast/proxy_forecast.py --out runs/proxy/v1 --infer-only   # reuse model.pt
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ -> `common`
from common.worldmap import N_CELLS, NSIDE, healpix_nest_neighbours

DIM = 2048
N_FILES_TOTAL = 13021


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_block(ids, workers=16, dtype=torch.float16):
    """Load `ids` latent files into one [n, 12288, 2048] fp16 tensor (threaded)."""
    out = torch.empty((len(ids), N_CELLS, DIM), dtype=dtype)

    def one(k):
        p = f"latents_2/latent_{ids[k]}.pt"
        out[k] = torch.load(p, map_location="cpu", weights_only=False)["latent"][0].to(dtype)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, range(len(ids)))):
            if (i + 1) % 200 == 0:
                print(f"    loaded {i + 1}/{len(ids)}", flush=True)
    return out


def plan_chunks(n_chunks, chunk_len, n_total=N_FILES_TOTAL, seed=0):
    """Contiguous blocks spread evenly over the record (so every season is represented)."""
    starts = np.linspace(0, n_total - chunk_len - 1, n_chunks).astype(int)
    ids, pairs = [], []
    for s in starts:
        base = len(ids)
        ids.extend(range(s, s + chunk_len))
        pairs.extend((base + i, base + i + 1) for i in range(chunk_len - 1))
    return ids, np.array(pairs)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class ProxyForecaster(nn.Module):
    """self + 8 HEALPix neighbours -> shared projection -> residual MLP -> tendency."""

    def __init__(self, dim=256, blocks=4, n_neigh=8, hidden=2):
        super().__init__()
        self.proj = nn.Linear(DIM, dim)
        self.mix = nn.Linear(dim * (n_neigh + 1), dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * hidden), nn.GELU(),
                          nn.Linear(dim * hidden, dim))
            for _ in range(blocks))
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, DIM)
        nn.init.zeros_(self.head.weight)          # start at exactly persistence (delta = 0)
        nn.init.zeros_(self.head.bias)

    def forward(self, ctx):
        """ctx [B, n_neigh+1, 2048] -- index 0 is the cell itself. Returns delta [B, 2048]."""
        h = self.proj(ctx).flatten(1)
        h = self.mix(h)
        for b in self.blocks:
            h = h + b(h)
        return self.head(self.head_norm(h))


def gather_ctx(x, cells, nb):
    """x [T,12288,2048] (any device), cells [B] long, nb [12288,8] -> ctx [B,9,2048].

    `nb=None` is the **ablation**: context is the cell alone, so the model can only learn a
    per-cell transformation of the current token. The gap between the two settings measures
    what spatial context is worth for a 6 h tendency (advection at ~20 m/s covers ~430 km
    against a ~200 km cell, so it should be large).
    """
    if nb is None:
        return x[cells][:, None]                               # [B, 1, 2048]
    idx = torch.cat([cells[:, None], nb[cells]], 1)            # [B, 9]
    return x[idx]


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(args, data, pairs, nb, dev):
    torch.manual_seed(args.seed)
    model = ProxyForecaster(args.dim, args.blocks,
                            n_neigh=0 if args.no_neighbours else 8).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  proxy params: {n_par / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(args.seed)
    nb_d = None if args.no_neighbours else nb.to(dev)
    n_val = max(1, len(pairs) // 10)
    val_pairs, tr_pairs = pairs[:n_val], pairs[n_val:]        # held-out PAIRS, not cells
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        p = tr_pairs[rng.integers(len(tr_pairs))]
        cells = torch.from_numpy(rng.choice(N_CELLS, args.batch, replace=False)).to(dev)
        a = data[p[0]].to(dev, non_blocking=True).float()
        b = data[p[1]].to(dev, non_blocking=True).float()
        ctx = gather_ctx(a, cells, nb_d)
        tgt = b[cells] - a[cells]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(ctx)
            loss = ((pred.float() - tgt) ** 2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            vs, vp = evaluate(model, data, val_pairs, nb_d, dev, args, rng)
            hist.append(dict(step=step, train_loss=float(loss), val_err=vs, val_persist=vp,
                             skill=1 - vs / vp))
            print(f"  step {step:5d}/{args.steps}  train {float(loss):8.1f}  "
                  f"val {vs:8.1f}  persist {vp:8.1f}  skill {1 - vs / vp:+.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return model, hist, val_pairs


@torch.no_grad()
def evaluate(model, data, val_pairs, nb_d, dev, args, rng, n_pairs=6):
    """Mean squared error of the proxy and of persistence over a val sample (ratio of sums)."""
    model.eval()
    se = sp = n = 0.0
    for p in val_pairs[rng.choice(len(val_pairs), min(n_pairs, len(val_pairs)), replace=False)]:
        cells = torch.from_numpy(rng.choice(N_CELLS, args.batch, replace=False)).to(dev)
        a = data[p[0]].to(dev).float()
        b = data[p[1]].to(dev).float()
        tgt = b[cells] - a[cells]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(gather_ctx(a, cells, nb_d))
        se += float(((pred.float() - tgt) ** 2).sum())
        sp += float((tgt ** 2).sum())                          # persistence = predict zero
        n += cells.numel()
    model.train()
    return se / n, sp / n


# ---------------------------------------------------------------------------
# full-dataset inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_all(model, nb, dev, out_path, batch_files=64, cell_chunk=4096):
    """err_proxy[t, cell] = ||pred_delta - true_delta||^2 for every transition t -> t+1."""
    model.eval()
    nb_d = nb.to(dev)
    err = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(N_FILES_TOTAL - 1, N_CELLS))
    prev = None
    t0 = time.time()
    for start in range(0, N_FILES_TOTAL, batch_files):
        ids = list(range(start, min(start + batch_files, N_FILES_TOTAL)))
        blk = load_block(ids, workers=24)                      # [n,12288,2048] fp16
        seq = blk if prev is None else torch.cat([prev[None], blk], 0)
        base = start - (0 if prev is None else 1)
        for k in range(seq.shape[0] - 1):
            a = seq[k].to(dev).float()
            b = seq[k + 1].to(dev).float()
            row = base + k
            for c0 in range(0, N_CELLS, cell_chunk):
                cells = torch.arange(c0, min(c0 + cell_chunk, N_CELLS), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(gather_ctx(a, cells, nb_d))
                d = pred.float() - (b[cells] - a[cells])
                err[row, c0:cells[-1].item() + 1] = (d ** 2).sum(1).cpu().numpy()
        prev = blk[-1]
        print(f"  inferred {min(start + batch_files, N_FILES_TOTAL)}/{N_FILES_TOTAL} "
              f"({time.time() - t0:.0f}s)", flush=True)
    err.flush()
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=20, help="contiguous file blocks to load")
    ap.add_argument("--chunk-len", type=int, default=150, help="files per block")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8192, help="cells per step")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-neighbours", action="store_true",
                    help="ABLATION: cell-only context, no HEALPix neighbours")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--no-infer", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    t0 = time.time()

    nb = torch.from_numpy(healpix_nest_neighbours(NSIDE).astype(np.int64))
    nb = torch.where(nb < 0, torch.arange(N_CELLS)[:, None], nb)   # -1 -> self (8 of 12288)
    print(f"neighbour table {tuple(nb.shape)}; {(nb < 0).sum()} unresolved", flush=True)

    model_path = os.path.join(args.out, "model.pt")
    if args.infer_only:
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        model = ProxyForecaster(ck["dim"], ck["blocks"]).to(dev)
        model.load_state_dict(ck["state"])
        hist = ck.get("history", [])
    else:
        ids, pairs = plan_chunks(args.chunks, args.chunk_len)
        print(f"loading {len(ids)} files -> {len(pairs)} transition pairs "
              f"({len(ids) * N_CELLS * DIM * 2 / 2**30:.0f} GiB fp16)", flush=True)
        data = load_block(ids, workers=24)
        print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
        model, hist, _ = train(args, data, pairs, nb, dev)
        torch.save(dict(state=model.state_dict(), dim=args.dim, blocks=args.blocks,
                        history=hist, config=vars(args)), model_path)
        del data
        print(f"model saved -> {model_path}", flush=True)

    skill = hist[-1]["skill"] if hist else float("nan")
    if not args.no_infer:
        if skill <= 0:
            print(f"\nABORT: held-out skill {skill:+.4f} <= 0 -- the proxy has not beaten "
                  f"persistence, so its error field is not a usable selection signal.\n"
                  f"Train longer / larger (--steps, --dim, --blocks) before running inference.")
            sys.exit(2)
        print(f"\nfull-dataset inference (held-out skill {skill:+.4f})", flush=True)
        infer_all(model, nb, dev, os.path.join(args.out, "err_proxy.npy"))

    json.dump(dict(config=vars(args), history=hist, final_skill=skill,
                   wall_min=(time.time() - t0) / 60,
                   contract="err_proxy[t,cell] = ||pred_delta - true_delta||^2, "
                            "NESTED, same shape as err_persist.npy"),
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. skill={skill:+.4f}")


if __name__ == "__main__":
    main()
