#!/usr/bin/env python3
"""Does the forecast engine's OUTPUT live in the same space as its INPUT? Measure it.

THE QUESTION. `extract_forecast_error.py` computes
`err[t,cell] = ||forecast_engine(latents_2[t]) - latents_2[t+1]||^2`. That is only meaningful
if the FE's output and the encoder's output are the same representation. They are not
guaranteed to be (`docs/wgen_architecture.md` §5b):

  * the encoder output is UN-normalised -- `ae_global_trailing_layer_norm: False` in every
    config (`engines.py:520`);
  * the FE contains a PARAMETER-FREE `LayerNorm(2048, elementwise_affine=False)` after block
    7 under `config_forecasting.yml` (`fe_layer_norm_after_blocks: [7]`, `engines.py:609-612`),
    so everything after it is anchored at per-token std 1.0 plus whatever blocks 8-15 add back;
  * nothing in training ties the two -- the base recipe's loss is `LossPhysical` on *decoded*
    fields, and even the latent SSL loss runs on `latent_pre_norm(tokens)`, i.e. NORMALISED
    tokens, while `latents_2` stores `z_pre_norm` (raw).

MEASURED LOCALLY (no checkpoint needed, `docs/wgen_architecture.md` §5b.0): the stored
latents have per-token mean +0.002 and **per-token std 1.69** (L2 76.2), against 1.000
(L2 45.25) for a parameter-free LayerNorm output. So the two spaces are ~1.7x apart AT MOST
-- a correctable bias, not a catastrophe. Whether any gap remains at the FE's *output*
depends on how much blocks 8-15 add back, which cannot be known without running the model.

THIS SCRIPT IS THAT ONE FORWARD PASS. It reports, on a handful of real transitions:

  1. the checkpoint's actual settings for the three switches that decide the answer
     (`fe_layer_norm_after_blocks`, `ae_global_trailing_layer_norm`, and the number of INPUT
     steps -- `model.py:689-691` SUMS the encoder output over input steps, so if that is >1
     then a single `latents_2[t]` is not the FE's input at all, independently of any norm);
  2. per-token mean / std / L2 of the FE input, the FE output, and the target;
  3. four candidate error definitions and each one's skill against persistence.

VERDICT RULE. If the FE output's per-token std comes back within ~10% of the target's
(~1.69), the two spaces are effectively aligned and the naive error in
`extract_forecast_error.py` is usable as-is. Otherwise pick from the alternatives it scores.

THE FOUR CANDIDATES (see `docs/wgen_architecture.md` §5b.1):
  naive     ||FE(x_t) - x_{t+1}||^2                  -- confounded if the scales differ
  ln_both   ||LN(FE(x_t)) - LN(x_{t+1})||^2          -- pattern error only, magnitude discarded
  rescaled  naive after matching FE output std to the target's (a single global scalar)
  rollout   ||FE^2(x_t) - FE(x_{t+1})||^2            -- BOTH sides in FE space, no mismatch by
            construction; but it measures trajectory divergence (self-consistency), not
            accuracy against truth, so read it as an ensemble-spread-like difficulty signal

`--synthetic` runs the whole diagnostic against a STAND-IN forecast engine built to the same
structure (residual blocks, optional parameter-free LayerNorm at block 7) with near-identity
init. It needs no checkpoint and exists to (a) verify this harness end-to-end and (b) show
what the output looks like when a mismatch is present by construction -- run it with
`--synthetic-ln` and without, and compare.

Usage (supercomputer, real):
  python3 src/supercomputer/fe_space_check.py --config <cfg.yml> --run-id <id> --n 8
Usage (this box, harness check):
  python3 src/supercomputer/fe_space_check.py --synthetic --synthetic-ln --n 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

DIM = 2048
N_CELLS = 12288


# ---------------------------------------------------------------------------
# stand-in forecast engine (harness verification; NOT a model of WGen's weights)
# ---------------------------------------------------------------------------
class SyntheticFE(nn.Module):
    """Same *structure* as ForecastingEngine: residual blocks, near-identity init, and
    optionally a parameter-free LayerNorm after block 7 (`fe_layer_norm_after_blocks: [7]`)."""

    def __init__(self, n_blocks=16, ln_after=(7,), dim=DIM, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList()
        self.kinds = []
        for i in range(n_blocks):
            mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                nn.Linear(dim * 2, dim))
            for m in mlp.modules():                       # engines.py:614-621 init_weights_final
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.001)
                    nn.init.normal_(m.bias, 0.0, 0.001)
            self.blocks.append(mlp)
            self.kinds.append("mlp")
            if i in ln_after:
                self.blocks.append(nn.LayerNorm(dim, elementwise_affine=False))
                self.kinds.append("ln")

    def forward(self, x, *_a, **_k):
        for b, k in zip(self.blocks, self.kinds):
            x = b(x) if k == "ln" else x + b(x)           # LN replaces; MLP is residual
        return x


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def token_stats(x):
    """x [cells, dim] -> per-token mean / std / L2, averaged over cells."""
    x = x.float()
    return dict(mean=float(x.mean(1).mean()), std=float(x.std(1).mean()),
                l2=float(x.norm(dim=1).mean()))


def sq_err(a, b):
    return ((a.float() - b.float()) ** 2).sum(1)          # [cells]


def layer_norm(x):
    x = x.float()
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def load_latent(path):
    return torch.load(path, map_location="cpu", weights_only=False)["latent"][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents-dir", default="latents_2")
    ap.add_argument("--n", type=int, default=6, help="transitions to test")
    ap.add_argument("--stride", type=int, default=2000, help="spacing between test times")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON here")
    # real mode
    ap.add_argument("--config", help="production WGen config .yml")
    ap.add_argument("--run-id", help="trained run_id (checkpoint)")
    # synthetic mode
    ap.add_argument("--synthetic", action="store_true", help="stand-in FE, no checkpoint")
    ap.add_argument("--synthetic-ln", action="store_true",
                    help="give the stand-in the parameter-free LayerNorm at block 7")
    ap.add_argument("--synthetic-blocks", type=int, default=16)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    cfg_report = {}
    if args.synthetic:
        ln_after = (7,) if args.synthetic_ln else ()
        fe = SyntheticFE(args.synthetic_blocks, ln_after).to(dev).eval()
        cfg_report = dict(mode="synthetic", fe_layer_norm_after_blocks=list(ln_after),
                          fe_num_blocks=args.synthetic_blocks,
                          ae_global_trailing_layer_norm=False, num_input_steps="n/a")
        print("*** SYNTHETIC stand-in forecast engine -- verifies this harness only. ***")
        print("*** Numbers below say nothing about the real WeatherGenerator weights. ***\n")
    else:
        if not (args.config and args.run_id):
            sys.exit("real mode needs --config and --run-id (or pass --synthetic)")
        # TODO(supercomputer): mirror extract_forecast_error.py's loader exactly.
        sys.path.insert(0, os.environ.get("WGEN_SRC", "/p/project/weathergen/WeatherGenerator/src"))
        from weathergen.common.config import Config                       # noqa: E402
        from weathergen.model.model_interface import init_model_and_shard  # noqa: E402
        cf = Config(args.config)
        model, model_params = init_model_and_shard(cf, args.run_id, device=dev)
        model.eval()
        fe = model.forecast_engine
        cfg_report = dict(
            mode="checkpoint", run_id=args.run_id,
            fe_layer_norm_after_blocks=list(cf.get("fe_layer_norm_after_blocks", [])),
            ae_global_trailing_layer_norm=bool(cf.get("ae_global_trailing_layer_norm", False)),
            fe_num_blocks=int(cf.get("fe_num_blocks", -1)),
            ae_global_dim_embed=int(cf.get("ae_global_dim_embed", -1)))
        print("Checkpoint config -- the three switches that decide the answer:")
        for k in ("fe_layer_norm_after_blocks", "ae_global_trailing_layer_norm", "fe_num_blocks"):
            print(f"  {k:34s} = {cfg_report[k]}")
        print("  NOTE also check the number of INPUT steps: model.py:689-691 SUMS the encoder\n"
              "       output over input steps, so if it is > 1 a single latents_2[t] is NOT\n"
              "       the FE's input, independently of any normalisation.\n")

    def run_fe(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=dev.startswith("cuda")):
            return fe(x.unsqueeze(0))[0].float() if x.dim() == 2 else fe(x)[0].float()

    times = [i * args.stride for i in range(args.n) if i * args.stride + 2 < 13021]
    rows, acc = [], {k: [0.0, 0.0] for k in ("naive", "ln_both", "rescaled", "rollout")}
    for t in times:
        x0 = load_latent(f"{args.latents_dir}/latent_{t}.pt").to(dev)
        x1 = load_latent(f"{args.latents_dir}/latent_{t + 1}.pt").to(dev)
        pred = run_fe(x0)
        s_in, s_out, s_tg = token_stats(x0), token_stats(pred), token_stats(x1)
        scale = s_tg["std"] / max(s_out["std"], 1e-9)

        persist = sq_err(x0, x1)
        e = dict(naive=sq_err(pred, x1),
                 ln_both=sq_err(layer_norm(pred), layer_norm(x1)),
                 rescaled=sq_err(pred * scale, x1),
                 rollout=sq_err(run_fe(pred), run_fe(x1)))
        # rollout's own baseline is persistence in FE space, not raw persistence
        base = dict.fromkeys(e, float(persist.sum()))
        base["rollout"] = float(sq_err(run_fe(x0), run_fe(x1)).sum())
        base["ln_both"] = float(sq_err(layer_norm(x0), layer_norm(x1)).sum())
        for k, v in e.items():
            acc[k][0] += float(v.sum())
            acc[k][1] += base[k]
        rows.append(dict(t=t, in_=s_in, out=s_out, target=s_tg, scale_gap=scale,
                         err={k: float(v.mean()) for k, v in e.items()},
                         persist=float(persist.mean())))
        print(f"t={t:6d}  in std {s_in['std']:.3f} | FE-out std {s_out['std']:.3f} | "
              f"target std {s_tg['std']:.3f} | gap x{scale:.3f}")

    print("\nper-token scale (mean over the tested transitions):")
    for nm, key in (("FE input  (latents_2[t])", "in_"), ("FE output (prediction)", "out"),
                    ("target    (latents_2[t+1])", "target")):
        v = np.mean([r[key]["std"] for r in rows])
        l2 = np.mean([r[key]["l2"] for r in rows])
        print(f"  {nm:28s} std {v:.4f}   L2 {l2:7.2f}")
    gap = float(np.mean([r["scale_gap"] for r in rows]))
    print(f"  scale gap (target/FE-out)       x{gap:.4f}")

    print("\nerror definitions, skill = 1 - sum(err)/sum(baseline)  (ratio of sums):")
    for k in ("naive", "rescaled", "ln_both", "rollout"):
        s, b = acc[k]
        print(f"  {k:9s} skill {1 - s / b:+.4f}   (err {s / len(rows) / N_CELLS:9.1f})")

    aligned = abs(gap - 1.0) <= 0.10
    verdict = ("ALIGNED: the FE output sits within 10% of the target's per-token scale, so the "
               "naive error in extract_forecast_error.py is usable as-is."
               if aligned else
               f"MISMATCH: the FE output is off the target scale by x{gap:.3f}. Do NOT use the "
               "naive error; pick whichever of rescaled / ln_both / rollout scores best above, "
               "and record the choice wherever the array is consumed.")
    print(f"\nVERDICT: {verdict}")
    if args.synthetic:
        print("(synthetic run -- this verdict describes the stand-in, not the real model)")

    out = dict(config=cfg_report, args=vars(args), scale_gap=gap, aligned=bool(aligned),
               verdict=verdict, rows=rows,
               skill={k: 1 - acc[k][0] / acc[k][1] for k in acc},
               wall_s=time.time() - t0)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
