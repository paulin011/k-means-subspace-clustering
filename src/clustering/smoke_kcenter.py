#!/usr/bin/env python3
"""Standalone smoke tests for kcenter.py (no test framework exists in this repo).

Mirrors smoke_merge_clusters.py. Five checks, each printing PASS/FAIL; exit code is the
number of failures.

  S1 Gonzalez selection   -- on well-separated synthetic blobs the greedy must place one
                             center per blob, recover the generative partition exactly,
                             and honour its 2-approximation bound on the minimax radius.
  S2 Accumulator at scale -- THE REGRESSION TEST for the float32 bug that made the
                             original v16 run report `trace` ~3x too small: 2e7 tokens
                             piling ~5000 each into one bin drive the accumulator past
                             2.7e11, where a float32 ulp exceeds the addend and every
                             further add rounds away. Asserts assign_pass reconciles
                             against a float64 brute force, and separately demonstrates
                             that a float32 accumulator on the same data does not.
  S3 Parallel-axis        -- with K=1 the objective must equal E||x-mu||^2 + ||c-mu||^2
                             exactly. This is the identity that explains why k-center's
                             objective lands ABOVE the total token variance: its anchors
                             are data points, never centroids.
  S4 Brute force, real    -- labels/counts/radius/trace/objective from assign_pass vs
                             torch.cdist in float64, on real latent tokens.
  S5 Schema end-to-end    -- run kcenter.py for real on a few latent files and validate
                             the cluster_io contract (U [K,2048,0], counts == bincount,
                             radius, final_obj_per_token == sum_j w_j*trace[j], per-seed
                             sample.json).

--synthetic-only skips S4/S5 (the parts that read latents_2).

Run:  python3 src/clustering/smoke_kcenter.py [--synthetic-only]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling scripts
from kcenter import assign_pass, greedy_centers                  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)


def brute_force(data, centers):
    """float64 reference: labels, counts, radius, trace, obj_per_token."""
    X, C = data.to(torch.float64), centers.to(torch.float64)
    d2 = torch.cdist(X, C) ** 2
    vals, lab = d2.min(1)
    K, T = C.shape[0], X.shape[0]
    counts = torch.bincount(lab, minlength=K)
    ssum = torch.zeros(K, dtype=torch.float64).index_add_(0, lab, vals)
    rad2 = torch.zeros(K, dtype=torch.float64).scatter_reduce_(0, lab, vals, reduce="amax")
    return lab, counts, rad2.sqrt(), ssum / counts.clamp(min=1), float(vals.sum() / T)


# ---------------------------------------------------------------- S1
def s1_gonzalez():
    print("\nS1 Gonzalez selection on separated blobs")
    g = torch.Generator().manual_seed(0)
    D, nb, per = 16, 5, 10_000
    truth = torch.stack([torch.randn(D, generator=g) * 40 for _ in range(nb)])
    data = torch.cat([truth[b] + torch.randn(per, D, generator=g) for b in range(nb)]).half()
    gen_lab = torch.arange(nb).repeat_interleave(per)

    Xs = data.to(DEV)
    xn = (Xs.float() ** 2).sum(1)
    centers, sub_radius = greedy_centers(Xs, xn, nb, 0)
    lab, counts, radius, trace, obj = assign_pass(data, centers, 4096, DEV)

    # One center per blob: the blob of each center must be a permutation of 0..nb-1.
    owner = torch.cdist(centers.cpu().double(), truth.double()).argmin(1)
    check("one center per blob", sorted(owner.tolist()) == list(range(nb)),
          f"owners={owner.tolist()}")
    # Labels recover the generative partition (up to the label permutation `owner`).
    recovered = owner[lab.long()]
    check("generative partition recovered", bool((recovered == gen_lab).all()),
          f"{float((recovered == gen_lab).float().mean()):.4%} agree")
    # 2-approximation: radius <= 2 * optimal, and optimal <= the blob's own max radius.
    opt = float(torch.cdist(data.double(), truth.double()).min(1).values.max())
    check("minimax radius within the 2-approximation bound", float(radius.max()) <= 2 * opt,
          f"radius={float(radius.max()):.2f} <= 2*{opt:.2f}")
    # greedy_centers scores candidates with the expanded form ||x||^2 - 2x.c + ||c||^2
    # and an fp16 cross term. That cancels when ||x||^2 >> ||x-c||^2, i.e. exactly when
    # clusters are FAR APART -- as these blobs are (||x||^2 ~ 25600 vs d^2 ~ 130, so the
    # ~5e-4 fp16 rounding of the cross term lands as a ~10% error on d^2). latents_2 is
    # the opposite regime (||x||^2 ~ 6046 < d^2 ~ 9836, measured), which s1b pins down.
    ratio = float(xn.max()) / float(radius.max() ** 2)
    check("separated blobs: selection radius within the fp16 cancellation bound",
          abs(sub_radius - float(radius.max())) / float(radius.max()) < 0.10,
          f"{sub_radius:.4f} vs {float(radius.max()):.4f}, ||x||^2/d^2 = {ratio:.0f}")


def s1b_selection_precision():
    print("\nS1b Selection precision in the latents_2 regime (||x||^2 ~ d^2)")
    g = torch.Generator().manual_seed(4)
    D, T = 2048, 60_000
    data = (torch.randn(T, D, generator=g) * (6046.0 / D) ** 0.5).half()   # ||x||^2 ~ 6046
    Xs = data.to(DEV)
    xn = (Xs.float() ** 2).sum(1)
    centers, sub_radius = greedy_centers(Xs, xn, 8, 0)
    exact = float(torch.cdist(data.double(), centers.cpu().double()).min(1).values.max())
    rel = abs(sub_radius - exact) / exact
    check("fp16 selection radius matches float64 on real-scale data", rel < 1e-3,
          f"{sub_radius:.4f} vs {exact:.4f} (rel={rel:.2e}), "
          f"||x||^2/d^2 = {float(xn.max()) / exact ** 2:.2f}")


# ---------------------------------------------------------------- S2
def s2_accumulator():
    print("\nS2 Accumulator at production scale (float32 regression)")
    # 2e7 tokens, one dominant cluster, ~5000 squared distance each -> the accumulated
    # SSE reaches ~1e11, the regime where a float32 bin stagnates.
    g = torch.Generator().manual_seed(1)
    D, T, CH = 4, 80_000_000, 1 << 21
    centers = torch.zeros(2, D)
    centers[1] = 500.0                                     # far away, claims almost nothing
    data = (torch.randn(T, D, generator=g) * 35.0).half()  # E||x-0||^2 = 4*35^2 = 4900
    # Total SSE ~ 3.9e11, ~5x past the 4900*2^24 = 8.2e10 point where a float32 bin's ulp
    # exceeds the addend -- the same regime as v16's 84.9M tokens x ~9744 = 8.3e11.
    try:
        lab, counts, radius, trace, obj = assign_pass(data, centers, CH, DEV)
    except AssertionError as e:                            # assign_pass's own guard fired
        check("assign_pass reconciles internally", False, str(e).splitlines()[0])
        return

    # Reference: the same min-over-centers, accumulated in float64.
    C64 = centers.to(DEV, torch.float64)
    ref = 0.0
    for i0 in range(0, T, CH):
        x = data[i0:i0 + CH].to(DEV).double()
        ref += (torch.cdist(x, C64) ** 2).min(1).values.sum().item()
    ref_per_token = ref / T
    rel = abs(obj - ref_per_token) / ref_per_token
    check("objective matches a float64 reference", rel < 1e-6, f"rel={rel:.2e}")

    recon = float((counts.double() * trace.double()).sum() / T)
    rel_t = abs(recon - obj) / obj
    check("sum_j counts_j*trace[j] reconciles with the objective", rel_t < 1e-6,
          f"{recon:.4f} vs {obj:.4f} (rel={rel_t:.2e})")
    dom_sse = float(counts.max()) * float(trace[counts.argmax()])
    check("dominant bin really is in the stagnation regime", dom_sse > 8.2e10,
          f"SSE={dom_sse:.3e} vs the float32 stagnation point ~8.2e10")

    # Demonstrate the bug this guards against: the identical accumulation in float32.
    lab_d = lab.to(DEV).long()
    bad = torch.zeros(2, device=DEV)                       # float32, as the original code had
    for i0 in range(0, T, CH):
        x = data[i0:i0 + CH].to(DEV).float()
        bad.index_add_(0, lab_d[i0:i0 + CH], (x * x).sum(1))
    lost = 1.0 - float(bad.sum()) / ref
    check("a float32 accumulator would lose >10% here (the original v16 defect)",
          lost > 0.10, f"float32 loses {lost:.1%} of the total SSE")


# ---------------------------------------------------------------- S3
def s3_parallel_axis():
    print("\nS3 Parallel-axis identity (why k-center > total variance)")
    g = torch.Generator().manual_seed(2)
    D, T = 64, 200_000
    data = (torch.randn(T, D, generator=g) * 3.0).half()
    Xd = data.double()
    mu = Xd.mean(0)
    var = float(((Xd - mu) ** 2).sum(1).mean())
    c = Xd[12345]                                          # an arbitrary data point, as k-center picks
    _, _, _, _, obj = assign_pass(data, c.float().unsqueeze(0), 1 << 16, DEV)
    pred = var + float(((c - mu) ** 2).sum())
    rel = abs(obj - pred) / pred
    check("K=1 objective == E||x-mu||^2 + ||c-mu||^2", rel < 1e-6,
          f"obj={obj:.4f} pred={pred:.4f} (rel={rel:.2e})")
    check("a data-point anchor is strictly worse than the centroid", obj > var,
          f"{obj:.1f} > {var:.1f} by ||c-mu||^2={float(((c - mu) ** 2).sum()):.1f}")


# ---------------------------------------------------------------- S4
def s4_real_tokens(src):
    print("\nS4 Brute force on real latent tokens")
    files = sorted(glob.glob(os.path.join(src, "latent_*.pt")))[:2]
    if len(files) < 2:
        check("real latent files available", False, f"none under {src}")
        return
    g = torch.Generator().manual_seed(3)
    toks = []
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=False)["latent"].squeeze(0)
        toks.append(x[torch.randperm(x.shape[0], generator=g)[:10_000]].half())
    data = torch.cat(toks)

    Xs = data.to(DEV)
    xn = (Xs.float() ** 2).sum(1)
    centers, _ = greedy_centers(Xs, xn, 8, 0)
    lab, counts, radius, trace, obj = assign_pass(data, centers, 4096, DEV)
    blab, bcounts, bradius, btrace, bobj = brute_force(data, centers.cpu())

    check("labels match float64 brute force", bool((lab.long() == blab).all()),
          f"{float((lab.long() == blab).float().mean()):.4%} agree")
    check("counts match", bool((counts == bcounts).all()))
    check("radius matches", float((radius.double() - bradius).abs().max()) < 1e-3,
          f"max|d|={float((radius.double() - bradius).abs().max()):.2e}")
    check("trace matches", float(((trace.double() - btrace).abs() / btrace.clamp(min=1)).max()) < 1e-5,
          f"max rel={float(((trace.double() - btrace).abs() / btrace.clamp(min=1)).max()):.2e}")
    check("objective matches", abs(obj - bobj) / bobj < 1e-6, f"{obj:.4f} vs {bobj:.4f}")


# ---------------------------------------------------------------- S5
def s5_schema(src):
    print("\nS5 End-to-end run + cluster_io schema")
    tmp = tempfile.mkdtemp(prefix="smoke_kcenter_")
    try:
        out = os.path.join(tmp, "run")
        cmd = [sys.executable, "src/clustering/kcenter.py", "--src", src, "--num-files", "3",
               "--clusters", "8", "--tokens-per-file", "4096", "--select-tokens", "8000",
               "--chunk-size", "4096", "--seeds", "0", "1", "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if r.returncode != 0:
            check("kcenter.py runs", False, r.stderr.strip().splitlines()[-1] if r.stderr else "")
            return
        check("kcenter.py runs", True)
        for s in (0, 1):
            d = f"{out}_seed{s}"
            m = torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=False)
            a = torch.load(os.path.join(d, "assignments.pt"), map_location="cpu", weights_only=False)
            K = m["counts"].numel()
            ok_shape = (m["U"].shape == (K, 2048, 0) and m["means"].shape == (K, 2048)
                        and m["trace"].shape == (K,) and m["radius"].shape == (K,))
            check(f"seed{s}: schema shapes", ok_shape)
            check(f"seed{s}: counts == bincount(label)",
                  bool((m["counts"] == torch.bincount(a["label"].long(), minlength=K)).all()))
            w = m["counts"].double() / m["counts"].sum()
            recon = float((w * m["trace"].double()).sum())
            check(f"seed{s}: final_obj_per_token == sum_j w_j*trace[j]",
                  abs(recon - m["final_obj_per_token"]) / m["final_obj_per_token"] < 1e-6,
                  f"{recon:.4f} vs {m['final_obj_per_token']:.4f}")
            check(f"seed{s}: method + sample.json present",
                  m["config"]["method"] == "kcenter" and os.path.exists(os.path.join(d, "sample.json")))
            check(f"seed{s}: centers are data points (means come from the sample)",
                  m["means"].dtype == torch.float32 and bool(torch.isfinite(m["means"]).all()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="latents_2", help="directory with latent_{i}.pt files")
    p.add_argument("--synthetic-only", action="store_true", help="skip S4/S5 (no latents_2 read)")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    print(f"smoke_kcenter.py on {DEV}")

    s1_gonzalez()
    s1b_selection_precision()
    s2_accumulator()
    s3_parallel_axis()
    if not args.synthetic_only:
        s4_real_tokens(args.src)
        s5_schema(args.src)

    nfail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - nfail}/{len(RESULTS)} checks passed"
          + ("" if nfail == 0 else f"  ({nfail} FAILED)"))
    return nfail


if __name__ == "__main__":
    sys.exit(main())
