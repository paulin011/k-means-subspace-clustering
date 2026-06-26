#!/usr/bin/env bash
# Overnight seed-robustness sweep (option A): v6 config (K=128, d=64), same file sample,
# init seeds 1 & 2 -- isolates initialization sensitivity, the main risk for K-subspaces.
#
# Pipeline (all detached via setsid nohup, survives the session):
#   1. wait for the in-flight v6 regeneration to release both GPUs
#   2. verify the regenerated v6 model (shape + counts invariant)
#   3. regenerate v6's reports with the FIXED code (corrected counts, "captured" wording)
#   4. seed 1 -> v7_seed1_d64  (subspace_kmeans + analyze_clusters + temporal_spatial)
#      20-min gap
#      seed 2 -> v8_seed2_d64  (same)
# Each run uses --files-from v6/sample.json so the FILE sample is held fixed; only --seed
# (the random-token init) changes. Separate --out dirs; no symlinks anywhere.
set -u
cd /home/psaher/latents
LOG=/home/psaher/latents/seed_sweep.log
PY=/usr/bin/python3
V6=subspace_kmeans_runs/v6_subspace_big_d64
SAMPLE=$V6/sample.json
GAP=1200                                                       # 20 min between seed runs
REGEN_PAT="subspace_kmeans.py.*--out subspace_kmeans_runs/v6_subspace_big_d64"

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== seed-sweep driver started (gap=${GAP}s) ==="

# 1. wait for v6 regen to finish (it holds both GPUs)
log "waiting for v6 regen to finish..."
while pgrep -f "$REGEN_PAT" >/dev/null 2>&1; do sleep 60; done
sleep 30                                                       # let file writes flush
log "regen process exited; verifying regenerated model..."

# 2. verify the regenerated v6 model is the real K=128 d=64 and counts are reconciled
if ! $PY - >>"$LOG" 2>&1 <<'PY'
import torch
V6 = "subspace_kmeans_runs/v6_subspace_big_d64"
m = torch.load(V6 + "/model.pt", map_location="cpu", weights_only=False)
a = torch.load(V6 + "/assignments.pt", map_location="cpu", weights_only=False)
assert tuple(m["U"].shape) == (128, 2048, 64), f"BAD U shape {tuple(m['U'].shape)}"
ok = bool((m["counts"].float() == torch.bincount(a["label"].long(), minlength=128).float()).all())
print(f"v6 model OK: U={tuple(m['U'].shape)} counts==bincount(assignments)={ok}")
assert ok, "counts invariant broken"
PY
then
  log "ABORT: v6 regen verification FAILED -- skipping seeds (see regen.log)."
  log "=== seed-sweep ABORTED ==="
  exit 1
fi

# 3. regenerate v6 reports with the fixed code
log "regenerating v6 reports with fixed code..."
$PY analyze_clusters.py --dir "$V6" --out "$V6/report.md" >>"$LOG" 2>&1 && log "  v6 report.md done"
$PY temporal_spatial.py --dir "$V6"                        >>"$LOG" 2>&1 && log "  v6 temporal_report.md done"

# 4. seed runs (v7, v8) -- same sample, different init seed
N=7
for SEED in 1 2; do
  OUT="subspace_kmeans_runs/v${N}_seed${SEED}_d64"
  log "=== seed $SEED -> $OUT (K=128 d=64, --files-from $SAMPLE) ==="
  mkdir -p "$OUT"
  $PY subspace_kmeans.py --files-from "$SAMPLE" --clusters 128 --dim 64 \
      --seed "$SEED" --tokens-per-file 12288 --max-ram-gb 420 --out "$OUT" >>"$LOG" 2>&1
  rc=$?
  log "  subspace_kmeans seed $SEED exit=$rc"
  if [ $rc -eq 0 ]; then
    $PY analyze_clusters.py --dir "$OUT" --out "$OUT/report.md" >>"$LOG" 2>&1 && log "  seed$SEED report.md done"
    $PY temporal_spatial.py --dir "$OUT"                        >>"$LOG" 2>&1 && log "  seed$SEED temporal_report.md done"
  else
    log "  seed $SEED FAILED -- skipping its reports."
  fi
  N=$((N + 1))
  if [ "$SEED" != "2" ]; then log "spacing ${GAP}s before next seed..."; sleep "$GAP"; fi
done

log "=== seed-sweep COMPLETE ==="
