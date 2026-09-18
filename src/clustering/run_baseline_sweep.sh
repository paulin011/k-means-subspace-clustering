#!/usr/bin/env bash
# Baseline sweep for the final report: plain k-means (subspace_kmeans --dim 0) and
# greedy k-center, both at K=128 on the SAME 7000-file token sample as v2/v6/v12
# (--files-from v6/sample.json), 3 init seeds each.
#   k-means : v13_kmeans_d0 (seed 0), v14_kmeans_seed1_d0, v15_kmeans_seed2_d0
#   k-center: v16_kcenter_seed{0,1,2}  (one process, one data load, 3 selections)
# Runs are strictly sequential: each holds the ~352 GB fp16 sample in RAM (512 GB box).
# Launch detached (survives the session):
#   setsid nohup bash src/clustering/run_baseline_sweep.sh >/dev/null 2>&1 &
set -u
cd /home/psaher/latents
LOG=/home/psaher/latents/logs/baseline_sweep.log
PY=/usr/bin/python3
SAMPLE=runs/clustering/v6_subspace_big_d64/sample.json
GAP=60

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "=== baseline sweep started ==="

# k-means (d=0), 3 init seeds
N=13
for SEED in 0 1 2; do
  SUFFIX=""; [ "$SEED" != "0" ] && SUFFIX="_seed${SEED}"
  OUT="runs/clustering/v${N}_kmeans${SUFFIX}_d0"
  log "=== k-means seed $SEED -> $OUT (K=128 d=0, --files-from $SAMPLE) ==="
  $PY src/clustering/subspace_kmeans.py --files-from "$SAMPLE" --clusters 128 --dim 0 \
      --seed "$SEED" --tokens-per-file 12288 --max-ram-gb 420 --out "$OUT" >>"$LOG" 2>&1
  rc=$?
  log "  subspace_kmeans d0 seed $SEED exit=$rc"
  if [ $rc -eq 0 ]; then
    $PY src/analysis/analyze_clusters.py --dir "$OUT" --out "$OUT/report.md" >>"$LOG" 2>&1 && log "  report.md done"
  else
    log "  k-means seed $SEED FAILED"
  fi
  N=$((N + 1))
  sleep "$GAP"
done

# k-center, 3 init seeds from one data load
OUT=runs/clustering/v16_kcenter
log "=== k-center -> ${OUT}_seed{0,1,2} (K=128, --files-from $SAMPLE) ==="
$PY src/clustering/kcenter.py --files-from "$SAMPLE" --clusters 128 --seed 0 \
    --seeds 0 1 2 --max-ram-gb 420 --out "$OUT" >>"$LOG" 2>&1
rc=$?
log "  kcenter exit=$rc"
if [ $rc -eq 0 ]; then
  for S in 0 1 2; do
    $PY src/analysis/analyze_clusters.py --dir "${OUT}_seed${S}" \
        --out "${OUT}_seed${S}/report.md" >>"$LOG" 2>&1 && log "  kcenter seed$S report.md done"
  done
else
  log "  k-center FAILED"
fi

log "=== baseline sweep COMPLETE ==="
