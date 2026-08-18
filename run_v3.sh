#!/usr/bin/env bash
# Wait for label propagation, then train v3 on the restored labels.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
LOG=run_v3.log
echo "=== waiting for propagation ===" | tee "$LOG"
while [ $(ls consensus_prop 2>/dev/null | wc -l) -lt 707 ]; do sleep 30; done
echo "propagation done at $(date)" | tee -a "$LOG"
while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -gt 68 ]; do sleep 30; done
echo "=== training v3 ===" | tee -a "$LOG"
$PY -u src/spine_train2.py --epochs 40 --val-every 4 --val-limit 25 \
    --consensus-dir consensus_prop --tag spine_v3 --wait-for-gpu 2>&1 | tee -a "$LOG"
echo "=== v3 finished $(date) ===" | tee -a "$LOG"
