#!/usr/bin/env bash
# Wait for the running training to release the GPU, then produce everything that
# should be ready to look at: the report figure, the test predictions, and the
# submission verification. Meant to be launched alongside a long unattended run.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
TAG=spine_unet_r34
LOG=after_training.log

echo "=== waiting for training to finish ===" | tee "$LOG"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
  sleep 60
done
echo "GPU free at $(date)" | tee -a "$LOG"

# let the card cool before another GPU job
while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -gt 68 ]; do
  sleep 30
done
echo "cooled to $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" | tee -a "$LOG"

BEST=checkpoints/${TAG}_best.pt
if [ ! -f "$BEST" ]; then
  echo "no best checkpoint at $BEST -- falling back to last" | tee -a "$LOG"
  BEST=checkpoints/${TAG}_last.pt
fi

echo "=== threshold calibration ===" | tee -a "$LOG"
$PY -u src/spine_calibrate.py --ckpt "$BEST" --tag "$TAG" --limit 25 2>&1 | tee -a "$LOG"

echo "=== report figure ===" | tee -a "$LOG"
$PY -u src/spine_report.py --ckpt "$BEST" --tag "$TAG" --limit 20 2>&1 | tee -a "$LOG"

echo "=== test predictions ===" | tee -a "$LOG"
CAL=calibration_${TAG}.json
ST=$($PY -c "import json;print(json.load(open('$CAL'))['best_spine_thresh'])" 2>/dev/null || echo 0.4)
MP=$($PY -c "import json;print(json.load(open('$CAL'))['best_min_spine_px'])" 2>/dev/null || echo 12)
echo "using spine_thresh=$ST min_spine_px=$MP" | tee -a "$LOG"
$PY -u src/spine_predict.py --ckpt "$BEST" --out submission.csv \
    --spine-thresh "$ST" --min-spine-px "$MP" 2>&1 | tee -a "$LOG"

echo "=== submission verification ===" | tee -a "$LOG"
$PY -u src/verify_submission.py submission.csv \
    --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images 2>&1 | tee -a "$LOG"

echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
