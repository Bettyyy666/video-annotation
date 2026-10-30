#!/usr/bin/env bash
# 12 / 26 end to end (GPU node, conda env `annotation`). From anywhere:
#   nohup bash subtask_seg/run_dev.sh -7 > subtask_seg/logs/run_dev.log 2>&1 &
#   SKIP_TRACK=1 bash subtask_seg/run_dev.sh -7      # gripper tracks / preds already up to date
# $1 = onset shift in frames (classifier lag, tools/measure_lag.py)
set -euo pipefail
SHIFT=${1:--7}
IDS="12_fold_four_towels 26_sort_cutlery_drawer_insert"
cd "$(dirname "$0")/.."                      # video-annotation/
export PYTHONUNBUFFERED=1
if [ -z "${SKIP_TRACK:-}" ]; then
  echo "== [A] SAM 2 tracking (mask prompts + size guard)  $(date)"
  python -m gripper.track video --videos $(for i in $IDS; do printf "%s.mp4 " "$i"; done)
  echo "== [A] open/closed classifier c1  $(date)"
  python -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt \
    --videos $(for i in $IDS; do printf "%s.mp4 " "$i"; done) --vis
  python -m gripper.track report || true
fi
cd subtask_seg
mkdir -p logs results
python tests/synthetic_check.py
for id in $IDS; do
  echo "== [B-D] $id  $(date)"
  python annotate.py ../data/$id.mp4 --onset-shift "$SHIFT"
done
python tools/eval_gt.py results ../data
echo "== done  $(date)"
