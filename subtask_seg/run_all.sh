#!/usr/bin/env bash
# All videos that have gripper tracks + predictions (../runs/gripper/{tracks,pred}/<id>.json).
# From video-annotation/subtask_seg, in a GPU shell:   bash run_all.sh [onset_shift]
# A failing video is logged and skipped, not fatal.
SHIFT=${1:-0}
mkdir -p results
for p in ../runs/gripper/pred/*.json; do
  id=$(basename "$p" .json)
  v=../data/$id.mp4; [ -f "$v" ] || v=../data/train/$id.mp4
  python -u annotate.py "$v" --onset-shift "$SHIFT" 2>&1 | tee -a results/run_log.txt \
    || echo "FAILED $id" | tee -a results/run_log.txt
done
