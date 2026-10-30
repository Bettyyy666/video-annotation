#!/bin/bash
#SBATCH -J annotate
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o logs/annotate_%j.out
# From video-annotation/subtask_seg:   sbatch slurm/annotate.sbatch 12_fold_four_towels [onset_shift]
# (gripper tracks + predictions for the video must exist, see README)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate annotation
ID=$1
SHIFT=${2:-0}
python -u annotate.py ../data/$ID.mp4 --onset-shift "$SHIFT"
