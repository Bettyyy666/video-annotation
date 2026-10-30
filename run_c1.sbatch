#!/bin/bash
#SBATCH -J run_c1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o run_c1_%j.log
# Full pipeline: train c1 -> classifier lag -> track 12, 26 -> JSON -> compare with GT.
# From video-annotation/:   sbatch run_c1.sbatch
set -eo pipefail
cd /users/rzhou52/data/users/rzhou52/video-annotation
# run as `bash run_c1.sbatch` from a shell with annotation already active, or activate it here
if [ "${CONDA_DEFAULT_ENV:-}" != annotation ]; then
  module load miniforge3/25.3.0-3 cuda
  source "${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh"
  conda activate annotation
fi
which python
nvidia-smi -L

python -u -m gripper.state export
python -u -m gripper.state train --name c1
python -u -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt
python -u -m gripper.track video --videos 12_fold_four_towels.mp4 26_sort_cutlery_drawer_insert.mp4
python -u -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt \
    --videos 12_fold_four_towels.mp4 26_sort_cutlery_drawer_insert.mp4 --vis
cd subtask_seg
python -u tools/measure_lag.py | tee lag.txt
SHIFT=$(grep -o 'ONSET_SHIFT = -\?[0-9]*' lag.txt | grep -o -- '-\?[0-9]*$' || echo 0)
echo "using ONSET_SHIFT=$SHIFT"
for v in 12_fold_four_towels 26_sort_cutlery_drawer_insert; do
  python -u annotate.py ../data/$v.mp4 --onset-shift "$SHIFT"
done
python -u tools/eval_gt.py results ../data
