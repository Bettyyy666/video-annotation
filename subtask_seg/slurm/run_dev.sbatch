#!/bin/bash
#SBATCH -J run_dev
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -n 6
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH -o logs/run_dev_%j.log
# 12 / 26 end to end as a batch job. From video-annotation/subtask_seg:
#   sbatch slurm/run_dev.sbatch [onset_shift]      (default -7; extra env e.g. SKIP_TRACK=1 is passed through)
set -eo pipefail
if [ "${CONDA_DEFAULT_ENV:-}" != annotation ]; then
  module load miniforge3/25.3.0-3 cuda
  source "${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh"
  conda activate annotation
fi
which python
nvidia-smi -L
bash /users/rzhou52/data/users/rzhou52/video-annotation/subtask_seg/run_dev.sh "${1:--7}"
