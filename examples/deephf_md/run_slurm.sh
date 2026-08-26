#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: sbatch --array=0-8 run_slurm.sh ARCHIVE MODEL BASIS OUTPUT_ROOT" >&2
  exit 2
fi

project_root=$(cd "$(dirname "$0")/../.." && pwd)
archive=$(realpath "$1")
model=$2
basis=$3
mkdir -p "$4"
output_root=$(realpath "$4")
if [ "${model^^}" != "NONE" ]; then
  model=$(realpath "$model")
fi

work_directory=$(mktemp -d)
trap 'rm -rf "$work_directory"' EXIT
unzip -q "$archive" '*/samples/*/*.xyz' -d "$work_directory"
mapfile -t samples < <(find "$work_directory" -type f -name '*.xyz' | sort)
sample_index=${SLURM_ARRAY_TASK_ID:-0}
if [ "$sample_index" -ge "${#samples[@]}" ]; then
  echo "Sample index $sample_index is outside the ${#samples[@]} extracted samples" >&2
  exit 2
fi

sample=${samples[$sample_index]}
sample_name=$(basename "$sample" .xyz)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$project_root"
uv run python scripts/run_ase_md.py \
  "$sample" \
  --model "$model" \
  --basis "$basis" \
  --temperature-k 100 \
  --timestep-fs 0.02 \
  --steps 100 \
  --output-directory "$output_root/$sample_name"
