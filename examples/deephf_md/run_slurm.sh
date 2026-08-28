#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 10 ]; then
  echo "Usage: sbatch --array=... run_slurm.sh SAMPLE_SOURCE MODEL BASIS OUTPUT_ROOT [STEPS] [TIMESTEP_FS] [TEMPERATURE_K] [SEED] [FORCE_MODE] [FINITE_DIFFERENCE_STEP_BOHR]" >&2
  exit 2
fi

project_root=$(realpath "${SLURM_SUBMIT_DIR:-$(pwd)}")
sample_source=$(realpath "$1")
model=$2
basis=$3
mkdir -p "$4"
output_root=$(realpath "$4")
steps=${5:-400}
timestep_fs=${6:-0.25}
temperature_k=${7:-100}
seed=${8:-20260821}
force_mode=${9:-analytic}
finite_difference_step_bohr=${10:-1e-4}
if [ "${model^^}" != "NONE" ]; then
  model=$(realpath "$model")
fi

work_directory=""
if [ -d "$sample_source" ]; then
  mapfile -t samples < <(find "$sample_source" -type f -name '*.xyz' | sort)
else
  work_directory=$(mktemp -d)
  trap 'rm -rf "$work_directory"' EXIT
  unzip -q "$sample_source" '*/samples/*/*.xyz' -d "$work_directory"
  mapfile -t samples < <(find "$work_directory" -type f -name '*.xyz' | sort)
fi
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
  --reference-family rks \
  --xc B3LYP5 \
  --grid-mode default \
  --grid-level 3 \
  --small-rho-cutoff 0 \
  --scf-conv-tol 1e-10 \
  --scf-conv-tol-grad 1e-7 \
  --temperature-k "$temperature_k" \
  --timestep-fs "$timestep_fs" \
  --steps "$steps" \
  --seed "$seed" \
  --force-mode "$force_mode" \
  --finite-difference-step-bohr "$finite_difference_step_bohr" \
  --output-directory "$output_root/$sample_name"
