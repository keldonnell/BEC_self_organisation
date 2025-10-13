#!/bin/bash -l
#SBATCH --job-name=smf_p0_sweep
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# --- Modules & env ---
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
module load applications/conda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smf_env

# --- Inputs from submit script ---
: "${P0_START:?Need P0_START}"
: "${P0_END:?Need P0_END}"
: "${N_INTERVALS:?Need N_INTERVALS}"
INDEX_OFFSET="${INDEX_OFFSET:-0}"

TID="${SLURM_ARRAY_TASK_ID:?missing array id}"
INDEX=$(( INDEX_OFFSET + TID ))

# --- Log what we're about to run ---
echo "JOB $SLURM_JOB_ID ARRAY_ID $TID  -> i=${INDEX} ; range=[${P0_START}, ${P0_END}] ; intervals=${N_INTERVALS}"

# --- Run job (your script computes p0 from s,e,n,i) ---
cd /home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation
mkdir -p logs

python gen_smf_1d.py \
  -f ivan_diss_params \
  -s "${P0_START}" \
  -e "${P0_END}" \
  -n "${N_INTERVALS}" \
  -i "${INDEX}"

