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
DENSITY_CENTERS="${DENSITY_CENTERS:-}"
DENSITY_WIDTH="${DENSITY_WIDTH:-}"
DENSITY_STRENGTH="${DENSITY_STRENGTH:-}"

if ! [[ "${N_INTERVALS}" =~ ^[0-9]+$ ]]; then
  echo "N_INTERVALS must be a non-negative integer, got '${N_INTERVALS}'" >&2
  exit 1
fi

NUM_PUMP_FRAMES=$(( N_INTERVALS + 1 ))

TID="${SLURM_ARRAY_TASK_ID:?missing array id}"
INDEX=$(( INDEX_OFFSET + TID ))

# --- Log what we're about to run ---
echo "JOB $SLURM_JOB_ID ARRAY_ID $TID  -> i=${INDEX} ; range=[${P0_START}, ${P0_END}] ; intervals=${N_INTERVALS} (frames=${NUM_PUMP_FRAMES})"
if [[ -n "${DENSITY_CENTERS}" || -n "${DENSITY_WIDTH}" || -n "${DENSITY_STRENGTH}" ]]; then
  echo "Density options: centers='${DENSITY_CENTERS:-}' width='${DENSITY_WIDTH:-}' strength='${DENSITY_STRENGTH:-}'"
fi

# --- Run job (your script computes p0 from s,e,n,i) ---
cd /home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation
mkdir -p logs

DENSITY_ARGS=()
if [[ -n "${DENSITY_CENTERS}" ]]; then
  centres_clean="${DENSITY_CENTERS//,/ }"
  centres_clean="${centres_clean//:/ }"
  centres_array=()
  if [[ -n "${centres_clean// /}" ]]; then
    read -r -a centres_array <<< "${centres_clean}"
  fi
  if [ "${#centres_array[@]}" -gt 0 ]; then
    valid_centres=()
    for centre in "${centres_array[@]}"; do
      centre_trimmed="${centre#"${centre%%[![:space:]]*}"}"
      centre_trimmed="${centre_trimmed%"${centre_trimmed##*[![:space:]]}"}"
      if [[ -n "${centre_trimmed}" ]]; then
        valid_centres+=( "${centre_trimmed}" )
      fi
    done
    if [ "${#valid_centres[@]}" -gt 0 ]; then
      DENSITY_ARGS+=( "--density-centers" )
      DENSITY_ARGS+=( "${valid_centres[@]}" )
    fi
  fi
fi

if [[ -n "${DENSITY_WIDTH}" ]]; then
  DENSITY_ARGS+=( "--density-width" "${DENSITY_WIDTH}" )
fi

if [[ -n "${DENSITY_STRENGTH}" ]]; then
  DENSITY_ARGS+=( "--density-strength" "${DENSITY_STRENGTH}" )
fi

python gen_smf_1d.py \
  -f ivan_diss_params \
  -s "${P0_START}" \
  -e "${P0_END}" \
  -n "${NUM_PUMP_FRAMES}" \
  -i "${INDEX}" \
  "${DENSITY_ARGS[@]}"
