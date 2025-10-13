#!/bin/bash -l
#SBATCH --job-name=smf_p0_sweep
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# Initialize Environment Modules in batch
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
module load applications/conda/miniconda3

# Poetry installed with --user? make sure PATH includes it
export PATH="$HOME/.local/bin:$PATH"
# If poetry isn't installed yet, do it once interactively:

: "${P0_START:?Need P0_START}"
: "${P0_END:?Need P0_END}"
: "${N_INTERVALS:?Need N_INTERVALS}"
INDEX_OFFSET="${INDEX_OFFSET:-0}"

TID="${SLURM_ARRAY_TASK_ID:?missing array id}"

P0_VAL=$(python3 - <<PY
start=float("${P0_START}")
end=float("${P0_END}")
m=int("${N_INTERVALS}")
tid=int("${TID}")
if not (0 <= tid <= m):
    raise SystemExit("Task id out of range")
p0 = start + (end - start) * (tid / m if m>0 else 0.0)
print(f"{p0:.10e}")
PY
)

INDEX=$(( INDEX_OFFSET + TID ))

cd /home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation
mkdir -p logs

# Run via Poetry
python3 gen_smf_1d.py \
  -f ivan_diss_params -n 1 \
  -s "${P0_VAL}" -e "${P0_VAL}" \
  -i "${INDEX}"

