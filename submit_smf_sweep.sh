#!/usr/bin/env bash
# Usage: ./submit_smf_sweep.sh P0_START P0_END N_INTERVALS
set -euo pipefail
[ "$#" -ge 3 ] || { echo "Usage: $0 P0_START P0_END N_INTERVALS" >&2; exit 1; }

P0_START="$1"
P0_END="$2"
N_INTERVALS="$3"

cd /home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation
mkdir -p logs

# If N_INTERVALS = M (number of intervals), indices should be 0..M (inclusive) => M+1 total values.
ARRAY_SPEC="0-${N_INTERVALS}"

sbatch \
  --chdir=/home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation \
  --export=ALL,P0_START="${P0_START}",P0_END="${P0_END}",N_INTERVALS="${N_INTERVALS}" \
  --array="${ARRAY_SPEC}" \
  run_smf_sweep.sbatch

