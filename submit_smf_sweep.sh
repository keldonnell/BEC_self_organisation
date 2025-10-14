#!/usr/bin/env bash
# Usage: ./submit_smf_sweep.sh P0_START P0_END N_INTERVALS [--density-centers c1,c2] [--density-width width] [--density-strength strength]
set -euo pipefail
[ "$#" -ge 3 ] || { echo "Usage: $0 P0_START P0_END N_INTERVALS" >&2; exit 1; }

P0_START="$1"
P0_END="$2"
N_INTERVALS="$3"
shift 3

DENSITY_CENTERS=""
DENSITY_WIDTH=""
DENSITY_STRENGTH=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --density-centers)
      shift
      [ "$#" -gt 0 ] || { echo "Error: --density-centers requires a value (comma or colon separated list)." >&2; exit 1; }
      DENSITY_CENTERS="$1"
      ;;
    --density-width)
      shift
      [ "$#" -gt 0 ] || { echo "Error: --density-width requires a numeric value." >&2; exit 1; }
      DENSITY_WIDTH="$1"
      ;;
    --density-strength)
      shift
      [ "$#" -gt 0 ] || { echo "Error: --density-strength requires a numeric value." >&2; exit 1; }
      DENSITY_STRENGTH="$1"
      ;;
    *)
      echo "Unrecognised argument: $1" >&2
      echo "Usage: $0 P0_START P0_END N_INTERVALS [--density-centers c1,c2] [--density-width width] [--density-strength strength]" >&2
      exit 1
      ;;
  esac
  shift
done

export P0_START P0_END N_INTERVALS
export DENSITY_CENTERS DENSITY_WIDTH DENSITY_STRENGTH

cd /home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation
mkdir -p logs

# If N_INTERVALS = M (number of intervals), indices should be 0..M (inclusive) => M+1 total values.
ARRAY_SPEC="0-${N_INTERVALS}"

sbatch \
  --chdir=/home/users/seb25178/Projects/BEC_SMF/BEC_self_organisation \
  --export=ALL \
  --array="${ARRAY_SPEC}" \
  run_smf_sweep.sbatch
