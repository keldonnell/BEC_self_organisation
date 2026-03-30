# Cluster Scripts Guide

This project stores cluster submission scripts in [`cluster/`](../cluster):

- `cluster/submit_smf_sweep.sh`
- `cluster/run_smf_sweep.sbatch`

`submit_smf_sweep.sh` launches a SLURM job array, and each array task runs one `p0` point via `gen_smf_1d.py`.

## SSH To The Cluster

Use exactly:

```bash
ssh <DS_USERNAME>@wildebeest.phys.strath.ac.uk
ssh phys-vole
```

## One-Time Setup On Cluster

1. Go to your project checkout:

```bash
cd /home/users/<DS_USERNAME>/.../BEC_self_organisation
```

2. Make sure scripts are executable:

```bash
chmod +x cluster/submit_smf_sweep.sh cluster/run_smf_sweep.sbatch
```

3. Ensure required environment exists (`smf_env` conda env, and SLURM available).

## Submit A Sweep

From project root on cluster:

```bash
./cluster/submit_smf_sweep.sh <P0_START> <P0_END> <N_INTERVALS>
```

Example:

```bash
./cluster/submit_smf_sweep.sh 2.2e-10 4.0e-10 40
```

This creates array indices `0..N_INTERVALS`, i.e. `N_INTERVALS + 1` jobs.

## What The SBATCH Script Does

`cluster/run_smf_sweep.sbatch`:

- loads conda module and activates `smf_env`
- reads array index (`SLURM_ARRAY_TASK_ID`)
- computes simulation index (`INDEX`)
- runs:

```bash
python <PROJECT_ROOT>/gen_smf_1d.py \
  -f ivan_diss_params_high_t_res \
  -s "$P0_START" -e "$P0_END" \
  -n "$((N_INTERVALS+1))" -i "$INDEX" \
  --extend-time-using-t0
```

## Monitor Jobs

```bash
squeue -u <DS_USERNAME>
```

Check logs in:

- `logs/<jobname>-<jobid>_<arrayid>.out`

## Common Workflow

1. SSH to `wildebeest`, then `phys-vole`.
2. `cd` to project root.
3. Pull/update code if needed.
4. Submit sweep with `./cluster/submit_smf_sweep.sh ...`.
5. Monitor via `squeue` and inspect `logs/`.
6. Analyse outputs under `patt1d_outputs/<filename>/`.
