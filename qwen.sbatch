#!/bin/bash
#SBATCH -J qwen
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --mem=20000
#SBATCH --time=12:00:00

# set -x

ENV_PATH=/tmpdir/gale/.conda/envs/hate

export MASTER_PORT=$(echo "${SLURM_JOB_ID} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
export LOGLEVEL=DEBUG

echo "HOSTNAME: $(hostname)"
echo "NODES : ${SLURM_JOB_NODELIST}"


srun apptainer exec \
    --env PYTHONUSERBASE=$ENV_PATH \
    --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
    torchrun \
        --nnodes ${SLURM_NNODES} \
        --nproc_per_node=4 \
        --rdzv_id ${RANDOM} \
        --rdzv_backend=c10d \
        --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        .Qwen_from_expl_to_impl.py