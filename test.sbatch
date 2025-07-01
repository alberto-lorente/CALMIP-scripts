#!/bin/bash
#SBATCH --job-name=multinode-example
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4

set -x
sleep 10

export MASTER_PORT=$(echo "${SLURM_JOB_ID} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
export LOGLEVEL=DEBUG

echo "HOSTNAME: $(hostname)"
echo "NODES : ${SLURM_JOB_NODELIST}"

srun apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif torchrun \
--nnodes ${SLURM_NNODES} \
--nproc_per_node 2 \
--rdzv_id ${RANDOM} \
--rdzv_backend c10d \
--rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
./multinode.py 2000 1000


