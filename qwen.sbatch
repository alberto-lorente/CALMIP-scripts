#!/bin/bash
#SBATCH -J qwen
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --time=18:00:00

# According to Olympe's documentation for PyTorch,
# and assuming "hate" is a clone of our your PyTorch environment that includes torchrun:

module load conda/24.11.1
conda activate hate

# set -x

# Based on Turpan's documentation for PyTorch multi-GPU/multinode setup

export MASTER_PORT=$(echo "${SLURM_JOB_ID} % 100000 % 50000 + 10001" | bc)
export MASTER_ADDR=$(hostname --ip-address)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
export LOGLEVEL=DEBUG

echo "HOSTNAME: $(hostname)"
echo "NODES : ${SLURM_JOB_NODELIST}"

torchrun \
        --nnodes ${SLURM_NNODES} \
        --nproc_per_node=4 \
        --rdzv_id ${RANDOM} \
        --rdzv_backend=c10d \
        --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
        Qwen_from_expl_to_impl.py


# torchrun \
#         --nnodes ${SLURM_NNODES} \
#         --nproc_per_node=4 \
#         --rdzv_id ${RANDOM} \
#         --rdzv_backend=c10d \
#         --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
#         Qwen_from_expl_to_impl.py -> change to the other script

# torchrun \
#         --nnodes ${SLURM_NNODES} \
#         --nproc_per_node=4 \
#         --rdzv_id ${RANDOM} \
#         --rdzv_backend=c10d \
#         --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \

# torchrun \
#         --nnodes ${SLURM_NNODES} \
#         --nproc_per_node=4 \
#         --rdzv_id ${RANDOM} \
#         --rdzv_backend=c10d \
#         --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
#         Qwen_from_expl_to_impl.py -> change to the other script
