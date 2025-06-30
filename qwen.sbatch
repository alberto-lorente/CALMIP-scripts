#!/bin/bash
#SBATCH -J qwen
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --mem=20000
#SBATCH --time=12:00:00

module load conda/24.11.1
conda activate hate

python3 Qwen_from_expl_to_impl.py