#!/bin/bash
#SBATCH -J fbert-ewc
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-node=36
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2
#SBATCH --mem=20000
#SBATCH --time=12:00:00

module load conda/24.11.1
conda activate hate

echo "$PWD"
mv initial_cl_fbert_agem_hp.py ..
mv initial_cl_fbert_mas.py ..
cd ..


python3 initial_cl_fbert_agem_hp.py
python3 initial_cl_fbert_mas.py