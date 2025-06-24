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

mv initial_cl_fbert_agem.py ..
mv initial_cl_fbert_mas.py ..
mv initial_cl_fbert_ewc.py ..
cd ..


python3 initial_cl_fbert_agem.py
python3 initial_cl_fbert_mas.py
python3 initial_cl_fbert_ewc.py