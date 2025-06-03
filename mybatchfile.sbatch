#!/bin/bash
#SBATCH -J big_baseline
#SBATCH -N 2
#SBATCH -n 12
#SBATCH --ntasks-per-node=12
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200000
#SBATCH --time=06:00:00
SBATCH --mail-user=albertolorente97@gmail.com

