#!/bin/bash
#SBATCH -J big_baseline
#SBATCH -N 1
#SBATCH -n 9
#SBATCH --ntasks-per-node=9
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200000
#SBATCH --time=06:00:00
#SBATCH --mail-user=albertolorente97@gmail.com

