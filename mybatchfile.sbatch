#!/bin/bash
#SBATCH -J big_baseline
#SBATCH -N 1
#SBATCH -n 9
#SBATCH --ntasks-per-node=9
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=06:00:00
