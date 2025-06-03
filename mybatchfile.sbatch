#!/bin/bash
#SBATCH -J my_job_name
#SBATCH -N 1
#SBATCH -n 9
#SBATCH --ntasks-per-node=9
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=01:00:00
