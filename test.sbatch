#!/bin/bash
#SBATCH -J mon_job
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif python mon_script.py