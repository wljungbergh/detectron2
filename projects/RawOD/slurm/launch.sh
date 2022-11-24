#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --output /workspaces/%u/slurm/%j.out
#SBATCH -p ztest
#SBATCH --no-requeue
#
singularity exec --nv \
  -B /workspaces:/workspaces \
  -B /workspaces/s0001387/raw_od/detectron2:/home/appuser/detectron2_repo \
  -B /workspaces/s0001387/raw_od/pascal_raw:/datasets \
  /workspaces/s0001387/raw_od/detectron2.sif \
  python3 projects/RawOD/train.py \
#
#EOF