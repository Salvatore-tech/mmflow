#!/bin/bash
#SBATCH --job-name=liteflownet2_kitti
#SBATCH --nodes=1
#SBATCH --partition=xgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
ulimit -s 10240
module load cuda/10.1

$MMFLOW/fineTuneKitti.sh