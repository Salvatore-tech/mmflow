#!/bin/bash
#SBATCH --job-name=liteFlowNet_finetuning
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
ulimit -s 10240
module load cuda/10.1

python tools/train.py configs/liteflownet2/liteflownet2_kitti_2015_augmented.py --work-dir=work_dir/liteflownet2_aug