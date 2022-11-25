#!/bin/bash
#SBATCH --job-name=finetuneDKITTI
#SBATCH --partition=xgpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
ulimit -s 10240
module load cuda/10.1

python tools/train.py configs/pwcnet/pwcnet_kitti_augmented.py --work-dir work_dir/pwcnetKitti_finetuned_kittiaug --no-validate
