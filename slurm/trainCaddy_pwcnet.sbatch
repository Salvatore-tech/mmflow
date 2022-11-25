#!/bin/bash
#SBATCH --job-name=trainCaddy
#SBATCH --partition=xgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --ntasks=2
#SBATCH --mem=30G
ulimit -s 10240
module load cuda/10.1

python tools/train.py configs/pwcnet/pwcnet_caddy.py --work-dir work_dir/pwcnet_dCADDY_freezed

