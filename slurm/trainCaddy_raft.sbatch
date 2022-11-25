#!/bin/bash
#SBATCH --job-name=sfineBothRAFT
#SBATCH --partition=xgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=4
#SBATCH --mem=30G
ulimit -s 10240
module load cuda/10.1

python $MMFLOW/tools/train.py $MMFLOW/configs/raft/raft_caddy_mix.py --work-dir $MMFLOW/work_dir/sfined_caddy_broad_then_dkitti
#python tools/train.py configs/raft/raft_caddy.py --work-dir work_dir/traindKITTI_schedule

