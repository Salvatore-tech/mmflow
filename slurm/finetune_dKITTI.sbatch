#!/bin/bash
#SBATCH --job-name=finetuneDKITTI
#SBATCH --partition=xgpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:tesla:4

ulimit -s 10240
module load cuda/10.1

python tools/train.py configs/pwcnet/pwcnet_finetune_dkitti.py --work-dir work_dir/pwcnetKitti_750k_schedule --no-validate

#GPUS=16 sh tools/slurm_train.sh xgpu finetuneSchedule configs/pwcnet/pwcnet_finetune_dkitti.py work_dir/pwcnetKitti_750k_schedule --no-validate
