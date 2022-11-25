#!/bin/bash
#SBATCH --job-name=mmflow
#SBATCH --partition=xgpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1

ulimit -s 10240
./eval_demo.sh


python demo/image_demo.py demo/frame_0001.png \
demo/frame_0002.png configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py \
/home/s.starace/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.pth results
