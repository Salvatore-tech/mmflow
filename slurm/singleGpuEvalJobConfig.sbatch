#!/bin/bash
#SBATCH --job-name=mmFlowKitti
#SBATCH --partition=xgpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1
ulimit -s 10240
module load cuda/10.1

python tools/test.py /home/s.starace/FlowNets/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py \
/home/s.starace/FlowNets/Models/mmlab/RAFT/raft_8x2_100k_mixed_368x768.pth \
--eval EPE

#python tools/test.py configs/pwcnet/pwcnet_ft_4x1_300k_kitti_320x896.py  checkpoints/pwcnet_ft_4x1_300k_kitti_320x896.pth  --eval EPE 1.63
#python tools/test.py configs/raft/raft_8x2_100k_mixed_368x768.py checkpoints/raft/raft_8x2_100k_mixed_368x768.pth --eval EPE 1.5936

python tools/test.py configs/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.py checkpoints/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth --eval EPE

python tools/test.py configs/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.py checkpoints/liteflownet2/liteflownet_ft_4x1_500k_kitti_320x896.pth --eval EPE
