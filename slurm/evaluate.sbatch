#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=xgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=4
#SBATCH --mem=30G
ulimit -s 10240
module load cuda/10.1

# RAFT EVALUATION

# TRAINED ON: CHAIRS+THINGS
  #  python tools/test.py configs/raft/raft_8x2_50k_kitti2015_288x960.py checkpoints/raft/raft_8x2_100k_flyingthings3d_400x720.pth --eval EPE Fl

# TRAINED ON: CHAIRS+THINGS+dCADDY (4432 imgs)
	# python tools/test.py configs/raft/raft_8x2_100k_flyingthings3d_400x720.py work_dir/raft_dCADDY_brodarski/iter_10000.pth --eval EPE Fl

# TRAINED ON: CHAIRS+THINGS + finetuned on dCADDY (10k imgs) - freezed last layer weights
 # python tools/test.py configs/raft/raft_caddy.py work_dir/raft_dCADDY_freezed_weightsAndBias/latest.pth --eval EPE Fl

# TRAINED ON :Chairs + Things3D_subset + dMix (freeze weights)
 # python tools/test.py configs/raft/raft_caddy.py work_dir/raft_dCADDY_mix_freezed_09/iter_40000.pth --eval EPE Fl
 
# TRAINED ON: Chairs + Things3D_subset + dKITTI hotswapped by dCaddy partition
python tools/test.py configs/raft/raft_caddy.py work_dir/s_fine_seq_last/latest.pth --eval EPE Fl
