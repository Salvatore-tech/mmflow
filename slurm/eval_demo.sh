python $MMFLOW/tools/test.py $MMFLOW/configs/raft/raft_8x2_50k_wirn22.py \
$MMFLOW/checkpoints/raft/raft_8x2_100k_mixed_368x768.pth --eval EPE Fl

# Evaluate liteflownet2 on kitti2015 trained on Flying Chairs + Flying Thing3d subset + Sintel + KITTI
#python $MMFLOW/tools/test.py $MMFLOW/configs/liteflownet2/liteflownet2_ft_4x1_500k_kitti_320x896.py $MMFLOW/checkpoints/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth --eval EPE