# weights:
	wKITTI_2015="$MMFLOW/checkpoints/raft/raft_8x2_50k_kitti2015_288x960.pth"
	wsyntetics="$MMFLOW/checkpoints/raft/raft_8x2_100k_flyingthings3d_400x720.pth"
	wdCADDY="$MMFLOW/work_dir/raft_dCADDY_freezed_07/iter_40000.pth"
	wdMix="$MMFLOW/work_dir/raft_dCADDY_mix_freezed_09/iter_40000.pth"

sut="/WIRN22/"
weights=$wsyntetics
dest_dir="predicted_KITTI15"

# python $MMFLOW/demo/image_demo.py \
# "$QUALITY_RESULT$sut/im0/00001.png" \
# "$QUALITY_RESULT$sut/im1/00002.png" \
# $MMFLOW/configs/raft/raft_kitti_test.py \
# "$weights" \
# "$QUALITY_RESULT$sut$dest_dir" \
# --out_prefix flow_frames_s1p1 \
#
# python $MMFLOW/demo/image_demo.py \
# "$QUALITY_RESULT$sut/im0/10001.png" \
# "$QUALITY_RESULT$sut/im1/10002.png" \
# $MMFLOW/configs/raft/raft_kitti_test.py \
# "$weights" \
# "$QUALITY_RESULT$sut$dest_dir" \
# --out_prefix flow_frames_s2p1 \
#
# python $MMFLOW/demo/image_demo.py \
# "$QUALITY_RESULT$sut/im0/20001.png" \
# "$QUALITY_RESULT$sut/im1/20002.png" \
# $MMFLOW/configs/raft/raft_kitti_test.py \
# "$weights" \
# "$QUALITY_RESULT$sut$dest_dir" \
# --out_prefix flow_frames_s3p1 \

python $MMFLOW/demo/image_demo.py \
"$QUALITY_RESULT$sut/im0/30001.png" \
"$QUALITY_RESULT$sut/im1/30002.png" \
$MMFLOW/configs/raft/raft_kitti_test.py \
"$weights" \
"$QUALITY_RESULT$sut$dest_dir" \
--out_prefix flow_frames_s4p1 \

# python $MMFLOW/demo/image_demo.py \
# "$QUALITY_RESULT$sut/im0/40001.png" \
# "$QUALITY_RESULT$sut/im1/40002.png" \
# $MMFLOW/configs/raft/raft_kitti_test.py \
# "$weights" \
# "$QUALITY_RESULT$sut$dest_dir" \
# --out_prefix flow_frames_s5p1 \
