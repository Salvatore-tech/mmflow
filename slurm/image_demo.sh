#image_demo.py frame1.png frame2.png config.py checkpoint.pth output_dir
#python $MMFLOW/demo/image_demo.py
#$MMFLOW/demo/frame_0001.png \
#$MMFLOW/demo/frame_0002.png \
#$MMFLOW/configs/raft/raft_8x2_100k_mixed_368x768.py \
#$MMFLOW/checkpoints/raft_8x2_100k_mixed_368x768.pth \
#./launchResults

python /home/s.starace/Dataset/WIRN_22_work/im0/00001.png \
/home/s.starace/Dataset/WIRN_22_work/im1/00002.png \
configs/raft/raft_caddy.py \
work_dir/raft_dCADDY_mix_freezed_09/iter_40000.pth \
./launchResults/debug

#python $MMFLOW/demo/image_demo.py data/WIRN2022/scena_1_test/left/00001.png \
#data/WIRN2022/scena_1_test/right/00002.png \
#configs/raft/raft_8x2_100k_mixed_368x768.py \
#work_dir/raft_dCADDY_brodarski/latest.pth \
#./launchResults/raft_dCADDY_brodarski

#python demo/image_demo.py data/WIRN2022/scena_1_test/left/00001.png \
#data/WIRN2022/scena_1_test/right/00002.png \
#configs/pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768.py \
#work_dir/pwcnet_caddy_brodarski_D+B/latest.pth \
#./launchResults/pwcnet_caddy_brodarski_D+B
