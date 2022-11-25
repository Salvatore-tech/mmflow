from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
import mmcv
import os

mmflow = os.environ['MMFLOW']
quality_result = os.environ['QUALITY_RESULT']

# Specify the path to model config and checkpoint file
config_file = mmflow + '/configs/raft/raft_kitti_test.py'
checkpoint_file = mmflow + '/checkpoints/raft/raft_8x2_50k_kitti2015_288x960.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test image pair, and save the results
img1 = quality_result + '/Sintel_final/im0/frame_0001.png'
img2 = quality_result+ '/Sintel_final/im1/frame_0002.png'
result = inference_model(model, img1, img2)

# save the optical flow file
#write_flow(result, flow_file='flow.flo')

# save the visualized flow map
flow_map = visualize_flow(result, save_file=quality_result + '/Sintel_final/predicted_KITTI15/' + 'flow_map.png')
