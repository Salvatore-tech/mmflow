_base_ = [
    '../_base_/models/raft.py', '../_base_/datasets/caddy_pwcnet.py',
    '/home/s.starace/FlowNets/mmflow/configs/_base_/schedules/pwcnet_plus_750k_schedule.py',
    '../_base_/default_runtime.py'

]

model = dict(
    decoder=dict(
        type='RAFTDecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_op_cfg=dict(type='CorrLookup', align_corners=True),
        gru_type='SeqConv',
        flow_loss=dict(type='SequenceLoss', gamma=0.85),
        act_cfg=dict(type='ReLU')),
    freeze_bn=True,
    test_cfg=dict(iters=32))

# model training and testing settings
train_cfg = dict(),
test_cfg = dict(),
init_cfg = dict(
    type='Kaiming',
    nonlinearity='leaky_relu',
    layer=['Conv2d', 'ConvTranspose2d'],
    mode='fan_in',
    bias=0)

# optimizer = dict(
#     type='AdamW',
#     lr=0.000125,
#     betas=(0.9, 0.999),
#     eps=1e-08,
#     weight_decay=0.00001,
#     amsgrad=False)
# optimizer_config = dict(grad_clip=dict(max_norm=1.))
# lr_config = dict(
#     policy='OneCycle',
#     max_lr=0.000125,
#     total_steps=100100,
#     pct_start=0.05,
#     anneal_strategy='linear')
#
# runner = dict(type='IterBasedRunner', max_iters=100000)
# checkpoint_config = dict(by_epoch=False, interval=5000)
# evaluation = dict(interval=5000, metric='EPE')

# Load model training on syntetic dataset and train on dCaddy
#load_from = 'checkpoints/raft/raft_8x2_100k_flyingthings3d_400x720.pth'  # noqa
resume_from = 'work_dir/raft_dCADDY_schedule_freezed_08/latest.pth'
# dist_params = dict(backend='nccl', port=29500)
