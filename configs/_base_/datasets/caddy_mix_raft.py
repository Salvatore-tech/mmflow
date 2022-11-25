wirn22_data_root = '/home/s.starace/Dataset/WIRN_22_work'
wirn22_dataset_type = 'WIRN22'
wirn22_img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)
crop_size_wirn22 = (960, 540)

caddy_data_root = '/home/s.starace/Dataset/Depthstillation_mix/dCADDY'
caddy_dataset_type = 'CADDY'
caddy_img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)
crop_size_caddy = (320, 240)

kitti_data_root = './data/KITTI_AUG'
kitti_dataset_type = 'KITTI2015AUG'
kitti_img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
crop_size_kitti = (288, 960)

global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

kitti_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.0,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5 / 3.14),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=crop_size_kitti,
        min_scale=-0.2,
        max_scale=0.4,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size_kitti),
    dict(type='Normalize', **caddy_img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]

caddy_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='Normalize', **caddy_img_norm_cfg),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=global_transform,
        relative_transform=relative_transform),
    dict(type='RandomCrop', crop_size=crop_size_caddy),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='InputResize', exponent=6),
    dict(type='Normalize', **caddy_img_norm_cfg),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape'
        ])
]

sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(type='Normalize', **caddy_img_norm_cfg),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape', 'pad'
        ])
]

d_caddy_train = dict(
    type=caddy_dataset_type,
    data_root=caddy_data_root,
    pipeline=caddy_train_pipeline,
    test_mode=False)

d_kitti_train = dict(
    type=kitti_dataset_type,
    data_root=kitti_data_root,
    pipeline=kitti_train_pipeline,
    test_mode=False)

kitti2015_val_test = dict(
    type='KITTI2015',
    data_root='data/KITTI_2015',
    pipeline=test_pipeline,
    test_mode=True)

wirn22_val_test = dict(
    type=wirn22_dataset_type,
    data_root=wirn22_data_root,
    pipeline=test_pipeline,
    test_mode=True)

sintel_clean_test = dict(
    type='Sintel',
    pipeline=sintel_test_pipeline,
    data_root='data/Sintel',
    test_mode=True,
    pass_style='clean')

sintel_final_test = dict(
    type='Sintel',
    pipeline=sintel_test_pipeline,
    data_root='data/Sintel',
    test_mode=True,
    pass_style='final')

data = dict(
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=5,
        drop_last=True,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[
        d_kitti_train
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[
            kitti2015_val_test,
           # sintel_clean_test, sintel_final_test
            # wirn22_val_test
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            # wirn22_val_test
            kitti2015_val_test,
            #sintel_clean_test, sintel_final_test
        ],
        separate_eval=True)
)

