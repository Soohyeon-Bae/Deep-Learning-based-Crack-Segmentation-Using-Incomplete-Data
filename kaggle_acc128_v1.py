_norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# dataset

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        pipeline=train_pipeline,
        dataset=dict(
            type='CustomDataset',
            data_root='dataset/exp1/kaggle_acc',
            pipeline=train_pipeline,
            split=None,
            img_dir='img',
            ann_dir='mask',
            img_suffix='.png',
            seg_map_suffix='.png',
            classes=('background', 'crack'),
            palette=[[0, 0, 0], [1, 1, 1]],
           )),
    val=dict(
        type='CustomDataset',
        data_root='dataset/exp1/val',
        pipeline=test_pipeline,
        split=None,
        img_dir='img',
        ann_dir='mask',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [1, 1, 1]],
        ),
    test=dict(
        type='CustomDataset',
        data_root='dataset/exp1/test',
        pipeline=test_pipeline,
        split=None,
        img_dir='img',
        ann_dir='mask',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [255, 255, 255]],
        ))

# default run time
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True

# schedule
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=1)
load_from = None
evaluation = dict(by_epoch=True, interval=1, metric='mIoU')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
                                      dict(type='WandbLoggerHook',
                                           init_kwargs=dict(project='crack_segmentation', name='accurate'))])
work_dir = './work_dirs/crack_segmentation/accurate'