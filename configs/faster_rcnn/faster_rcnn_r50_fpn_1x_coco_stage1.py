_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='GMMShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            gmm_k=4,
            eta=12,
            lam_box_loss=1,
            cls_lambda=1,
            warm_epoch=12,
            unc_type='ep',
            lambda_unc=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        label_type2weight=[1,2,2]
    )
)
data_root = 'C:/Users/Alex/WorkSpace/dataset/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1024, 800), (1024, 500)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
percent = 10
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
            ann_file=[
                data_root + 'annotations/semi_supervised/instances_train2017.1@'+str(percent)+'.json',
            ],
            img_prefix=[data_root + 'train2017/', data_root + 'VOC2012/'],
            ),
            val=dict(
                ann_file=data_root + 'annotations/instances_val2017.json',
                img_prefix=data_root + 'val2017/',
            ),
            test=dict(
                ann_file=data_root + 'annotations/semi_supervised/instances_train2017.1@'+str(percent)+'-unlabeled.json',
                img_prefix=data_root + 'train2017/',
                pipeline=test_pipeline
            )
)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=13)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='RoiEpochSetHook')]