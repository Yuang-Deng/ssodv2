_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='GMMShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            gmm_k=4,
            eta=12,
            lam_box_loss=1,
            cls_lambda=1,
            warm_epoch=12,
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
data_root = 'C:/Users/Alex/WorkSpace/dataset/voc/VOCdevkit/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            ),
            val=dict(
                ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
                img_prefix=data_root + 'VOC2007/',
            ),
            test=dict(
                ann_file=data_root + 'VOC2012/ImageSets/Main/trainval.txt',
                img_prefix=data_root + 'VOC2012/',
            )
)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='RoiEpochSetHook')]


# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[120000, 160000])
# runner = dict(type='IterBasedRunner', max_iters=180000)
# checkpoint_config = dict(interval=10000)
# evaluation = dict(interval=10000, metric='mAP')

# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[120000, 160000])

# # Runner type
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# runner = dict(_delete_=True, type='IterBasedRunner', max_iters=180000)

# checkpoint_config = dict(interval=10000)
# evaluation = dict(interval=10000, metric='bbox')