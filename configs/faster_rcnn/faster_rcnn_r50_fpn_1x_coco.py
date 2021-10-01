_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/voc07.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    train_cfg=dict(
        rcnn=dict(
            gmm = 3)),
)

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