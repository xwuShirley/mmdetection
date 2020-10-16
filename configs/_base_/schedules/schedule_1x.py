# optimizer
optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
#load_from = "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth"
#'./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' # noqa