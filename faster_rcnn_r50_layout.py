model = dict(
    type='FasterRCNN',
    pretrained='/home/yinhan/epoch_50.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=None,
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHeadLayout',#<=======================
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,#<=======================
            reg_class_agnostic=False,
            with_reg = False,
            with_cls = False,
            )
            ))
test_cfg = dict(rpn=None,rcnn=dict())
