num_class = 1000

dataset = dict(
    type='ImageNet',
    train_root='/home/t/dataset/imagenet/train',
    train_list='imagenet_label/train_labeled.txt',
    test_root='/home/t/dataset/imagenet/val',
    test_list='imagenet_label/val_labeled.txt',
    batchsize=64,
    num_workers=16,
    num_class=num_class,
    mode="linear")

total_epochs = 100
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001, alpha_wd=0)
lr_config = dict(policy='step', milestones=[30, 60, 90])

pretrained = None

use_pws = False
norm_cfg = dict(type='BN', requires_grad=True)
conv_cfg = dict(type='conv')
neck_norm_cfg = None
zero_init_residual = False
if use_pws:
    norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
    conv_cfg = dict(type='pws', gamma=1e-4, equiv=False, initalpha=True, mode="fan_in")
    neck_norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
    zero_init_residual = False

backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        zero_init_residual=zero_init_residual)

neck=dict(
    type='ReluNeck',
    in_channels=2048,
    frozen_state=False,
    norm_cfg=neck_norm_cfg
)

head=dict(
    type='LinearClsHead',
    num_classes=num_class,
    in_channels=2048,
    topk=(1, 5),
)

logger = dict(interval=100)
saver = dict(interval=10)
