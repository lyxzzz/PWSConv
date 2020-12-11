num_class = 10

dataset = dict(
    type='CIFAR10',
    train_root='/home/t/dataset/cifar10',
    test_root='/home/t/dataset/cifar10',
    num_workers=16,
    batchsize=128,
    num_class=num_class)

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, alpha_wd=0)
lr_config = dict(policy='step', milestones=[100, 150])
total_epochs = 200

pretrained = None

use_pws = True
norm_cfg = dict(type='BN', requires_grad=True)
conv_cfg = dict(type='conv')
neck_norm_cfg = None
zero_init_residual = True
if use_pws:
    norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
    conv_cfg = dict(type='pws', gamma=1e-3, equiv=False, initalpha=False, mode="fan_out")
    neck_norm_cfg = dict(type='GN', num_groups=1, requires_grad=True)
    zero_init_residual = False

backbone=dict(
        type='ResNet_Cifar',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        zero_init_residual=zero_init_residual)

neck=dict(
    type='ReluNeck',
    in_channels=512,
    frozen_state=False,
    norm_cfg=neck_norm_cfg
)
        
head=dict(
    type='LinearClsHead',
    num_classes=num_class,
    in_channels=512,
    topk=(1, 5),
)

knn=dict(    
    l2norm=True,
    topk_percent=0.2
)

logger = dict(interval=100)
saver = dict(interval=20)
