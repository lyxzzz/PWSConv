from . import backbone
from . import head
from . import neck
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from mmcv.runner import load_checkpoint
import logging


backbone_dict = {}
for name in getattr(backbone, "__all__"):
    backbone_dict[name] = getattr(backbone, name)

head_dict = {}
for name in getattr(head, "__all__"):
    head_dict[name] = getattr(head, name)

neck_dict = {}
for name in getattr(neck, "__all__"):
    neck_dict[name] = getattr(neck, name)

class Model(nn.Module):
    def __init__(self, backbone, neck, head, pretrained):
        super(Model, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.backbone.init_weights()
        if self.neck is not None:
            self.neck.init_weights()
        
        if pretrained is not None:
            load_checkpoint(self, pretrained, strict=True)
        
        self.head = head
        self.head.init_weights()

        self.logger = logging.getLogger("myselfsup")
        self.logger.info("{}".format(self))
    
    def forward(self, x, label):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        out = self.head(out, label)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        if self.neck is not None:
            out = self.neck(out)
        return out

class SSModel(nn.Module):
    def __init__(self, backbone, neck, head, pretrained):
        super(SSModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()
        

        self.relu = nn.ReLU(inplace=True)
        
        if pretrained is not None:
            load_checkpoint(self, pretrained)
        
        self.logger = logging.getLogger("myselfsup")
        self.logger.info("{}".format(self))
    
    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z1 = self.relu(z1)

        z2 = self.backbone(x2)
        z2 = self.relu(z2)
        
        if self.neck is not None:
            z1 = self.neck(z1)
            z2 = self.neck(z2)

        out = self.head(z1, z2)
        return out
    
    def forward_knn(self, x):
        out = self.backbone(x)
        out = self.relu(out)
        if self.neck is not None:
            out = self.neck(out)
            out = self.relu(out)

        return out

def build_model(cfg):
    backbone_cfg = cfg.pop("backbone")
    backbone_type = backbone_cfg.pop("type")
    if backbone_type in backbone_dict:
        model_backbone = backbone_dict[backbone_type](**backbone_cfg)
    else:
        raise ValueError(f'backbone_type={backbone_type} does not support.')

    if "neck" in cfg:
        neck_cfg = cfg.pop("neck")
        neck_type = neck_cfg.pop("type")
        if neck_type in neck_dict:
            model_neck = neck_dict[neck_type](**neck_cfg)
        else:
            raise ValueError(f'neck_type={neck_type} does not support.')
    else:
        model_neck = None

    head_cfg = cfg.pop("head")
    head_type = head_cfg.pop("type")
    if head_type in head_dict:
        model_head = head_dict[head_type](**head_cfg)
    else:
        raise ValueError(f'backbone_type={head_type} does not support.')

    if "pretrained" in cfg:
        pretrained = cfg.pop("pretrained")
    else:
        pretrained = None
    model = Model(model_backbone, model_neck, model_head, pretrained)
    return model

def build_ssmodel(cfg):
    backbone_cfg = cfg.pop("backbone")
    backbone_type = backbone_cfg.pop("type")
    if backbone_type in backbone_dict:
        model_backbone = backbone_dict[backbone_type](**backbone_cfg)
    else:
        raise ValueError(f'backbone_type={backbone_type} does not support.')

    if "neck" in cfg:
        neck_cfg = cfg.pop("neck")
        neck_type = neck_cfg.pop("type")
        if neck_type in neck_dict:
            model_neck = neck_dict[neck_type](**neck_cfg)
        else:
            raise ValueError(f'neck_type={neck_type} does not support.')
    else:
        model_neck = None

    head_cfg = cfg.pop("head")
    head_type = head_cfg.pop("type")
    if head_type in head_dict:
        model_head = head_dict[head_type](**head_cfg)
    else:
        raise ValueError(f'backbone_type={head_type} does not support.')

    if "pretrained" in cfg:
        pretrained = cfg.pop("pretrained")
    else:
        pretrained = None
    model = SSModel(model_backbone, model_neck, model_head, pretrained)
    return model

def build_optimizer(cfg, model, ckpt=None):

    para_alpha = [p[1] for p in model.named_parameters() if 'alpha' in p[0]]
    paras = [p[1] for p in model.named_parameters() if not ('alpha' in p[0])]

    cfgoptimizer = cfg.pop("optimizer")
    opt_type = cfgoptimizer.pop("type")

    if "alpha_wd" in cfgoptimizer:
        weightdecay_alpha = cfgoptimizer.pop("alpha_wd")
    else:
        weightdecay_alpha = 0

    if opt_type == "SGD":
        optimizer = torch.optim.SGD([{'params': para_alpha, 'weight_decay': weightdecay_alpha}, {'params': paras}], **cfgoptimizer)
    else:
        raise ValueError(f'optimizer={opt_type} does not support.')

    if ckpt:
        print('==> Resuming optimizer..')
        optimizer.load_state_dict(ckpt['optimizer'])
        # print(optimizer.state_dict())
    return optimizer

def build_lrscheduler(cfg, optimizer, last_epoch=-1):
    lr_cfg = cfg.pop("lr_config")
    lr_policy = lr_cfg.pop("policy")

    if lr_policy == "step":
        scheduler = lr_scheduler.MultiStepLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    elif lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, last_epoch=last_epoch, **lr_cfg)
    else:
        raise ValueError(f'scheduler={lr_policy} does not support.')

    return scheduler
