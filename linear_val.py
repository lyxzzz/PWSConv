import argparse

from mmcv.runner import load_checkpoint
from mmcv import Config
import torch
import random
import numpy as np

from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/resnet50_imagenet.py", type=str)
parser.add_argument('--ckpt', default="test.pth", type=str)
parser.add_argument('--linear', default=True, type=bool)

if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = Config.fromfile(args.config)
    
    dataset = data_loader(config)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config, save_file=False)
    
    if args.linear:
        model = build_model(config)
        print(f'==> evaluate from {args.ckpt}..')
        load_checkpoint(model, args.ckpt)
        model = model.to(device)

        evaluate_cls(model, device, test_loader, testsize, logger=logger)()
    else:
        model = build_ssmodel(config)
        print(f'==> evaluate from {args.ckpt}..')
        load_checkpoint(model, args.ckpt)
        model = model.to(device)

        evaluate_knn(model, device, test_loader, testsize, config, logger=logger)()