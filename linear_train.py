import argparse

from mmcv.parallel import MMDataParallel
from mmcv import Config
import torch
import random
import numpy as np

from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="configs/resnet34_cifar10.py", type=str)
parser.add_argument('--gpu_ids', default=[0], type=list)
parser.add_argument('--workdir', default="", type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pretrained', default=None, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    config = Config.fromfile(args.config)

    if args.pretrained is not None:
        config['pretrained'] = args.pretrained
    
    dataset = data_loader(config)
    testsize = dataset.testsize()
    train_loader, test_loader = dataset.get_loader()

    logger = Logger(args, config)
    logger.print(f"{config.text}")
    logger.print(f"{dataset.transform_train}")
    logger.print(f"{dataset.transform_test}")
    saver = Saver(config, logger)

    if args.resume is None:
        print('==> Training from scratch..')
        ckpt = None
        start_epoch = -1
    else:
        print(f'==> Resuming from {args.resume}..')
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch']
        if "initalpha" in config['backbone']['conv_cfg']:
            config['backbone']['conv_cfg']['initalpha'] = False

    model = build_model(config)
    model = model.to(device)
    # model = MMDataParallel(model.to(device), device_ids=args.gpu_ids)
    
    optimizer = build_optimizer(config, model, ckpt)

    scheduler = build_lrscheduler(config, optimizer, start_epoch)
    total_epoch = config["total_epochs"]

    evaluate = evaluate_cls(model, device, test_loader, testsize, logger)

    for epoch in range(start_epoch + 1, total_epoch):
        epoch_train(model, epoch, device, train_loader, optimizer, scheduler, logger, saver)
        evaluate(epoch)
