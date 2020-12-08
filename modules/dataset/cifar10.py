import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from functools import partial
from mmcv.parallel import collate
import copy

from .build import MultiEpochsDataLoader
from .build import CudaDataLoader
from .build import DataPrefetcher

class SS_CIFAR(data.Dataset):
    def __init__(self, root, train, transform):
        super(SS_CIFAR, self).__init__()

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train)

        if isinstance(transform, list):
            self.pipeline1 = transform[0]
            self.pipeline2 = transform[1]
        else:
            self.pipeline1 = copy.deepcopy(transform)
            self.pipeline2 = transform

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        img, _ = self.dataset.__getitem__(idx)

        aug1 = self.pipeline1(img)
        aug2 = self.pipeline2(img)
        return aug1, aug2

class CIFAR10():
    def __init__(self, train_root, test_root, batchsize, num_workers, num_class=10, mode="linear"):
        self.img_norm_cfg = dict(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010))
            
        if mode == "linear":
            self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(**self.img_norm_cfg)])

            self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(**self.img_norm_cfg)])

            self.trainset = torchvision.datasets.CIFAR10(root=train_root, train=True, download=True, transform=self.transform_train)

            self.testset = torchvision.datasets.CIFAR10(root=test_root, train=False, download=True, transform=self.transform_test)

        elif mode == "selfsupervisied":
            self.transform_train = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply(
                                                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4,hue=0.1)], 
                                                    p=0.8),
                                                transforms.RandomGrayscale(0.2),
                                                # transforms.GaussianBlur(kernel_size=int(self.args.img_size * 0.1), sigma=(0.1, 2.0)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(**self.img_norm_cfg)])

            self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(**self.img_norm_cfg)])

            self.trainset = SS_CIFAR(root=train_root, train=True, transform=self.transform_train)

            self.testset = torchvision.datasets.CIFAR10(root=test_root, train=False, download=True, transform=self.transform_test)

        # self.train_loader = CudaDataLoader(
        #     MultiEpochsDataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True),
        #     "cuda")

        # self.test_loader = CudaDataLoader(
        #     MultiEpochsDataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True),
        #     "cuda")

        # self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True, 
        #                                               num_workers=num_workers, pin_memory=True)
        # self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, 
        #                                             num_workers=num_workers, pin_memory=True)

        self.train_loader = DataPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True, 
                                                        num_workers=num_workers, pin_memory=True))
        self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, 
                                                    num_workers=num_workers, pin_memory=True))

    def get_loader(self):
        return self.train_loader, self.test_loader

    def testsize(self):
        return self.testset.__len__()