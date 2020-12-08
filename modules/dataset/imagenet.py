import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from .dataset_type import ImageList
from .build import DataPrefetcher
from .build import MultiEpochsDataLoader
from .build import CudaDataLoader

META_FILE = "meta.bin"

class ImageNet_Dataset(data.Dataset):
    def __init__(self, root, transforms, list_file, num_class, preload=False):
        super(ImageNet_Dataset, self).__init__()
        self.root = root
        self.list_file = list_file
        self.num_class = num_class

        self.imagelist = ImageList(root, list_file, num_class, preload)

        self.pipeline = transforms

        if self.imagelist.has_labels:
            self.item_func = self.__get_labeled_item
        else:
            self.item_func = self.__get_unlabeled_item

    def __len__(self):
        return self.imagelist.get_length()
    
    def __get_labeled_item(self, idx):
        img, target = self.imagelist.get_sample(idx)
        if self.pipeline is not None:
            img = self.pipeline(img)
        
        return img, target

    def __get_unlabeled_item(self, idx):
        img = self.imagelist.get_sample(idx)
        target = torch.ones(1)
        if self.pipeline is not None:
            if isinstance(self.pipeline, list):
                img1 = self.pipeline[0](img).unsqueeze(0)
                img2 = self.pipeline[1](img).unsqueeze(0)
                img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
                return img_cat, target
            else:
                img = self.pipeline(img)
                return img, target
    
    def __getitem__(self, idx):
        return self.item_func(idx)

class ImageNet():
    def __init__(self, train_root, test_root, train_list, test_list, batchsize, num_workers, num_class=1000, mode="linear"):
        self.img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])

        if mode == "linear":
            self.transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(**self.img_norm_cfg)])

            self.transform_test = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(**self.img_norm_cfg)])

            self.trainset = ImageNet_Dataset(train_root, self.transform_train, train_list, num_class, preload=False)

            self.testset = ImageNet_Dataset(test_root, self.transform_test, test_list, num_class, preload=False)
        
            # self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True)

            # self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            self.train_loader = DataPrefetcher(torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers))

            self.test_loader = DataPrefetcher(torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=num_workers))

            # self.train_loader = CudaDataLoader(
            #     MultiEpochsDataLoader(self.trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True),
            #     "cuda", queue_size=8)
            
            # self.test_loader = CudaDataLoader(
            #     MultiEpochsDataLoader(self.testset, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True),
            #     "cuda", queue_size=8)

    def get_loader(self):
        return self.train_loader, self.test_loader

    def testsize(self):
        return self.testset.__len__()
    
    def trainsize(self):
        return self.trainset.__len__()
