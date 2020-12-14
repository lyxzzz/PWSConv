import torchvision

root = "/home/liuyuxiang/dataset/imagenet"
# torchvision.datasets.imagenet.parse_train_archive(root)
# torchvision.datasets.imagenet.parse_devkit_archive(root)
torchvision.datasets.imagenet.parse_val_archive(root)