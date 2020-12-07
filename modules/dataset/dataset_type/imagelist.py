import os
from PIL import Image

class ImageList(object):
    def __init__(self, root, list_file, num_class, preload=False):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.preload = preload
        self.has_labels = len(lines[0].split()) == 2
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
            for l in self.labels:
                assert l < num_class
        else:
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        if self.preload:
            self.data_infos = []
            for idx in range(len(self.fns)):
                img = Image.open(self.fns[idx])
                img = img.convert('RGB')
                target = self.labels[idx]
                self.data_infos.append([img, target])

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.preload:
            return self.data_infos[idx]
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img
    
    def has_labels(self):
        return self.has_labels