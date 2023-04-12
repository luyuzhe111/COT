from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np


class CIFAR10v2(CIFAR10):
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform

        if train: 
            data = np.load(root + "/" + 'cifar102_train.npz', allow_pickle=True)
        else: 
            data = np.load(root + "/" + 'cifar102_test.npz', allow_pickle=True)
            
        self.data = data["images"]
        self.targets = data["labels"]

    def __len__(self): 
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
