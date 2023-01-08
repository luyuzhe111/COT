import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, clean_path, transform=None):
        self.filenames = glob.glob(os.path.join(clean_path, "train/*/*/*.JPEG"))
        self.transform = transform
        self.id_dict = {}
        for i, line in enumerate(open(os.path.join(clean_path, 'wnids.txt'), 'r')):
            self.id_dict[line.replace('\n', '')] = i
        self.data = np.stack([read_image(img_path) if read_image(img_path).shape[0] != 1
                    else read_image(img_path, ImageReadMode.RGB) for img_path in self.filenames])
        self.targets = [self.id_dict[img_path.replace("\\", "/").split("/")[-3]] for img_path in self.filenames]
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TestTinyImageNetDataset(Dataset):
    def __init__(self, clean_path, transform=None):
        self.filenames = glob.glob(os.path.join(clean_path, "val/images/*.JPEG"))
        self.transform = transform
        self.id_dict = {}
        for i, line in enumerate(open(os.path.join(clean_path, 'wnids.txt'), 'r')):
            self.id_dict[line.replace('\n', '')] = i

        self.cls_dic = {}
        for i, line in enumerate(open(os.path.join(clean_path, 'val/val_annotations.txt'), 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

        self.data = np.stack([read_image(img_path) if read_image(img_path).shape[0] != 1
                    else read_image(img_path, ImageReadMode.RGB) for img_path in self.filenames])
        self.targets = [self.cls_dic[img_path.replace("\\", "/").split('/')[-1]] for img_path in self.filenames]
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TinyImageNetCorruptedDataset(Dataset):
    def __init__(self, corrupted_path, corruption_type, corruption_severity, transform=None):
        self.filenames = glob.glob(os.path.join(corrupted_path, corruption_type, str(corruption_severity), "*/*.JPEG"))
        self.transform = transform
        self.id_dict = {}
        for i, line in enumerate(open(os.path.join(corrupted_path, 'wnids.txt'), 'r')):
            self.id_dict[line.replace('\n', '')] = i
        self.data = np.stack([read_image(img_path) for img_path in self.filenames])
        self.targets = [self.id_dict[img_path.replace("\\", "/").split('/')[-2]] for img_path in self.filenames]
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target