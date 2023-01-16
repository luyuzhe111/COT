import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os, glob


class TinyImageNet(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.id_dict = {}
        for i, line in enumerate(open(os.path.join(data_path, 'wnids.txt'), 'r')):
            self.id_dict[line.replace('\n', '')] = i
        
        self.split = split
        if split == 'train':
            self.filenames = sorted(glob.glob(os.path.join(data_path, "train/*/*/*.JPEG")))
            self.targets = [self.id_dict[img_path.replace("\\", "/").split("/")[-3]] for img_path in self.filenames]
        elif split == 'test':
            self.filenames = sorted(glob.glob(os.path.join(data_path, "val/images/*.JPEG")))
            self.cls_dic = {}
            for i, line in enumerate(open(os.path.join(data_path, 'val/val_annotations.txt'), 'r')):
                a = line.split('\t')
                img, cls_id = a[0], a[1]
                self.cls_dic[img] = self.id_dict[cls_id]
            
            self.targets = [self.cls_dic[img_path.replace("\\", "/").split('/')[-1]] for img_path in self.filenames]

        self.transform = transform
        self.data = [{'img_dir': filename, 'target': target} for (filename, target) in zip(self.filenames, self.targets)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img, target = self.data[idx]['img_dir'], self.data[idx]['target']
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TinyImageNetCorrupted(Dataset):
    def __init__(self, corrupted_path, corruption_type, corruption_severity, transform=None):
        self.filenames = sorted(glob.glob(os.path.join(corrupted_path, corruption_type, str(corruption_severity), "*/*.JPEG")))
        self.transform = transform
        self.id_dict = {}
        for i, line in enumerate(open(os.path.join(corrupted_path, 'wnids.txt'), 'r')):
            self.id_dict[line.replace('\n', '')] = i

        self.targets = [self.id_dict[img_path.replace("\\", "/").split('/')[-2]] for img_path in self.filenames if img_path.replace("\\", "/").split('/')[-2] in self.id_dict]
        self.data = [{'img_dir': filename, 'target': target} for (filename, target) in zip(self.filenames, self.targets)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img, target = self.data[idx]['img_dir'], self.data[idx]['target']
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target