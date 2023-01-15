import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
from torch_datasets.tiny_imagenet import *
from torch_datasets.configs import get_train_val_size
from torch_datasets.cifar20 import CIFAR20, get_coarse_labels


def load_image_dataset(corruption_type,
                       clean_path,
                       corruption_path,
                       corruption_severity=0,
                       split='test',
                       num_samples=50000,
                       dsname="cifar-10",
                       seed=1):

    assert split in ['train', 'test'], 'unknown split'
    training_flag = True if split == 'train' else False

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dsname == "cifar-10":
        dataset = datasets.CIFAR10(clean_path,
                                   train=training_flag,
                                   transform=transform,
                                   download=True)
    
    elif dsname == "cifar-20":
        dataset = CIFAR20(clean_path,
                          train=training_flag,
                          transform=transform,
                          download=True)

    elif dsname == "cifar-100":
        dataset = datasets.CIFAR100(clean_path,
                                    train=training_flag,
                                    transform=transform,
                                    download=True)
    
    elif dsname == 'tiny-imagenet':
        dataset = TinyImageNet(clean_path, split=split, transform=transform)

    
    if corruption_severity > 0:
        assert not training_flag
        if dsname in ['cifar-10', 'cifar-100']:
            path_images = os.path.join(corruption_path, corruption_type + '.npy')
            path_labels = os.path.join(corruption_path, 'labels.npy')
            dataset.data = np.load(path_images)[(corruption_severity - 1) * 10000:corruption_severity * 10000]
            dataset.targets = list(np.load(path_labels)[(corruption_severity - 1) * 10000:corruption_severity * 10000])
            dataset.targets = [int(item) for item in dataset.targets]
        
        elif dsname == 'cifar-20':
            path_images = os.path.join(corruption_path, corruption_type + '.npy')
            path_labels = os.path.join(corruption_path, 'labels.npy')
            dataset.data = np.load(path_images)[(corruption_severity - 1) * 10000:corruption_severity * 10000]
            dataset.targets = list(get_coarse_labels(np.load(path_labels))[(corruption_severity - 1) * 10000:corruption_severity * 10000])
            dataset.targets = [int(item) for item in dataset.targets]

        elif dsname == 'tiny-imagenet':
            dataset = TinyImageNetCorrupted(corruption_path, corruption_type, corruption_severity, transform=transform)


    # randomly permute data
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    number_samples = len(dataset.data)
    index_permute = torch.randperm(number_samples)
    dataset.data = [dataset.data[i] for i in index_permute]
    # dataset.targets = np.array([int(item) for item in dataset.targets])
    dataset.targets = [dataset.targets[i] for i in index_permute]

    train_size, val_size = get_train_val_size(dsname)
    if training_flag:
        train_inds = index_permute[:train_size]
        val_inds = index_permute[train_size:]

        train_set = torch.utils.data.Subset(dataset, train_inds)
        val_set = torch.utils.data.Subset(dataset, val_inds)

        print("train size:", len(train_set))
        print("valid size:", len(val_set))

        return train_set, val_set

    else:
        # randomly subsample data
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if num_samples < 10000:
            indices = torch.randperm(10000)[:num_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
            print('number of test data: ', len(dataset))

        return dataset
