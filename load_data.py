import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
import os
import numpy as np
from torch_datasets.configs import get_transforms
from torch_datasets.breeds import get_breeds_dataset
from torch_datasets.tiny_imagenet import TinyImageNet, TinyImageNetCorrupted
from torch_datasets.cifar20 import CIFAR20, get_coarse_labels


def load_train_dataset(dsname, iid_path, n_val_samples, seed=1, pretrained=True):
    transform = get_transforms(dsname, 'train', pretrained)
    
    if dsname == "CIFAR-10":
        dataset = CIFAR10(iid_path, train=True, transform=transform, download=True)

    elif dsname == "CIFAR-20":
        dataset = CIFAR20(iid_path, train=True, transform=transform, download=True)

    elif dsname == "CIFAR-100":
        dataset = CIFAR100(iid_path, train=True, transform=transform, download=True)
    
    elif dsname == 'Tiny-ImageNet':
        dataset = TinyImageNet(iid_path, split='train', transform=transform)
    
    elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        dataset = get_breeds_dataset(iid_path, dsname, 'same', split='train', transform=transform)

    else:
        raise ValueError('unknown dataset')

    assert n_val_samples > 0, 'no validation set'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    n_samples = len(dataset.data)
    index_permute = torch.randperm(n_samples)
    dataset.data = [dataset.data[i] for i in index_permute]
    dataset.targets = [dataset.targets[i] for i in index_permute]

    train_size = n_samples - n_val_samples

    train_inds = index_permute[:train_size]
    val_inds = index_permute[train_size:]

    train_set = torch.utils.data.Subset(dataset, train_inds)
    val_set = torch.utils.data.Subset(dataset, val_inds)

    print("train size:", len(train_set))
    print("valid size:", len(val_set))

    return train_set, val_set


def load_test_dataset(dsname, iid_path, subpopulation, corr_path, corr, corr_sev, n_test_sample, seed=1, pretrained=True):
    transform = get_transforms(dsname, 'test', pretrained)

    # test on non-corrupted data
    if dsname == "CIFAR-10":
        dataset = CIFAR10(iid_path, train=False, transform=transform, download=True)
    elif dsname == "CIFAR-20":
        dataset = CIFAR20(iid_path, train=False, transform=transform, download=True)
    elif dsname == "CIFAR-100":
        dataset = CIFAR100(iid_path, train=False, transform=transform, download=True)
    elif dsname == 'Tiny-ImageNet':
        dataset = TinyImageNet(iid_path, split='test', transform=transform)
    elif dsname in ['Living-17', 'Nonliving-26']:
        dataset = get_breeds_dataset(iid_path, dsname, subpopulation, split='test', transform=transform)
    else:
        raise ValueError('unknown dataset')
    
    # test on corrupted data
    if corr != 'clean':
        if dsname in ['CIFAR-10', 'CIFAR-100']:
            path_images = os.path.join(corr_path, corr + '.npy')
            path_labels = os.path.join(corr_path, 'labels.npy')
            dataset.data = np.load(path_images)[(corr_sev - 1) * 10000:corr_sev * 10000]
            dataset.targets = list(np.load(path_labels)[(corr_sev - 1) * 10000:corr_sev * 10000])
            dataset.targets = [int(item) for item in dataset.targets]
        
        elif dsname == 'CIFAR-20':
            path_images = os.path.join(corr_path, corr + '.npy')
            path_labels = os.path.join(corr_path, 'labels.npy')
            dataset.data = np.load(path_images)[(corr_sev - 1) * 10000:corr_sev * 10000]
            dataset.targets = list(get_coarse_labels(np.load(path_labels))[(corr_sev - 1) * 10000:corr_sev * 10000])
            dataset.targets = [int(item) for item in dataset.targets]

        elif dsname == 'Tiny-ImageNet':
            dataset = TinyImageNetCorrupted(corr_path, corr, corr_sev, transform=transform)
        
        elif dsname in ['Living-17', 'Nonliving-26']:
            dataset = get_breeds_dataset(
                iid_path, dsname, subpopulation, split='test', transform=transform, corr=corr, corr_sev=corr_sev 
            )

    # randomly subsample test set to see sample complexity
    if n_test_sample != -1 and n_test_sample < 10000:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        indices = torch.randperm(10000)[:n_test_sample]
        dataset = torch.utils.data.Subset(dataset, indices)
        print('number of test data: ', len(dataset))

    return dataset
        
    
