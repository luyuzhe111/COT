import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import os
import numpy as np
from torch_datasets.configs import get_transforms
from torch_datasets.breeds import get_breeds_dataset
from torch_datasets.imagenet import get_imagenet_dataset
from torch_datasets.tiny_imagenet import TinyImageNetCorrupted
from torch_datasets.cifar20 import get_coarse_labels
from torch_datasets.cifar10 import CIFAR10v2
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.wilds_dataset import WILDSSubset



def load_train_dataset(dsname, iid_path, n_val_samples, seed=1, pretrained=True):
    transform = get_transforms(dsname, 'train', pretrained)
    
    if dsname == "CIFAR-10":
        dataset = CIFAR10(iid_path, train=True, transform=transform, download=True)

    elif dsname == "CIFAR-100":
        dataset = CIFAR100(iid_path, train=True, transform=transform, download=True)
    
    elif dsname == 'ImageNet':
        dataset = ImageFolder(f"{iid_path}/imagenetv1/train/", transform=transform)

    elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        dataset = get_breeds_dataset(iid_path, dsname, 'same', split='train', transform=transform)
    
    elif dsname == 'Camelyon17':
        dataset = Camelyon17Dataset(download=True, root_dir=iid_path)
    
    elif dsname == 'FMoW':
        dataset = FMoWDataset(download=True, root_dir=iid_path, use_ood_val=True)
    
    elif dsname == 'RxRx1':
        dataset = RxRx1Dataset(download=True, root_dir=iid_path)
    
    elif dsname == 'Amazon':
        dataset = AmazonDataset(download=True, root_dir=iid_path)
    
    elif dsname == 'CivilComments':
        dataset = CivilCommentsDataset(download=True, root_dir=iid_path)

    else:
        raise ValueError('unknown dataset')
    
    if dsname in ['ImageNet', 'Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        train_set = dataset
    elif dsname == 'Camelyon17':
        train_set = dataset.get_subset('train', transform=transform)
    elif dsname == 'FMoW':
        train_set = dataset.get_subset('train', transform=transform)
    elif dsname == 'RxRx1':
        train_set = dataset.get_subset('train', transform=get_transforms(dsname, 'train', pretrained))
    elif dsname == 'Amazon':
        train_set = dataset.get_subset('train', transform=transform)
    elif dsname == 'CivilComments':
        train_set = dataset.get_subset('train', transform=transform)
    else:
        assert n_val_samples > 0, 'no validation set'
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        n_samples = len(dataset.data)
        index_permute = torch.randperm(n_samples)
        dataset.data = [dataset.data[i] for i in index_permute]
        dataset.targets = [dataset.targets[i] for i in index_permute]

        train_size = n_samples - n_val_samples

        train_inds = index_permute[:train_size]
        train_set = torch.utils.data.Subset(dataset, train_inds)
    
    print("train size:", len(train_set))

    return train_set


def load_val_dataset(dsname, iid_path, n_val_samples, seed=1, pretrained=True):
    transform = get_transforms(dsname, 'train', pretrained)
    
    if dsname == 'ImageNet':
        val_set = ImageFolder(f"{iid_path}/imagenetv1/val/", transform=transform)
    elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        val_set = get_breeds_dataset(iid_path, dsname, 'same', split='val', transform=transform)
    elif dsname == 'Camelyon17':
        dataset = Camelyon17Dataset(download=True, root_dir=iid_path)
        val_set = dataset.get_subset('id_val', transform=transform)
    elif dsname == 'FMoW':
        dataset = FMoWDataset(download=True, root_dir=iid_path, use_ood_val=True)
        val_set = dataset.get_subset('id_val', transform=transform)
    elif dsname == 'RxRx1':
        dataset = RxRx1Dataset(download=True, root_dir=iid_path)
        val_set = dataset.get_subset('id_test', transform=get_transforms(dsname, 'val', pretrained))
    elif dsname == 'Amazon':
        dataset = AmazonDataset(download=True, root_dir=iid_path)
        val_set = dataset.get_subset('id_val', transform=transform)
    elif dsname == 'CivilComments':
        dataset = CivilCommentsDataset(download=True, root_dir=iid_path)
        val_set = dataset.get_subset('val', transform=transform)
    else:
        if dsname == "CIFAR-10":
            dataset = CIFAR10(iid_path, train=True, transform=transform, download=True)

        elif dsname == "CIFAR-100":
            dataset = CIFAR100(iid_path, train=True, transform=transform, download=True)
        
        assert n_val_samples > 0, 'no validation set'
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        n_samples = len(dataset.data)
        index_permute = torch.randperm(n_samples)
        dataset.data = [dataset.data[i] for i in index_permute]
        dataset.targets = [dataset.targets[i] for i in index_permute]

        val_inds = index_permute[n_samples - n_val_samples:]
        val_set = torch.utils.data.Subset(dataset, val_inds)

    print("valid size:", len(val_set))
    return val_set


def load_test_dataset(dsname, iid_path, subpopulation, corr_path, corr, corr_sev, n_test_sample, seed=1, pretrained=True):
    transform = get_transforms(dsname, 'test', pretrained)

    if dsname == "CIFAR-10":
        dataset = CIFAR10(iid_path, train=False, transform=transform, download=True)
    elif dsname == "CIFAR-100":
        dataset = CIFAR100(iid_path, train=False, transform=transform, download=True)
    elif dsname == 'ImageNet':
        dataset = ImageFolder(f"{iid_path}/imagenetv1/test/", transform=transform)
    elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        dataset = get_breeds_dataset(iid_path, dsname, subpopulation, split='test', transform=transform)
    elif dsname == 'Camelyon17':
        dataset = Camelyon17Dataset(download=True, root_dir=iid_path)
    elif dsname == 'FMoW':
        dataset = FMoWDataset(download=True, root_dir=iid_path, use_ood_val=True)
    elif dsname == 'RxRx1':
        dataset = RxRx1Dataset(download=True, root_dir=iid_path)
    elif dsname == 'Amazon':
        dataset = AmazonDataset(download=True, root_dir=iid_path)
    elif dsname == 'CivilComments':
        dataset = CivilCommentsDataset(download=True, root_dir=iid_path)
    else:
        raise ValueError('unknown dataset')
    
    if corr != 'clean':
        if dsname == 'CIFAR-10':
            if corr == 'collection':
                dataset = CIFAR10v2(root="./data/CIFAR-10-V2", train=True, download=True, transform=transform)
            else:
                path_images = os.path.join(corr_path, corr + '.npy')
                path_labels = os.path.join(corr_path, 'labels.npy')
                dataset.data = np.load(path_images)[(corr_sev - 1) * 10000:corr_sev * 10000]
                dataset.targets = list(np.load(path_labels)[(corr_sev - 1) * 10000:corr_sev * 10000])
                dataset.targets = [int(item) for item in dataset.targets]
                
        elif dsname == 'CIFAR-100':
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
        
        elif dsname == 'ImageNet':
            dataset = get_imagenet_dataset(iid_path, subpopulation, transform, corr, corr_sev)
        
        elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
            dataset = get_breeds_dataset(
                iid_path, dsname, subpopulation, split='test', transform=transform, corr=corr, corr_sev=corr_sev 
            )
        
        elif dsname == 'FMoW':
            full_dataset = FMoWDataset(download=True, root_dir=iid_path, use_ood_val=True)
            assert corr in ['13-16', '16-18']
            if corr == '13-16':
                full_testset = full_dataset.get_subset('val', transform=transform)

            elif corr == '16-18':
                full_testset = full_dataset.get_subset('test', transform=transform)
            
            if corr_sev == 0:
                dataset = full_testset
            else:
                groups = full_dataset._eval_groupers['region'].metadata_to_group(full_testset.metadata_array)
                ind = np.where( groups == (corr_sev - 1) )[0]
                dataset = WILDSSubset(full_testset, ind, None)
        
        elif dsname == 'RxRx1':
            full_dataset = RxRx1Dataset(download=True, root_dir=iid_path)
            assert corr in ['batch-1', 'batch-2']
            if corr == 'batch-1':
                dataset = full_dataset.get_subset('val', transform=transform)
            elif corr == 'batch-2':
                dataset = full_dataset.get_subset('test', transform=transform)
        
        elif dsname == 'Amazon':
            full_dataset = AmazonDataset(download=True, root_dir=iid_path)
            assert corr in ['group-1', 'group-2']
            if corr == 'group-1':
                dataset = full_dataset.get_subset('val', transform=transform)
            elif corr == 'group-2':
                dataset = full_dataset.get_subset('test', transform=transform)
        
        elif dsname == 'CivilComments':
            full_dataset = CivilCommentsDataset(download=True, root_dir=iid_path)
            assert corr_sev in [i for i in range(8)]
            
            if corr_sev == 0:
                dataset = full_dataset.get_subset('test', transform=transform)
            else:
                testset = full_dataset.get_subset('test', transform=transform)
                groups = full_dataset._eval_groupers[corr_sev - 1].metadata_to_group(testset.metadata_array)
                
                idx = np.concatenate((np.where(groups==1)[0], np.where(groups==3)[0]))
                dataset = WILDSSubset(testset, idx, transform=None)
        
        elif dsname == 'Camelyon17':
            if corr == 'hospital-1':
                dataset = dataset.get_subset('val', transform=transform)
            elif corr == 'hospital-2':
                dataset = dataset.get_subset('test', transform = transform)
            else:
                raise ValueError('unknown corruption')
            
    # randomly subsample test set to see sample complexity
    # if n_test_sample != -1 and n_test_sample < 10000:
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     indices = torch.randperm(10000)[:n_test_sample]
    #     dataset = torch.utils.data.Subset(dataset, indices)
    #     print('number of test data: ', len(dataset))

    return dataset
        
    
