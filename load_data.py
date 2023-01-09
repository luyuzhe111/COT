import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tiny_imagenet import *


def load_image_dataset(corruption_type,
                       clean_path,
                       corruption_path,
                       corruption_severity=0,
                       datatype='test',
                       num_samples=50000,
                       type="cifar-10",
                       seed=1):
    """
    Returns:
        pytorch dataset object
    """
    assert datatype == 'test' or datatype == 'train'
    training_flag = True if datatype == 'train' else False

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if type == "cifar-10":
        dataset = datasets.CIFAR10(clean_path,
                                   train=training_flag,
                                   transform=transform,
                                   download=True)
    elif type == "cifar-100":
        dataset = datasets.CIFAR100(clean_path,
                                    train=training_flag,
                                    transform=transform,
                                    download=True)
    else:
        if training_flag:
            dataset = TrainTinyImageNetDataset(clean_path, transform=transform)
        else:
            dataset = TestTinyImageNetDataset(clean_path, transform=transform)

    if corruption_severity > 0:
        if type != 'tiny-imagenet':
            assert not training_flag
            path_images = os.path.join(corruption_path, corruption_type + '.npy')
            path_labels = os.path.join(corruption_path, 'labels.npy')
            dataset.data = np.load(path_images)[(corruption_severity - 1) * 10000:corruption_severity * 10000]
            dataset.targets = list(np.load(path_labels)[(corruption_severity - 1) * 10000:corruption_severity * 10000])
            dataset.targets = [int(item) for item in dataset.targets]
        else:
            dataset = TinyImageNetCorruptedDataset(corruption_path, corruption_type, corruption_severity, transform=transform)


    # randomly permute data
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    number_samples = dataset.data.shape[0]
    index_permute = torch.randperm(number_samples)
    dataset.data = dataset.data[index_permute]
    dataset.targets = np.array([int(item) for item in dataset.targets])
    dataset.targets = dataset.targets[index_permute].tolist()

    train_size = max(40000, int(number_samples * 0.8))
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
