import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as TF
import torch.optim as optim
from model import (
    ResNet18, ResNet50, DenseNet121, VGG11, ViT_B_16, EfficientNetB4,
    initialize_bert_based_model, initialize_bert_transform
)
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from collections import Counter
import random


def get_expected_label_distribution(dataset):
    if dataset == 'FMoW':
        full_set = FMoWDataset(download=True, root_dir='./data', use_ood_val=True)
        val_set = full_set.get_subset('id_val', transform=None)
        label_counts = Counter(val_set.y_array.tolist())
        total_count = len(val_set.y_array)
        label_dist = [label_counts[i] / total_count for i in range(len(label_counts))]
        return label_dist

    elif dataset == 'RxRx1':
        full_set = RxRx1Dataset(download=True, root_dir='./data')
        val_set = full_set.get_subset('id_test', transform=None)
        label_counts = Counter(val_set.y_array.tolist())
        total_count = len(val_set.y_array)
        label_dist = [label_counts[i] / total_count for i in range(len(label_counts))]
        return label_dist
    
    elif dataset == 'Amazon':
        full_set = AmazonDataset(download=False, root_dir='./data')
        val_set = full_set.get_subset('id_val', transform=None)
        label_counts = Counter(val_set.y_array.tolist())
        total_count = len(val_set.y_array)
        label_dist = [label_counts[i] / total_count for i in range(len(label_counts))]
        return label_dist
    
    elif dataset == 'CivilComments':
        full_set = CivilCommentsDataset(download=True, root_dir=f"./data")
        val_set = full_set.get_subset('val', transform=None)
        label_counts = Counter(val_set.y_array.tolist())
        total_count = len(val_set.y_array)
        label_dist = [label_counts[i] / total_count for i in range(len(label_counts))]
        return label_dist
        
    config = {
        'CIFAR-10': [1 / 10] * 10,
        'CIFAR-100': [1 / 100] * 100,
        'ImageNet': [1 / 1000] * 1000,
        'Living-17': [1 / 17] * 17,
        'Nonliving-26': [1 / 26] * 26,
        'Entity-13': [1 / 13] * 13,
        'Entity-30': [1 / 30] * 30,
    }

    return config[dataset]


def sample_label_dist(ds, n_class, sample_size):
    dist = get_expected_label_distribution(ds)
    labels = sum([[i] * int(dist[i] * sample_size) for i in range(n_class)], [])
    
    remainder = sample_size - len(labels)
    
    r_labels = random.choices(
        list(range(n_class)), 
        weights=get_expected_label_distribution(ds), 
        k=remainder
    )
    
    labels = labels + r_labels
    
    return torch.as_tensor(labels)


def get_n_classes(dataset):
    n_class = {
        'CIFAR-10': 10,
        'CIFAR-100': 100,
        'ImageNet': 1000,
        'Tiny-ImageNet': 200,
        'Living-17': 17,
        'Nonliving-26': 26,
        'Entity-13': 13,
        'Entity-30': 30,
        'FMoW': 62,
        'RxRx1': 1139,
        'Amazon': 5,
        'CivilComments':2
    }

    return n_class[dataset]


def get_transforms(dataset, split, pretrained):
    if dataset in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        if pretrained:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    
    elif dataset == 'ImageNet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
    
    elif dataset in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575])
		])
    
    elif dataset == 'FMoW':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
    
    elif dataset == 'RxRx1':
        def standardize(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=(1, 2))
            std = x.std(dim=(1, 2))
            std[std == 0.] = 1.
            return TF.normalize(x, mean, std)
        
        t_standardize = transforms.Lambda(lambda x: standardize(x))

        angles = [0, 90, 180, 270]
        def random_rotation(x: torch.Tensor) -> torch.Tensor:
            angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
            if angle > 0:
                x = TF.rotate(x, angle)
            return x
        
        t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

        if split == 'train':
            transforms_ls = [
                t_random_rotation,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                t_standardize,
            ]
        else:
            transforms_ls = [
                transforms.ToTensor(),
                t_standardize,
            ]
        transform = transforms.Compose(transforms_ls)
    
    elif dataset == 'Amazon':
        transform = initialize_bert_transform('distilbert-base-uncased', max_token_length=512)
    
    elif dataset == 'CivilComments':
        transform = initialize_bert_transform('distilbert-base-uncased', max_token_length=300)
    
    return transform


def get_optimizer(dsname, net, lr, pretrained):
    if dsname in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        if pretrained:
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        else:
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    elif dsname == 'ImageNet':
        return optim.Adam(net.parameters(), lr=lr)
    
    elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    elif dsname == 'FMoW':
        return optim.Adam(net.parameters(), lr=lr)
    
    elif dsname == 'RxRx1':
        return optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    elif dsname in ['Amazon', 'CivilComments']:
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optim.AdamW(params, lr=lr)


def get_lr_scheduler(dsname, opt, pretrained, T_max=-1):
    if dsname in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        if pretrained:
            return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
        else:
            return optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
    
    elif dsname == 'ImageNet':
       return optim.lr_scheduler.MultiStepLR(opt, milestones=[100], gamma=1)
    
    elif dsname in ['Living-17', 'Nonliving-26']:
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 300], gamma=0.1)
    
    elif dsname in ['Entity-13', 'Entity-30']:
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
    
    elif dsname == 'FMoW':
        return optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.96)
    
    elif dsname == 'RxRx1':
        return optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-4, div_factor=1e12, pct_start=0.11, final_div_factor=1e12,
            cycle_momentum=False, base_momentum=0, max_momentum=0, total_steps=T_max
        )
    
    elif dsname == 'Amazon':
        return optim.lr_scheduler.PolynomialLR(opt, total_iters=3, power=1, verbose=True)

    elif dsname == 'CivilComments':
        return optim.lr_scheduler.PolynomialLR(opt, total_iters=5, power=1, verbose=True)


def get_models(arch, n_class, model_seed, pretrained):
    if arch == 'resnet18':
        model = ResNet18(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'resnet50':
        model = ResNet50(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'efficientnet_b4':
        model = EfficientNetB4(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'densenet121':
        model = DenseNet121(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'vit_b_16':
        model = ViT_B_16(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'vgg11':
        model = VGG11(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'distilbert-base-uncased':
        model = initialize_bert_based_model(n_class)
    else:
        raise ValueError('incorrect model name')

    return model

