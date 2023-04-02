import torchvision.transforms as transforms
import torch.optim as optim
from model import ResNet18, ResNet50, DenseNet121, VGG11, ViT_B_16

def get_train_val_size(dataset):
    config = {
        'CIFAR-10': [50000, 10000],
        'CIFAR-100': [50000, 10000],
        'tiny-imagenet': [90000, 10000]
    }

    return config[dataset]

def get_expected_label_distribution(dataset):
    config = {
        'CIFAR-10': [1 / 10] * 10,
        'CIFAR-100': [1 / 100] * 100,
        'tiny-imagenet': [1 / 200] * 200,
        'Living-17': [1 / 17] * 17,
        'Nonliving-26': [1 / 26] * 26
    }

    return config[dataset]


def get_n_classes(dataset):
    n_class = {
        'CIFAR-10': 10,
        'CIFAR-100': 100,
        'Tiny-ImageNet': 200,
        'Living-17': 17,
        'Nonliving-26': 26,
        'Entity-13': 13,
        'Entity-30': 30,
    }

    return n_class[dataset]


def get_transforms(dataset, split, pretrained):
    if dataset in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        if pretrained:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
    
    elif dataset in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575])
		])
    
    return transform

def get_optimizer(dsname, net, lr, pretrained):
    if dsname in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        if pretrained:
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
        else:
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    elif dsname in ['Living-17', 'Nonliving-26', 'Entity-13', 'Entity-30']:
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


def get_lr_scheduler(dsname, opt, pretrained, T_max=-1):
    if dsname in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        if pretrained:
            return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
        else:
            return optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)
    elif dsname in ['Living-17', 'Nonliving-26']:
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 300], gamma=0.1)
    elif dsname in ['Entity-13', 'Entity-30']:
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=0.1)


def get_models(arch, n_class, model_seed, pretrained):
    if arch == 'resnet18':
        model = ResNet18(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'resnet50':
        model = ResNet50(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'densenet121':
        model = DenseNet121(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'vit_b_16':
        model = ViT_B_16(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    elif arch == 'vgg11':
        model = VGG11(num_classes=n_class, seed=model_seed, pretrained=pretrained)
    else:
        raise ValueError('incorrect model name')

    return model

