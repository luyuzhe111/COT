import torchvision.transforms as transforms
import torch.optim as optim
from model import ResNet18, ResNet50, DenseNet121, VGG11, ViT_B_16

def get_train_val_size(dataset):
    config = {
        'cifar-10': [50000, 10000],
        'cifar-100': [50000, 10000],
        'tiny-imagenet': [90000, 10000]
    }

    return config[dataset]

def get_expected_label_distribution(dataset):
    config = {
        'cifar-10': [1 / 10] * 10,
        'cifar-100': [1 / 100] * 100,
        'tiny-imagenet': [1 / 200] * 200
    }

    return config[dataset]


def get_n_classes(dataset):
    n_class = {
        'CIFAR-10': 10,
        'CIFAR-100': 100,
        'Tiny-ImageNet': 200,
        'Living-17': 17
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
    
    elif dataset == 'Living-17':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575])
		])
    
    return transform

def get_optimizer(dsname, net):
    if dsname in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        return optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)
    elif dsname == 'Living-17':
        return optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


def get_lr_scheduler(dsname, opt, T_max=-1):
    if dsname in ['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet']:
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
    elif dsname == 'Living-17':
        return optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 300, 450], gamma=0.1)


def get_models(arch, n_class, model_seed, alt_model_seed, pretrained):
    if arch == 'resnet18':
        base_model = ResNet18(num_classes=n_class, seed=model_seed, pretrained=pretrained)
        base_model_alt = ResNet18(num_classes=n_class, seed=alt_model_seed, pretrained=pretrained)
    elif arch == 'resnet50':
        base_model = ResNet50(num_classes=n_class, seed=model_seed, pretrained=pretrained)
        base_model_alt = ResNet50(num_classes=n_class, seed=alt_model_seed, pretrained=pretrained)
    elif arch == 'densenet121':
        base_model = DenseNet121(num_classes=n_class, seed=model_seed, pretrained=pretrained)
        base_model_alt = DenseNet121(num_classes=n_class, seed=alt_model_seed, pretrained=pretrained)
    elif arch == 'vit_b_16':
        base_model = ViT_B_16(num_classes=n_class, seed=model_seed, pretrained=pretrained)
        base_model_alt = ViT_B_16(num_classes=n_class, seed=alt_model_seed, pretrained=pretrained)
    elif arch == 'vgg11':
        base_model = VGG11(num_classes=n_class, seed=model_seed, pretrained=pretrained)
        base_model_alt = VGG11(num_classes=n_class, seed=alt_model_seed, pretrained=pretrained)
    else:
        raise ValueError('incorrect model name')

    return base_model, base_model_alt

