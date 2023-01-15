import torch
import torch.nn as nn
import torchvision.models as models


def ViT_B_16(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(768, num_classes)

    return model

def ResNet18(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(512, num_classes)
    return resnet18


def ResNet50(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet50 = models.resnet50(pretrained=True).cuda()
    resnet50.fc = nn.Linear(2048, num_classes).cuda()
    return resnet50


def ResNet101(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet101 = models.resnet101(pretrained=True).cuda()
    resnet101.fc = nn.Linear(2048, num_classes).cuda()
    return resnet101

def DenseNet121(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    densenet121 = models.densenet121(pretrained=True).cuda()
    densenet121.classifier = nn.Linear(1024, num_classes).cuda()
    return densenet121


def VGG11(num_classes=10, seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    vgg11 = models.vgg11(pretrained=True).cuda()
    vgg11.classifier[-1] = nn.Linear(4096, num_classes).cuda()
    return vgg11

