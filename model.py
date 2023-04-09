import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertForSequenceClassification, DistilBertModel
from transformers import BertTokenizerFast, DistilBertTokenizerFast

import os
os.environ['CURL_CA_BUNDLE'] = ''


def ViT_B_16(num_classes=10, seed=123, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = models.vit_b_16(pretrained=pretrained)
    model.heads.head = nn.Linear(768, num_classes)

    return model

def ResNet18(num_classes=10, seed=123, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet18 = models.resnet18(pretrained=pretrained)
    resnet18.fc = nn.Linear(512, num_classes)
    return resnet18


def ResNet50(num_classes=10, seed=123, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet50 = models.resnet50(pretrained=pretrained).cuda()
    resnet50.fc = nn.Linear(2048, num_classes).cuda()
    return resnet50


def ResNet101(num_classes=10, seed=123, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet101 = models.resnet101(pretrained=pretrained).cuda()
    resnet101.fc = nn.Linear(2048, num_classes).cuda()
    return resnet101

def DenseNet121(num_classes=10, seed=123, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    densenet121 = models.densenet121(pretrained=pretrained).cuda()
    densenet121.classifier = nn.Linear(1024, num_classes).cuda()
    return densenet121


def VGG11(num_classes=10, seed=123, pretrained=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    vgg11 = models.vgg11(pretrained=pretrained).cuda()
    vgg11.classifier[-1] = nn.Linear(4096, num_classes).cuda()
    return vgg11


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs
    

def initialize_bert_based_model(num_classes):
    model = DistilBertClassifier.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes
    )
    return model


def initialize_bert_transform(net, max_token_length=512):
    # assert 'bert' in config.model
    # assert config.max_token_length is not None

    tokenizer = getBertTokenizer(net)
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        if net == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif net == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform


def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer



