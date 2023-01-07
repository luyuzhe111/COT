import argparse
import torchvision.models as models
import torch.nn as nn

from projnorm import *
from load_data import *
from model import ResNet18, ResNet50, VGG11

"""# Configuration"""
parser = argparse.ArgumentParser(description='ProjNorm.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--cifar_data_path', default='./data/CIFAR-10/', type=str)
parser.add_argument('--cifar_corruption_path', default='./data/CIFAR-10-C/', type=str)
parser.add_argument('--pseudo_iters', default=50, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--train_epoch', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
args = vars(parser.parse_args())


def train(net, trainloader):
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['train_epoch'] * len(trainloader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args['train_epoch']):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 20 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('Epoch: ', epoch, '(', batch_idx, '/', len(trainloader), ')',
                      'Loss: %.3f | Acc: %.3f%% (%d/%d)| Lr: %.5f' % (
                          train_loss / (batch_idx + 1), 100. * correct / total, correct, total, current_lr))
            scheduler.step()
        
        if epoch % 5 == 0 and epoch > 0:
            torch.save(net, f"{save_dir_path}/base_model_{args['seed']}_{epoch}.pt")
    
    net.eval()

    return net


if __name__ == "__main__":
    # save path
    data_type = "cifar-100" if args["num_classes"] == 100 else "cifar-10"

    save_dir_path = f"./checkpoints/{data_type}/{args['arch']}"
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # setup train / val_iid loaders
    trainset, valset = load_cifar_image(corruption_type='clean',
                                clean_cifar_path=args['cifar_data_path'],
                                corruption_cifar_path=args['cifar_corruption_path'],
                                corruption_severity=0,
                                type=data_type,
                                seed=args['seed'],
                                datatype='train')
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args['batch_size'], shuffle=True)

    # init and train base model
    if args['arch'] == 'resnet18':
        base_model = ResNet18(num_classes=args['num_classes'], seed=args['seed']).cuda()
    elif args['arch'] == 'resnet50':
        base_model = ResNet50(num_classes=args['num_classes'], seed=args['seed']).cuda()
    elif args['arch'] == 'vgg11':
        base_model = VGG11(num_classes=args['num_classes'], seed=args['seed']).cuda()
    else:
        raise ValueError('incorrect model name')

    print('begin training...')
    base_model = train(base_model, trainloader)
    base_model.eval()
    torch.save(base_model, f"{save_dir_path}/base_model_{args['seed']}_{args['train_epoch']}.pt")
    print('base model saved to', f"{save_dir_path}/base_model_{args['seed']}.pt")

    # init ProjNorm
    PN = ProjNorm(base_model=base_model)

    # train iid reference model
    if args['arch'] == 'resnet18':
        ref_model = ResNet18(num_classes=args['num_classes'], seed=args['seed']).cuda()
    elif args['arch'] == 'resnet50':
        ref_model = ResNet50(num_classes=args['num_classes'], seed=args['seed']).cuda()
    elif args['arch'] == 'vgg11':
        ref_model = VGG11(num_classes=args['num_classes'], seed=args['seed']).cuda()
    else:
        raise ValueError('incorrect model name')

    PN.update_ref_model(trainloader,
                        ref_model,
                        lr=args['lr'],
                        pseudo_iters=args['pseudo_iters'])
    
    torch.save(PN.reference_model.eval(), f"{save_dir_path}/ref_model_{args['seed']}.pt")
    print('reference model saved to', f"{save_dir_path}/ref_model_{args['seed']}_{args['pseudo_iters']}.pt")

    print('========finished========')
