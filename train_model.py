import argparse
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from load_data import *
from torch_datasets.configs import (
    get_n_classes, get_optimizer, get_lr_scheduler, get_models
)


def main():
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--dataset', default='CIFAR-10', type=str)
    parser.add_argument('--data_path', default='./data/CIFAR-10', type=str)
    parser.add_argument('--n_val_samples', default=10000, type=int)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--train_epoch', default=20, type=int)

    parser.add_argument('--dataset_seed', default=1, type=int)
    parser.add_argument('--model_seed', default=1, type=int)
    parser.add_argument('--alt_model_seed', default=10, type=int)

    args = parser.parse_args()

    dsname = args.dataset
    n_class = get_n_classes(dsname)
    
    save_dir_path = f"./checkpoints/{dsname}/{args.arch}"
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # setup train/val_iid loaders
    trainset, _ = load_train_dataset(dsname=dsname,
                                     iid_path=args.data_path,
                                     n_val_samples=args.n_val_samples,
                                     seed=args.dataset_seed)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init and train base model
    base_model, base_model_alt = get_models(args.arch, n_class, args.model_seed, args.alt_model_seed, args.pretrained)
    base_model, base_model_alt = base_model.to(device), base_model_alt.to(device)
    
    print('begin training...')
    base_model = train(base_model, trainloader, save_dir_path, args, device, alt=False)
    base_model.eval()
    torch.save(base_model, f"{save_dir_path}/base_model_{args.model_seed}.pt")
    print('base model saved to', f"{save_dir_path}/base_model_{args.model_seed}.pt")

    base_model_alt = train(base_model_alt, trainloader, save_dir_path, args, device, alt=True)
    base_model_alt.eval()
    torch.save(base_model_alt, f"{save_dir_path}/base_model_{args.alt_model_seed}.pt")
    print('base model alternate saved to', f"{save_dir_path}/base_model_{args.alt_model_seed}.pt")


def train(net, trainloader, save_dir, args, device, alt=False):
    net.train()
    optimizer = get_optimizer(args.dataset, net)
    scheduler = get_lr_scheduler(args.dataset, optimizer, T_max=args.train_epoch)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.train_epoch):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
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
                          train_loss / (batch_idx + 1), 100. * correct / total, correct, total, current_lr)
                     )
        
        scheduler.step()

        if epoch % 5 == 0:
            if not alt:
                torch.save(net, f"{save_dir}/base_model_{args.model_seed}_{epoch}.pt")
            else:
                torch.save(net, f"{save_dir}/base_model_alt_{args.alt_model_seed}_{epoch}.pt")

    net.eval()

    return net


if __name__ == "__main__":
    main()