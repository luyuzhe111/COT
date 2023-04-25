import argparse
import torchvision.models as models
import torch.nn as nn
from load_data import *
from torch_datasets.configs import (
    get_n_classes, get_optimizer, get_lr_scheduler, get_models
)
import time
import torch.backends.cudnn as cudnn


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
    parser.add_argument('--save_interval', default=50, type=int)
    parser.add_argument('--resume_epoch', default=0, type=int)

    parser.add_argument('--dataset_seed', default=1, type=int)
    parser.add_argument('--model_seed', default=1, type=int)

    args = parser.parse_args()

    print(vars(args))

    dsname = args.dataset
    n_class = get_n_classes(dsname)
    
    if args.pretrained:
        save_dir_path = f"./checkpoints/{dsname}/{args.arch}/pretrained"
    else:
        save_dir_path = f"./checkpoints/{dsname}/{args.arch}/scratch"
    
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # setup train/val_iid loaders
    trainset = load_train_dataset(dsname=dsname,
                                  iid_path=args.data_path,
                                  n_val_samples=args.n_val_samples,
                                  pretrained=args.pretrained,
                                  seed=args.dataset_seed)

    valset = load_val_dataset(dsname=dsname,
                              iid_path=args.data_path,
                              n_val_samples=args.n_val_samples,
                              pretrained=args.pretrained,
                              seed=args.dataset_seed)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init and train base modeltrain_model.py
    model = get_models(args.arch, n_class, args.model_seed, args.pretrained).to(device)

    n_device = torch.cuda.device_count()
    print('available devices:', n_device)
    model = torch.nn.DataParallel( model, device_ids=range(n_device) )
    cudnn.benchmark = False

    optimizer = get_optimizer(args.dataset, model, args.lr, args.pretrained)
    scheduler = get_lr_scheduler(args.dataset, optimizer, args.pretrained, T_max=args.train_epoch * len(trainloader))

    resume_epoch = args.resume_epoch
    if resume_epoch > 0:
        ckpt_dir = f"{save_dir_path}/base_model_{args.model_seed}-{resume_epoch}.pt"
        ckpt = torch.load(ckpt_dir, map_location=device)
        model = ckpt['model']
        optimizer = get_optimizer(args.dataset, model, args.lr, args.pretrained)
        scheduler = get_lr_scheduler(args.dataset, optimizer, args.pretrained, T_max=args.train_epoch)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['optimizer_state_dict'])
    
    print('begin training...')
    train(model, optimizer, scheduler, trainloader, valloader, save_dir_path, args, device)


def train(net, optimizer, scheduler, trainloader, valloader, save_dir, args, device):
    net.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(1, args.train_epoch + 1):
        train_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for batch_idx, items in enumerate(trainloader):
            inputs, targets = items[0], items[1]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

            if batch_idx % 100 == 0:   
                print(f"time used: {time.time() - start}s")
            
            if args.dataset == 'RxRx1':
                scheduler.step()
        
        if args.dataset != 'RxRx1':
            scheduler.step()

        end = time.time()
        print(f"time used: {end - start}s")

        if epoch % args.save_interval == 0:
            torch.save({
                'model': net, 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            },
            f"{save_dir}/base_model_{args.model_seed}-{epoch + args.resume_epoch}.pt")

        if epoch % 10 == 0:
            net.eval()
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for items in valloader:
                    inputs, targets = items[0], items[1]
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            net.train()
            
            print(f'Epoch {epoch} Validation Acc: {val_correct / val_total}')
        
        if args.resume_epoch + epoch >= args.train_epoch:
            break

    net.eval()
    torch.save({
        'model': net, 
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    },
    f"{save_dir}/base_model_{args.model_seed}-{args.train_epoch}.pt")
    print('base model saved to', f"{save_dir}/base_model_{args.model_seed}-{args.train_epoch}.pt")

    return net


if __name__ == "__main__":
    main()