#!/bin/sh
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

# cifar-10
seed=1

# ResNet18
python init_ref_model.py --data_path ./data/CIFAR-10 --data_type cifar-10 --num_classes 10 --arch resnet18 --train_epoch 20 --pseudo_iters 1000 --lr 0.001 --batch_size 64 --seed ${seed}

# ResNet50
python init_ref_model.py --data_path ./data/CIFAR-10 --data_type cifar-10 --num_classes 10 --arch resnet50 --train_epoch 20 --pseudo_iters 1000 --lr 0.001 --batch_size 64 --seed ${seed}

# VGG11
python init_ref_model.py --data_path ./data/CIFAR-10 --data_type cifar-10 --num_classes 10 --arch vgg11 --train_epoch 20 --pseudo_iters 1000 --lr 0.001 --batch_size 64 --seed ${seed}