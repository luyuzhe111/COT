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
python train_model.py --dataset CIFAR-10 --data_path ./data/CIFAR-10 --arch resnet18 --batch_size 200 --train_epoch 300 --lr 0.1 --model_seed 1
