#!/bin/sh
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

seed=1

# ResNet50
python train_model.py --dataset Living-17 --data_path ./data/imagenetv1 --arch resnet50 --batch_size 128 --train_epoch 450 --lr 0.1