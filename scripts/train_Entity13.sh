#!/bin/sh
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

# ResNet50
python train_model.py --dataset Entity-13 --data_path ./data/ImageNet --arch resnet50 --batch_size 128 --train_epoch 300 --lr 0.1 --model_seed 1