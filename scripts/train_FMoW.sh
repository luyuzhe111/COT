#!/bin/sh
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

# ResNet50
python train_model.py --dataset FMoW --pretrained --data_path ./data/ --arch resnet50 --batch_size 64 --train_epoch 50 --lr 0.0001 --model_seed 1