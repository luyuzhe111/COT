#!/bin/sh
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot
model_seed=$1

# ResNet50
python train_model.py --dataset Camelyon17 --pretrained --data_path ./data/ --arch resnet50 --batch_size 32 --train_epoch 5 --lr 0.001 --model_seed ${model_seed}