#!/bin/sh
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot
resume_epoch=0

# ResNet50
python train_model.py --pretrained --resume_epoch ${resume_epoch} --dataset ImageNet --data_path ./data/ImageNet --arch resnet50 --batch_size 64 --train_epoch 10 --lr 0.00001 --model_seed 1