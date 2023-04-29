#!/bin/sh
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot
resume_epoch=200
arch=resnet50
seed=0

# ResNet50
python train_model.py --resume_epoch ${resume_epoch} --dataset Entity-13 --data_path ./data/ImageNet --arch ${arch} --batch_size 128 --train_epoch 300 --lr 0.1 --model_seed ${seed}