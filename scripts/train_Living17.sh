#!/bin/sh
#SBATCH -N 1
#SBATCH -t 21:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot
seed=$1
arch=resnet50
python train_model.py --dataset Living-17 --data_path ./data/ImageNet --arch ${arch} --batch_size 128 --train_epoch 450 --lr 0.1 --model_seed ${seed}