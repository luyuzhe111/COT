#!/bin/sh
#SBATCH -N 1
#SBATCH -t 18:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot


seed=0
arch='resnet50' #'efficientnet_b4'

if [ ${arch} == 'resnet50' ]
then
    lr=0.1
else 
    lr=0.01
fi
python train_model.py --dataset CIFAR-10 --data_path ./data/CIFAR-10 --arch ${arch} --batch_size 200 --train_epoch 300 --lr ${lr} --model_seed ${seed}