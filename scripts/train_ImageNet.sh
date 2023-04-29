#!/bin/sh
#SBATCH -N 1
#SBATCH -t 22:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot
arch=resnet50
seed=0
resume_epoch=0
save_interval=5

python train_model.py --pretrained --resume_epoch ${resume_epoch} --dataset ImageNet --data_path ./data/ImageNet --arch ${arch} --batch_size 64 --save_interval ${save_interval} --train_epoch 10 --lr 0.00001 --model_seed ${seed}