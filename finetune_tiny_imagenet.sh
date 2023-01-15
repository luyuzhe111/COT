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
python init_ref_model.py --data_path ./data/Tiny-ImageNet --data_type tiny-imagenet --num_classes 200 --arch resnet50 --train_epoch 20 --pseudo_iters 2000 --lr 0.001 --batch_size 64 --seed ${seed}

# DenseNet101
python init_ref_model.py --data_path ./data/Tiny-ImageNet --data_type tiny-imagenet --num_classes 200 --arch densenet121 --train_epoch 20 --pseudo_iters 2000 --lr 0.001 --batch_size 64 --seed ${seed}

# ViT B 16
python init_ref_model.py --data_path ./data/Tiny-ImageNet --data_type tiny-imagenet --num_classes 200 --arch vit_b_16 --train_epoch 20 --pseudo_iters 2000 --lr 0.001 --batch_size 64 --seed ${seed}
