#!/bin/bash

seed=$1
arch='resnet18'
lr=0.1
python train_model.py --dataset CIFAR-10 --data_path ./data/CIFAR-10 --arch ${arch} --batch_size 200 --train_epoch 300 --lr ${lr} --model_seed ${seed}