#!/bin/bash

arch=resnet50
seed=$1
resume_epoch=0
save_interval=5

python train_model.py --pretrained --resume_epoch ${resume_epoch} --dataset ImageNet --data_path ./data/ImageNet --arch ${arch} --batch_size 64 --save_interval ${save_interval} --train_epoch 10 --lr 0.0001 --model_seed ${seed}