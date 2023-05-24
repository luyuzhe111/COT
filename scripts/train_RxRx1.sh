#!/bin/bash

model_seed=$1

# ResNet50
python train_model.py --dataset RxRx1 --pretrained --data_path ./data/ --arch resnet50 --batch_size 75 --train_epoch 90 --lr 0.0001 --model_seed ${model_seed}