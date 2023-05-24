#!/bin/bash

model_seed=$1

# ResNet50
python train_model.py --dataset Camelyon17 --data_path ./data/ --arch resnet50 --batch_size 32 --train_epoch 5 --lr 0.001 --model_seed ${model_seed}