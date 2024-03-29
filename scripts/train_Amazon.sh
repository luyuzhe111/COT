#!/bin/bash

model_seed=$1

# distilbert-base-uncased
python train_model.py --dataset Amazon --pretrained --data_path ./data/ --arch distilbert-base-uncased --batch_size 8 --train_epoch 3 --lr 0.00001 --model_seed ${model_seed}