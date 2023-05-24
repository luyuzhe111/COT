#!/bin/bash

metrics="ATC-NE COT COTT-MC"
data_path="./data/" 
dataset="Amazon"
n_test_samples=-1
n_val_samples=10000
batch_size=128
arch="distilbert-base-uncased"
model_seed=$1
ckpt_epoch=3
corruptions='group-1 group-2'

for metric in ${metrics}
    do
    for corruption in ${corruptions}
        do
        python run_estimation.py --pretrained --model_seed ${model_seed} --corruption ${corruption} --arch ${arch} --metric ${metric} --dataset ${dataset} --subpopulation natural  --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}
        done
    done