#!/bin/sh

metrics="GDE ATC-NE COTT-val-MC"
data_path="./data/" 
dataset="Camelyon17"
n_test_samples=-1
n_val_samples=10000
batch_size=128
arch="resnet50"
model_seed=$1
ckpt_epoch=5
corruptions='hospital-1 hospital-2'

for metric in ${metrics}
    do
    for corruption in ${corruptions}
        do
        python run_estimation.py --model_seed ${model_seed} --corruption ${corruption} --arch ${arch} --metric ${metric} --dataset ${dataset} --subpopulation natural  --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}
        done
    done