#!/bin/bash

metrics="GDE" # "AC DoC IM GDE ATC-MC ATC-NE COT COTT-MC"
data_path="./data/ImageNet"
dataset="Entity-13"
n_test_samples=-1
n_val_samples=10000
batch_size=128
arch=resnet50
model_seed=$1
ckpt_epoch=300
pretrained="False"
corr_path="./data/ImageNet/imagenet-c"

for metric in ${metrics}
    do 
        python run_estimation.py --arch ${arch} --model_seed ${model_seed} --metric ${metric} --dataset ${dataset} --subpopulation novel --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}
        python run_estimation.py --arch ${arch} --model_seed ${model_seed} --metric ${metric} --dataset ${dataset} --subpopulation same  --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}

        corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"
        for corruption in ${corruptions}
            do
                for level in {1..5}
                    do
                        echo ${corruption} ${level}
                        if [[ ${pretrained} == 'True' ]]
                        then
                            python run_estimation.py --pretrained --model_seed ${model_seed} --dataset ${dataset} --subpopulation same  --corruption ${corruption} --severity ${level} --corruption_path ${corr_path} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path}
                            python run_estimation.py --pretrained --model_seed ${model_seed} --dataset ${dataset} --subpopulation novel --corruption ${corruption} --severity ${level} --corruption_path ${corr_path} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path}
                        else
                            python run_estimation.py --model_seed ${model_seed} --dataset ${dataset} --subpopulation same  --corruption ${corruption} --severity ${level} --corruption_path ${corr_path} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path}
                            python run_estimation.py --model_seed ${model_seed} --dataset ${dataset} --subpopulation novel --corruption ${corruption} --severity ${level} --corruption_path ${corr_path} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path}
                        fi
                    done
            done
    
    done