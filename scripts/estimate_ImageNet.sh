#!/bin/bash

metrics="GDE" # "AC DoC IM GDE ATC-MC ATC-NE COT COTT-MC"
data_path="./data/ImageNet"
dataset="ImageNet"
corruption_path="./data/ImageNet"
n_test_samples=-1
n_val_samples=10000
batch_size=200
arch=resnet50
pretrained="True"
model_seed=$1
ckpt_epoch=10

for metric in ${metrics}
    do
        for group in {0..3}
            do
                python run_estimation.py --pretrained --subpopulation natural --dataset ${dataset} --corruption collection --severity ${group} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
            done
    done


corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"

echo "pretrained model used"

for metric in ${metrics}
    do
        python run_estimation.py --pretrained --dataset ${dataset} --corruption clean --severity 0 --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
        for corruption in ${corruptions}
            do
                for level in {1..5}
                    do
                        echo ${corruption} ${level}
                        python run_estimation.py --pretrained --dataset ${dataset} --corruption ${corruption} --severity ${level} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
                    done
            done
    done