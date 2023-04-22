#!/bin/sh
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metrics="AC DoC IM ATC COT COTT"
data_path="./data/CIFAR-10"
dataset="CIFAR-10"
corruption_path="./data/CIFAR-10-C"
n_test_samples=-1
n_val_samples=10000
batch_size=200
arch=resnet18
pretrained="False"
model_seed=1
ckpt_epoch=300

corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"

echo ${corruptions}

for metric in ${metrics}
    do
    if [[ ${pretrained} == 'True' ]]
    then
        echo "pretrained model used"
        python run_estimation.py --pretrained --dataset ${dataset} --corruption clean --severity 0 --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
        python run_estimation.py --pretrained --dataset ${dataset} --corruption collection --subpopulation natural --severity 0 --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
    else
        echo "scratch model used"
        python run_estimation.py --dataset ${dataset} --corruption clean --severity 0 --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
        python run_estimation.py --dataset ${dataset} --corruption collection --subpopulation natural --severity 0 --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
    fi

    for corruption in ${corruptions}
        do
            for level in {1..5}
                do
                    echo ${corruption} ${level}
                    if [[ ${pretrained} == 'True' ]]
                    then
                        python run_estimation.py --pretrained --dataset ${dataset} --corruption ${corruption} --severity ${level} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
                    else
                        python run_estimation.py --dataset ${dataset} --corruption ${corruption} --severity ${level} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
                    fi
                done
        done
    done