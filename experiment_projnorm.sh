#!/bin/sh
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cifar_data_path="./data/CIFAR-100"
cifar_corruption_path="./data/CIFAR-100-C"
model_seed="1_15"
seed=1
n_class=100
arch=resnet50

python run_projnorm.py --corruption clean --severity 0 --cifar_data_path ${cifar_data_path} --cifar_corruption_path ${cifar_corruption_path} --num_classes ${n_class} --arch ${arch} --num_ood_samples 10000 --lr 0.001 --batch_size 64 --model_seed ${model_seed} --seed ${seed}

for corruption in brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast
do
    for level in {1..5}
        do
            echo ${corruption} ${level}
            python run_projnorm.py --corruption ${corruption} --severity ${level} --cifar_data_path ${cifar_data_path} --cifar_corruption_path ${cifar_corruption_path} --num_classes ${n_class} --arch ${arch} --num_ood_samples 10000 --lr 0.001 --batch_size 64 --model_seed ${model_seed} --seed ${seed}
        done
done


