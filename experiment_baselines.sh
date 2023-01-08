#!/bin/sh
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cifar_data_path="./data/CIFAR-10"
cifar_corruption_path="./data/CIFAR-10-C"
n_class=10
seed=1
model_seed="1_15"

for arch in resnet18 resnet50 vgg11

do
    python run_baselines.py --corruption clean --severity 0 --num_classes ${n_class} --arch ${arch} --model_seed ${model_seed} --seed ${seed} --cifar_data_path ${cifar_data_path} --cifar_corruption_path ${cifar_corruption_path}
    
    for corruption in brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast 
        do
            for level in {1..5}
                do
                    echo ${corruption} ${level}
                    python run_baselines.py --corruption ${corruption} --severity ${level} --num_classes ${n_class} --arch ${arch} --model_seed ${model_seed} --seed ${seed} --cifar_data_path ${cifar_data_path} --cifar_corruption_path ${cifar_corruption_path}
                done
        done
done