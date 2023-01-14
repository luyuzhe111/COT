#!/bin/sh
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

data_path="./data/CIFAR-100"
data_type='cifar-100'
corruption_path="./data/CIFAR-100-C"
n_class=100
seed=1
model_seed="1"

for arch in vgg11

do
    python run_baselines.py --data_type ${data_type} --corruption clean --severity 0 --num_classes ${n_class} --arch ${arch} --model_seed ${model_seed} --seed ${seed} --data_path ${data_path} --corruption_path ${corruption_path}
    
    for corruption in brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast 
        do
            for level in {1..5}
                do
                    echo ${corruption} ${level}
                    python run_baselines.py --data_type ${data_type} --corruption ${corruption} --severity ${level} --num_classes ${n_class} --arch ${arch} --model_seed ${model_seed} --seed ${seed} --data_path ${data_path} --corruption_path ${corruption_path}
                done
        done
done