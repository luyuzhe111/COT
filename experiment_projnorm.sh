#!/bin/sh
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

data_path="./data/CIFAR-100"
corruption_path="./data/CIFAR-100-C"
data_type="cifar-100"
model_seed="1"
seed=1
n_class=100
arch=vgg11

python run_projnorm.py --corruption clean --severity 0 --data_path ${data_path} --data_type ${data_type} --corruption_path ${corruption_path} --num_classes ${n_class} --arch ${arch} --num_ood_samples 10000 --lr 0.001 --batch_size 64 --model_seed ${model_seed} --seed ${seed}

for corruption in brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast
do
    for level in {1..5}
        do
            echo ${corruption} ${level}
            python run_projnorm.py --corruption ${corruption} --severity ${level} --data_path ${data_path} --data_type ${data_type} --corruption_path ${corruption_path} --num_classes ${n_class} --arch ${arch} --num_ood_samples 10000 --lr 0.001 --batch_size 64 --model_seed ${model_seed} --seed ${seed}
        done
done


