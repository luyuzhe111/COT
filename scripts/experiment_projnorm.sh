#!/bin/sh
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

data_path="./data/Tiny-ImageNet"
corruption_path="./data/Tiny-ImageNet-C"
data_type="tiny-imagenet"
model_seed="1"
seed=1
n_class=200
arch=vgg11

if [[ ${data_type} == "cifar-10" ]] || [[ ${data_type} == "cifar-100" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"
elif [[ ${data_type} == "tiny-imagenet" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur contrast"
fi

echo ${corruptions}

python run_projnorm.py --corruption clean --severity 0 --data_path ${data_path} --data_type ${data_type} --corruption_path ${corruption_path} --num_classes ${n_class} --pseudo_iters 1000 --arch ${arch} --num_ood_samples 10000 --lr 0.001 --batch_size 64 --model_seed ${model_seed} --seed ${seed}

for corruption in ${corruptions}
do
    for level in {1..5}
        do
            echo ${corruption} ${level}
            python run_projnorm.py --corruption ${corruption} --severity ${level} --data_path ${data_path} --data_type ${data_type} --corruption_path ${corruption_path} --num_classes ${n_class} --pseudo_iters 1000 --arch ${arch} --num_ood_samples 10000 --lr 0.001 --batch_size 64 --model_seed ${model_seed} --seed ${seed}
        done
done