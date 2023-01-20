#!/bin/sh
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

data_path="./data/Tiny-ImageNet"
data_type='tiny-imagenet'
corruption_path="./data/Tiny-ImageNet-C"
n_class=200
seed=1
model_seed="1"
arch="vgg11"

if [[ ${data_type} == "cifar-10" ]] || [[ ${data_type} == "cifar-100" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"
elif [[ ${data_type} == "tiny-imagenet" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur contrast"
fi

python run_baselines.py --data_type ${data_type} --corruption clean --severity 0 --num_classes ${n_class} --arch ${arch} --model_seed ${model_seed} --seed ${seed} --data_path ${data_path} --corruption_path ${corruption_path}
for corruption in ${corruptions}
    do
        for level in {1..5}
            do
                echo ${corruption} ${level}
                python run_baselines.py --data_type ${data_type} --corruption ${corruption} --severity ${level} --num_classes ${n_class} --arch ${arch} --model_seed ${model_seed} --seed ${seed} --data_path ${data_path} --corruption_path ${corruption_path}
            done
    done




    