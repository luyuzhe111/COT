#!/bin/sh
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metric="EMD"
data_path="./data/Tiny-ImageNet"
data_type="tiny-imagenet"
corruption_path="./data/Tiny-ImageNet-C"
n_class=200
num_ood_samples=10000
batch_size=128
arch=vgg11
ref="val"
model_seed="1"

if [[ ${data_type} == "cifar-10" ]] || [[ ${data_type} == "cifar-100" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"
elif [[ ${data_type} == "tiny-imagenet" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur contrast" 
fi

echo ${corruptions}

python run_fot.py --data_type ${data_type} --corruption clean --severity 0 --model_seed ${model_seed} --ref ${ref} --num_ood_samples ${num_ood_samples} --batch_size ${batch_size} --num_classes ${n_class} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}

for corruption in ${corruptions}
    do
        for level in {1..5}
            do
                echo ${corruption} ${level}
                python run_fot.py --data_type ${data_type} --corruption ${corruption} --severity ${level} --ref ${ref} --model_seed ${model_seed} --num_ood_samples ${num_ood_samples} --batch_size ${batch_size} --num_classes ${n_class} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
            done
    done