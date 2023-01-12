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
data_path="./data/CIFAR-100"
data_type="cifar-100"
corruption_path="./data/CIFAR-100-C"
n_class=100
num_ood_samples=10000
n_ref_sample=50000
batch_size=64
arch=resnet18
ref="val"
model_seed="1"

python run_fot.py --data_type ${data_type} --corruption clean --severity 0 --model_seed ${model_seed} --ref ${ref} --num_ref_samples ${n_ref_sample} --num_ood_samples ${num_ood_samples} --batch_size ${batch_size} --num_classes ${n_class} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}

for corruption in brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast
    do
        for level in {1..5}
            do
                echo ${corruption} ${level}
                python run_fot.py --data_type ${data_type} --corruption ${corruption} --severity ${level} --ref ${ref} --model_seed ${model_seed} --num_ref_samples ${n_ref_sample} --num_ood_samples ${num_ood_samples} --batch_size ${batch_size} --num_classes ${n_class} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
            done
    done
