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
data_path="./data/CIFAR-10"
dataset="CIFAR-10"
corruption_path="./data/CIFAR-10-C"
n_val_samples=10000
batch_size=128
arch=resnet18
model_seed=1

if [[ ${dataset} == "CIFAR-10" ]] || [[ ${dataset} == "CIFAR-100" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"
elif [[ ${dataset} == "tiny-imagenet" ]]
then
    corruptions="brightness defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur contrast" 
fi

echo ${corruptions}

for n_test_samples in 5000 # 2000 1000 500 200 100
    do
        python run_estimation.py --dataset ${dataset} --corruption clean --severity 0 --model_seed ${model_seed} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}

        for corruption in ${corruptions}
            do
                for level in {1..5}
                    do
                        echo ${corruption} ${level}
                        python run_estimation.py --dataset ${dataset} --corruption ${corruption} --severity ${level} --model_seed ${model_seed} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
                    done
            done
    done