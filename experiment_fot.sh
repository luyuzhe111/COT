#!/bin/sh
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --export=ALL
#SBATCH --exclusive


source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metric=wd
cifar_data_path="./data/CIFAR-10"
cifar_corruption_path="./data/CIFAR-10-C"
n_class=10

for arch in resnet18
do
    for layer in logits
    do
        for corruption in elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast # brightness defocus_blur 
        do
            for level in {1..5}
                do
                    echo ${corruption} ${level}
                    python run_fot.py --corruption ${corruption} --severity ${level} --cifar_data_path ${cifar_data_path} --cifar_corruption_path ${cifar_corruption_path} --num_classes ${n_class} --arch ${arch} --metric ${metric}
                done
        done
    done
done