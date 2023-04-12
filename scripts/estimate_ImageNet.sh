#!/bin/sh
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metric="ATC"
data_path="./data/ImageNet"
dataset="ImageNet"
corruption_path="./data/ImageNet"
n_test_samples=-1
n_val_samples=10000
batch_size=200
arch=resnet50
pretrained="True"
model_seed=1
ckpt_epoch=10

corruptions="brightness defocus_blur elastic_transform fog frost gaussian_blur gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate saturate shot_noise snow spatter speckle_noise zoom_blur contrast"

# echo "pretrained model used"
# python run_estimation.py --pretrained --dataset ${dataset} --corruption clean --severity 0 --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch} --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}

# for corruption in ${corruptions}
#     do
#         for level in {1..5}
#             do
#                 echo ${corruption} ${level}
#                 python run_estimation.py --pretrained --dataset ${dataset} --corruption ${corruption} --severity ${level} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
#             done
#     done

for group in {0..3}
    do
        python run_estimation.py --pretrained --subpopulation natural --dataset ${dataset} --corruption collection --severity ${group} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
    done