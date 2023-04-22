#!/bin/sh
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metrics="AC DoC IM ATC"
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

for metric in ${metrics}
    do
        for group in {0..3}
            do
                python run_estimation.py --pretrained --subpopulation natural --dataset ${dataset} --corruption collection --severity ${group} --model_seed ${model_seed} --ckpt_epoch ${ckpt_epoch}  --n_test_samples ${n_test_samples} --batch_size ${batch_size} --arch ${arch} --metric ${metric} --data_path ${data_path} --corruption_path ${corruption_path}
            done
    done