#!/bin/sh
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --export=ALL
#SBATCH --exclusive


module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metric="COTT"
data_path="./data/" 
dataset="Amazon"
n_test_samples=-1
n_val_samples=10000
batch_size=128
arch="distilbert-base-uncased"
model_seed=1
ckpt_epoch=3
corruptions='group-1 group-2'

for corruption in ${corruptions}
    do
    python run_estimation.py --pretrained --corruption ${corruption} --arch ${arch} --metric ${metric} --dataset ${dataset} --subpopulation natural  --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}
    done