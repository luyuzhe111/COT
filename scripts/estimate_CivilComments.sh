#!/bin/sh
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --export=ALL
#SBATCH --exclusive


module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metrics="AC DoC IM COT COTT"
data_path="./data/" 
dataset="CivilComments"
n_test_samples=-1
n_val_samples=10000
batch_size=128
arch="distilbert-base-uncased"
model_seed=1
ckpt_epoch=5
corruption='identity'

for metric in ${metrics}
    do 
        for level in {0..7}
            do
            python run_estimation.py --pretrained --corruption ${corruption} --severity ${level} --arch ${arch} --metric ${metric} --dataset ${dataset} --subpopulation natural  --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}
            done
    done