#!/bin/sh
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --export=ALL
#SBATCH --exclusive

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

# distilbert-base-uncased
python train_model.py --dataset Amazon --pretrained --data_path ./data/ --arch distilbert-base-uncased --batch_size 8 --train_epoch 3 --lr 0.00001 --model_seed 1