

module load cuda/11.1.0

source ~/.bashrc
conda activate ood

cd /usr/workspace/lu35/Documents/fot

metric="EMD"
data_path="./data/imagenetv1"
dataset="Living-17"
n_test_samples=-1
n_val_samples=10000
batch_size=128
arch=resnet50
model_seed=1
ckpt_epoch=450

python run_estimation.py --arch ${arch} --metric ${metric} --dataset ${dataset} --batch_size ${batch_size} --n_val_samples ${n_val_samples} --n_test_samples ${n_test_samples} --data_path ${data_path} --ckpt_epoch ${ckpt_epoch}