import glob
import os
import random
import shutil
from tqdm import tqdm

train_dir = "../../data/ImageNet/imagenetv1/train"
val_dir="../../data/ImageNet/imagenetv1/val"

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

    files = glob.glob(train_dir + '/**/*.JPEG')
    random.seed(10)
    random.shuffle(files)

    val_files = files[:50000]
    breakpoint()
    for f in tqdm(val_files):
        folder = os.path.dirname(f).split('/')[-1]
        file = os.path.basename(f)
        os.makedirs(f'{val_dir}/{folder}', exist_ok=True)
        shutil.move(f, f'{val_dir}/{folder}/{file}')
else:
    val_files = glob.glob(val_dir + '/**/*.JPEG')
    print(f'{len(val_files)} validation samples.')