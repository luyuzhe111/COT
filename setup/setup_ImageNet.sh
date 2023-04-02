wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

mkdir ../data/ImageNet/imagenetv1/

mv ILSVRC2012_img_train.tar ../data/ImageNet/imagenetv1/
mv ILSVRC2012_img_val.tar ../data/ImageNet/imagenetv1/

cd ../data/ImageNet/imagenetv1/

mkdir ./val/ && tar -xf ILSVRC2012_img_val.tar -C ./val/

mkdir ./train/ && tar -xf ILSVRC2012_img_train.tar -C ./train/
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

cd ../val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

cd ../../
mkdir imagenet-c
wget https://zenodo.org/record/2235448/files/blur.tar
tar -xvf "blur.tar" -C  ./imagenet-c/

wget https://zenodo.org/record/2235448/files/digital.tar
tar -xvf "digital.tar" -C  ./imagenet-c/

wget https://zenodo.org/record/2235448/files/extra.tar
tar -xvf "extra.tar" -C  ./imagenet-c/

wget https://zenodo.org/record/2235448/files/noise.tar
tar -xvf "noise.tar" -C  ./imagenet-c/

wget https://zenodo.org/record/2235448/files/weather.tar
tar -xvf "weather.tar" -C  ./imagenet-c/








