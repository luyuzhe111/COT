wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

imgnet_dir=data/ImageNet

mkdir -p ${imgnetdir}/imagenetv1/

mv ILSVRC2012_img_train.tar /data/ImageNet/imagenetv1/
mv ILSVRC2012_img_val.tar /data/ImageNet/imagenetv1/

cd data/ImageNet/imagenetv1/

mkdir ./val/ && tar -xf ILSVRC2012_img_val.tar -C ./val/
mkdir ./train/ && tar -xf ILSVRC2012_img_train.tar -C ./train/

cd train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

cd ../val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..
mv val test

cd ../
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

wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvf imagenet-r.tar -C  ./
rm -rf imagenet-r.tar

gdown http://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
unzip ImageNet-Sketch.zip -d ./
mv ./sketch/ ./imagenet-sketch/
rm -rf ImageNet-Sketch.zip


echo "Downloading Imagenetv2..."
mkdir -p ./imagenetv2

wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz  -C  ./imagenetv2/  
rm -rf  imagenetv2-matched-frequency.tar.gz
python ../../setup/Imagenet/ImageNet_v2_reorg.py --dir ./imagenetv2/imagenetv2-matched-frequency-format-val --info ./imagenet_class_hierarchy/dataset_class_info.json

wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz
tar -xvf imagenetv2-threshold0.7.tar.gz -C  ./imagenetv2/
rm -rf imagenetv2-threshold0.7.tar.gz
python ../../setup/Imagenet/ImageNet_v2_reorg.py --dir ./imagenetv2/imagenetv2-threshold0.7-format-val --info ./imagenet_class_hierarchy/dataset_class_info.json

wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz
tar -xvf imagenetv2-top-images.tar.gz -C  ./imagenetv2/
rm -rf imagenetv2-top-images.tar.gz
python ../../setup/Imagenet/ImageNet_v2_reorg.py --dir ./imagenetv2/imagenetv2-top-images-format-val --info ./imagenet_class_hierarchy/dataset_class_info.json