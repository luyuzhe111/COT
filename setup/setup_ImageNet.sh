wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

mkdir ../data/imagenetv1/

mv ILSVRC2012_img_train.tar ../data/imagenetv1/
mv ILSVRC2012_img_val.tar ../data/imagenetv1/

cd ../data/imagenetv1/

mkdir ./val/ && tar -xf ILSVRC2012_img_val.tar -C ./val/

mkdir ./train/ && tar -xf ILSVRC2012_img_train.tar -C ./train/
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done






