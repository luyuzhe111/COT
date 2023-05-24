mkdir -p ./data/CIFAR-10/
cd ./data/CIFAR-10/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz

cd ..
mkdir ./CIFAR-10-C/
cd ./CIFAR-10-C/
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar

tar -xvf CIFAR-10-C.tar

cd ..
mkdir ./CIFAR-10-V2

cd ./CIFAR-10-V2
wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_train.npz

wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_test.npz