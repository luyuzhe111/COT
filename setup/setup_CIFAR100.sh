#!/bin/bash

## Download CIFAR10-C
echo "Downloading CIFAR-100 C..."

wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf "CIFAR-100-C.tar" -C  ./data/
rm -rf "CIFAR-100-C.tar"

echo "CIFAR100-C downloaded"