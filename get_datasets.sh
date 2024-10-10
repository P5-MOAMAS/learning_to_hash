#!/usr/bin/env bash

echo "Retrieving cifar10 dataset for python"
curl https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz -o cifar10.tar.gz

echo "Extracting cifar10 dataset"
tar xf cifar10.tar.gz

echo "Cleaning up..."
rm cifar10.tar.gz

echo "Success!"
