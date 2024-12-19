# Learning to hash
A semester project by group cs-24-dat-5-01 at Aalborg University.
[Repository link](https://github.com/P5-MOAMAS/learning_to_hash)
In the era of big data, efficient similarity search methods are crucial for managing and retrieving information from large-scale, high-dimensional datasets. Hashing-based methods, particularly learning-to-hash approaches, have emerged as powerful tools for encoding data into compact binary representations. These representations allow for fast and memory-efficient similarity comparisons in Hamming space.

This project explores and benchmarks various hashing techniques, including traditional methods Locality-Sensitive Hashing (LSH), Iterative Quantization (ITQ) and Spectral Hashing (SH), as well as advanced deep learning-based approaches such as Deep Supervised Hashing (DSH), HashNet, and Bi-half Net. Our work is grounded in an evaluation framework, leveraging datasets like MNIST, CIFAR-10, and NUS-WIDE to assess accuracy, scalability, and retrieval performance across methods and configurations.

By analyzing the trade-offs between computational cost and retrieval accuracy, this project provides insights into the practical applications of learning-to-hash techniques for image retrieval of large scale image databases. 

The repository contains all the necessary resources, including pre-trained models, extracted features, and setup instructions.

## Setup environment

Python version used [3.12.3 link](https://www.python.org/downloads/release/python-3123/)

### Models and data

#### Models

All pre-trained models can be found as releases in this repository.\
Models should be extracted to "~/learning_to_hash/saved_models".\

#### Features

All pre-extracted features can be found as releases in this repository.\
Features should be extracted to "~/learning_to_hash/features".\

#### Datasets

Mnist-digits and Cifar-10 will automatically get downloaded as needed.\
Nuswide-81-M should be extracted to "~/learning_to_hash/data" and can be downloaded using the following
link: [Google Drive](https://drive.google.com/file/d/0B7IzDz-4yH_HMFdiSE44R1lselE/view?usp=sharing&resourcekey=0-w5zM4GH9liG3rtoZoWzXag),
[original repository](https://github.com/thuml/HashNet/blob/master/pytorch/README.md)

### Installing required packages

All packages used are contained in the pyproject.toml and can be installed using the following methods:

#### Virtual Environment for Linux/WSL

```bash
python -m venv venv
. ./venv/bin/activate
pip install -e .
```

#### Virtual Environment for Windows

```bash
python -m venv venv
./venv/Scripts/activate.bat
pip install -e .
```

#### Global install

```bash
pip install -e .
```

## Training models

### Bihalf

The default config for Bihalf trains on Cifar-10 at 8-bits, this can be altered in "models/deep/unsupervised_image_bit.py".\
Run the following command in the root of the repository to train:

```bash
python3 models/deep/unsupervised_image_bit.py
```

### Deep supervised hashing

The default config for DSH trains on Cifar-10 at 8-bits, this can be altered in "models/DSH/train.py".\
Run the following command in the root of the repository to train:

```bash
python3 models/DSH/train.py
```

### HashNet

The default config for HashNet trains on Cifar-10 at 8-bits, this can be altered in "models/deep/hashnet.py".\
Run the following command in the root of the repository to train:

```bash
python3 models/deep/hashnet.py
```

## Feature extraction

In the case that no GPU is available the flag "-c True" can be used to ensure CPU extraction.\
The following command extracts Mnist, to extract the other datasets replace it with "cifar-10" or "nuswide":

```bash
python3 utility/alexnet_extractor.py -d mnist
```

## Calculating metrics

### Feature extraction

To measure the average extraction time use the following command, to measure the other datasets replace it with "cifar-10" or "nuswide":

```bash
python3 utility/alexnet_extractor.py -d mnist -e True
```

### Metrics for methods

All methods have their own method calculation files, they can be simply just be run to calculate metrics for all datasets on all bits sizes (8, 16, 32, 64).
Example:

```bash
python3 bihalf_metrics_example.py
```

Each files have a config within which can be altered to specify what to calculate metrics for.
