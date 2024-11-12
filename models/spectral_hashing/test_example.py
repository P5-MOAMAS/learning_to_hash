# Import Spectral Hashing
import spectral_hash as SpectralHash

import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.models import VGG16_Weights

##### TEST BELOW NEED A FROG PICTURE OR JUST USE A CIFAR IMAGE

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalization parameters for ImageNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

cifar10_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

subset_indices = list(range(25000))
cifar10_subset = Subset(cifar10_dataset, subset_indices)

batch_size = 128
train_loader = DataLoader(cifar10_subset, batch_size=batch_size, shuffle=False)

spectral_hash = SpectralHash(
    data_source=train_loader,
    num_bits=16,
    k_neighbors=100,
    n_components=None,
    batch_size=batch_size,
    device=None,
)

cifar10_image, _ = cifar10_subset[0]  # Get the first image as a tensor

query_image_path = "spectral-hashing\\frog2.jpg"
if not os.path.isfile(query_image_path):
    print(f"Query image not found: {query_image_path}")
else:
    spectral_hash.compute_map()
    query_image = Image.open(query_image_path).convert("RGB")

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=query_image,
        num_neighbors=30,
        visualize=True,
    )

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=cifar10_image,
        num_neighbors=30,
        visualize=True,
    )

    cifar10_image, _ = cifar10_subset[1]  # Get the first image as a tensor

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=cifar10_image,
        num_neighbors=30,
        visualize=True,
    )
    query_image = Image.open("spectral-hashing\\truckman.jpg").convert("RGB")

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=query_image,
        num_neighbors=30,
        visualize=True,
    )

    cifar10_image, _ = cifar10_subset[2]  # Get the first image as a tensor

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=cifar10_image,
        num_neighbors=30,
        visualize=True,
    )

    cifar10_image, _ = cifar10_subset[3]  # Get the first image as a tensor

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=cifar10_image,
        num_neighbors=30,
        visualize=True,
    )
