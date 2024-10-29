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

# Define transform
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalization parameters for ImageNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Download cifar10 dataset, can be changed to use any other dataset
cifar10_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Create subsets
subset_indices = list(range(10000))
cifar10_subset = Subset(cifar10_dataset, subset_indices)

#Define batch size and dataloader
batch_size = 128
train_loader = DataLoader(cifar10_subset, batch_size=batch_size, shuffle=False)

# Call spectral hashing initialization
spectral_hash = SpectralHash.SpectralHashing(
    data_source=train_loader,
    num_bits=32,
    k_neighbors=100,
    n_components=None,
    batch_size=batch_size,
    device=None,
)

# Test image, replace with a picture that you have saved
query_image_path = "spectral-hashing\\frog2.jpg"
if not os.path.isfile(query_image_path):
    print(f"Query image not found: {query_image_path}")
else:
    # Computes mAP
    spectral_hash.compute_map()
    query_image = Image.open(query_image_path).convert("RGB")

    # Queries the picture and retrieves similar images
    retrieved_images, retrieved_distances = spectral_hash.query(
        image=query_image,
        num_neighbors=10,
        visualize=True,
    )