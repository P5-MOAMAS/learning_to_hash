import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from models.lsh import Lsh
from tqdm import tqdm
from metrics import metrics_framework
from metrics import feature_loader

# Define transformation: Convert to tensor and normalize
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Ensure images are of consistent size
        transforms.ToTensor(),
    ]
)

# Load the features from the CIFAR-10 dataset
fl = feature_loader.FeatureLoader("cifar-10")
cifar10_validation = fl.validation
features = []
for i in range(len(fl.training)):
    features.append(fl.training[i][1])
del fl

# Flatten the images for compatibility with LSH (each image as a 1D feature vector)
features = np.array(features)

# Set LSH parameters
num_tables = 5
num_bits_per_table = 16
pca_components = 50

# Initialize LSH with the features data
image_lsh = Lsh.LSH(
    features,
    num_tables=num_tables,
    num_bits_per_table=num_bits_per_table,
    pca_components=pca_components,
)

# Select a query image and remove it from the dataset
query_image = features[0]
features = np.delete(features, 0, axis=0)
print("Removed the query image from the dataset.")

# Query LSH to find hash codes for the query image
query_hash_codes = image_lsh.query(query_image)
print("Hash codes for query image:", query_hash_codes)
metrics_framework.calculate_metrics(image_lsh.query, cifar10_validation, False)