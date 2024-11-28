import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from Lsh import LSH  # Assume the LSH class is saved as Lsh.py
from tqdm import tqdm

# Set the number of images to use
num_images_to_use = 50000

# Define transformation: Convert to tensor and normalize
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Ensure images are of consistent size
        transforms.ToTensor(),
    ]
)

# Load CIFAR-10 dataset
cifar10_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Set the batch size for loading images
batch_size = 256

# Adjust num_images_to_use to avoid a small last batch
num_images_to_use = (num_images_to_use // batch_size) * batch_size
print(f"Using {num_images_to_use} images for LSH.")

# Create a DataLoader for the CIFAR-10 dataset
train_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False)

# Initialize lists to store the images and labels
images = []
labels = []

# Collect images and labels in batches
print("Loading images...")
for image_batch, label_batch in tqdm(train_loader, desc="Loading data"):
    images.extend(image_batch.numpy().transpose(0, 2, 3, 1))

    # Stop when we have the required number of images
    if len(images) >= num_images_to_use:
        images = images[:num_images_to_use]
        break

# Convert lists to numpy arrays
images = np.array(images)
print(f"Loaded {len(images)} images.")

# Flatten the images for compatibility with LSH (each image as a 1D feature vector)
images = images.reshape(images.shape[0], -1)

# Set LSH parameters
num_tables = 5
num_bits_per_table = 64
pca_components = 50

# Initialize LSH with the images data
image_lsh = LSH(
    images,
    num_tables=num_tables,
    num_bits_per_table=num_bits_per_table,
    pca_components=pca_components,
)

# Select a query image and remove it from the dataset
query_image = images[0]
query_label = labels[0]
images = np.delete(images, 0, axis=0)
labels = np.delete(labels, 0, axis=0)
print("Removed the query image from the dataset.")

# Query LSH to find hash codes for the query image
query_hash_codes = image_lsh.query(query_image)
print("Hash codes for query image:", query_hash_codes)
