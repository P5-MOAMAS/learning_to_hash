import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from LSH_v4 import LSH


# Set the number of images to use
num_images_to_use = 50000

# Define transformation: Convert to tensor and normalize
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

# Load CIFAR-10 dataset
cifar10_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Set the batch size
batch_size = 256

# Adjust num_images_to_use to avoid small last batch
num_images_to_use = (num_images_to_use // batch_size) * batch_size
print(f"Using {num_images_to_use} images for LSH.")

# Create a DataLoader for the CIFAR-10 dataset
train_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False)

# Initialize lists to store the images and labels
images = []
labels = []

# Iterate over the dataset to collect images and labels in batches
print("Loading images...")
for image_batch, label_batch in tqdm(train_loader, desc="Loading data"):
    images.extend(image_batch.numpy().transpose(0, 2, 3, 1))
    labels.extend(label_batch.numpy())

    # Check if we've reached the desired number of images
    if len(images) >= num_images_to_use:
        images = images[:num_images_to_use]  # Trim to the required number
        labels = labels[:num_images_to_use]  # Trim to the required number
        break

# Convert the lists of images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)
print(f"Loaded {len(images)} images.")

# --- New Code to Remove Query Image ---
# Select the query image and label
query_image = images[0]
query_label = labels[0]

# Remove the query image and label from the dataset
images = np.delete(images, 0, axis=0)
labels = np.delete(labels, 0, axis=0)
print("Removed the query image from the dataset.")
# ---------------------------------------

# Parameters to customize
num_tables = 5  # Number of hash tables
num_bits_per_table = 10  # Number of hash functions (bits) per table
pca_components = 50  # Number of PCA components

# Initialize and run the LSH
image_lsh = LSH(
    images,
    labels,
    num_tables=num_tables,
    num_bits_per_table=num_bits_per_table,
    pca_components=pca_components,
)
# Query example
k = 10  # Number of nearest neighbors to return
similar_indices = image_lsh.query(query_image, k)

# Visualize the query image and similar images
image_lsh.visualize_images(similar_indices, query_image=query_image)
