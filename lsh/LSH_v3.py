from sklearn.decomposition import PCA
from collections import defaultdict
import hashlib
from scipy.spatial.distance import cosine
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set the number of images to use
num_images_to_use = 10000

# Define transformation: Convert to tensor and normalize
transform = transforms.Compose(
    [
        transforms.Resize(
            (32, 32)
        ),  # Resize to a smaller size (e.g., 32x32) use smaller when not much ram is available.
        transforms.ToTensor(),
    ]
)

# Load STL-10 dataset (unlabeled data)
stl10_unlabeled_dataset = torchvision.datasets.STL10(
    root="./data", split="unlabeled", download=True, transform=transform
)

# Create a DataLoader for the unlabeled dataset
train_loader = DataLoader(stl10_unlabeled_dataset, batch_size=32, shuffle=False)

# Initialize an empty list to store the images
images = []
batch_count = 0

# Iterate over the dataset to collect images in batches
for image_batch, _ in train_loader:
    images.extend(
        image_batch.numpy().transpose(0, 2, 3, 1)
    )  # Convert tensors to numpy arrays (N, H, W, C) N number of batches height widht and number of channels
    batch_count += image_batch.size(0)

    # Check if we've reached the desired number of images
    if len(images) >= num_images_to_use:
        images = images[:num_images_to_use]  # Trim to the required number
        break

# Convert the list of images to a numpy array
images = np.array(images)
print(f"Loaded {len(images)} images, each resized and flattened.")


class LSH_min:
    def __init__(self, num_hashes, num_tables, images, pca_components=None):
        """
        Initialize the LSH_min with MinHash and PCA transformation.

        :param num_hashes: Number of hash functions for MinHash
        :param num_tables: Number of hash tables
        :param images: The dataset of images
        :param pca_components: Optional, number of PCA components (if None, defaults to num_hashes)
        """
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        self.images = self._normalize_images(images)  # Normalize the images first
        self.pca_components = (
            pca_components if pca_components is not None else num_hashes
        )
        self.pca = PCA(
            n_components=self.pca_components
        )  # Apply PCA with specified or default components
        self.transformed_images = self.pca.fit_transform(
            self._flatten_images(self.images)
        )  # PCA-transformed images

        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
        self.minhash_permutations = self._generate_permutations(num_hashes)

        self.store_images(self.transformed_images)

    def _normalize_images(self, images):
        """Normalize images to have pixel values in the range [0, 1]."""
        return np.array(images) / 255.0

    def _flatten_images(self, images):
        """Flatten the images for PCA transformation."""
        return np.array([image.flatten() for image in images])

    def _generate_permutations(self, num_hashes):
        """Generate random permutations for MinHash."""
        permutations = []
        max_value = 2**32 - 1
        for _ in range(num_hashes):
            a, b = np.random.randint(1, max_value, size=2)
            permutations.append((a, b))
        return permutations

    def _minhash_signature(self, image):
        """Generate MinHash signature for the image."""
        image_set = set(np.nonzero(image)[0])  # Use non-zero indices as the "set"
        signature = []

        for a, b in self.minhash_permutations:
            min_hash_value = min(((a * x + b) % (2**32 - 1)) for x in image_set)
            signature.append(min_hash_value)

        return np.array(signature)

    def _hash_buckets(self, minhash_signature):
        """Map MinHash signature to a bucket using a second hash function."""
        return hashlib.md5(minhash_signature.tobytes()).hexdigest()

    def store_images(self, images):
        """Store images in hash tables using MinHash signatures."""
        for id, image in enumerate(images):
            minhash_signature = self._minhash_signature(image)
            for table_id in range(self.num_tables):
                bucket = self._hash_buckets(minhash_signature)
                self.hash_tables[table_id][bucket].append(id)

    def _euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def _cosine_distance(self, vec1, vec2):
        return cosine(vec1, vec2)

    def query(self, query_image, n_neighbors=5, distance_metric="cosine"):
        """Query the nearest neighbors of a given image."""
        candidate_indices = set()
        query_image_normalized = self._normalize_images(
            [query_image]
        )  # Normalize query image
        query_image_flat = self.pca.transform(
            query_image_normalized.flatten().reshape(1, -1)
        )[0]
        minhash_signature = self._minhash_signature(query_image_flat)

        for table_id in range(self.num_tables):
            bucket = self._hash_buckets(minhash_signature)
            candidate_indices.update(self.hash_tables[table_id].get(bucket, []))

        # Compute distances for all candidates
        distances = []
        for idx in candidate_indices:
            image_flat = self.transformed_images[idx]

            if distance_metric == "euclidean":
                dist = self._euclidean_distance(query_image_flat, image_flat)
            elif distance_metric == "cosine":
                dist = self._cosine_distance(query_image_flat, image_flat)
            else:
                raise ValueError(
                    "Unsupported distance metric. Use 'euclidean' or 'cosine'."
                )

            distances.append((idx, dist))

        return sorted(distances, key=lambda x: x[1])[:n_neighbors]


##### TESTING

# config with highest F1score so far
# Number of hash bits 37
# Number of buckets 42
# Number of PCA components 300
# Number of neighbors 6

Lsh_min = LSH_min(
    37, 42, images=images, pca_components=300
)  # config with highest F1score so far

query_image = images[7]
result = Lsh_min.query(query_image, 6, "euclidean")


fig, axes = plt.subplots(1, 7, figsize=(20, 6))

axes[0].imshow(query_image.reshape(32, 32, 3))
axes[0].set_title("Query Image")
axes[0].axis("off")

for i, (neighbor_id, distance) in enumerate(result):
    neighbor_image = images[neighbor_id].reshape(32, 32, 3)
    axes[i + 1].imshow(neighbor_image)
    axes[i + 1].set_title(f"Neighbor {i+1}\nDistance: {distance:.2f}")
    axes[i + 1].axis("off")


plt.tight_layout()
plt.show()
