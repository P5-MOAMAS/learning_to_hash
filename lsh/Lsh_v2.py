#Version 2 of LSH this time using the math as given by the theory about hamming space, l1 and l2 norms and random projection.

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import hamming, cosine


##### Images, cifar-10 using pytorches loader

num_images_to_use = 40000

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

limited_train_dataset = torch.utils.data.Subset(train_dataset, range(num_images_to_use))

train_loader = DataLoader(limited_train_dataset, batch_size=32, shuffle=False)

images = []
for i, (image_batch, _) in enumerate(train_loader):
    for img in image_batch:
        images.append(img.numpy().transpose(1, 2, 0))
        if len(images) >= num_images_to_use:
            break
    if len(images) >= num_images_to_use:
        break

images = np.array(images)
print(f"Loaded {len(images)} images, each resized and flattened.")




##### LSH class

class LSH:
    def __init__(self, num_planes, num_tables, images):
        self.num_planes = num_planes
        self.num_tables = num_tables
        self.images = images
        self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
        self.planes = [self._generate_random_planes(images.shape[1] * images.shape[2] * images.shape[3]) for _ in range(num_tables)]

        self.store_images(images)

    def _generate_random_planes(self, dim):
        return np.random.randn(self.num_planes, dim)

    def _hash(self, planes, image):
        image = image.flatten()
        return ''.join(['1' if np.dot(plane, image) > 0 else '0' for plane in planes])

    def store_images(self, images):
        for id, image in enumerate(images):
            for table_id in range(self.num_tables):
                hash_key = self._hash(self.planes[table_id], image)
                self.hash_tables[table_id][hash_key].append(id)

    def _euclidean_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def _hamming_distance(self, vec1, vec2):
        return hamming(vec1, vec2)

    def _cosine_distance(self, vec1, vec2):
        return cosine(vec1, vec2)

    def query(self, query_image, n_neighbors=5, distance_metric="cosine"):
        candidate_indices = set()
        query_image_flat = query_image.flatten()

        for table_id in range(self.num_tables):
            hash_key = self._hash(self.planes[table_id], query_image)
            candidate_indices.update(self.hash_tables[table_id].get(hash_key, []))

        distances = []
        for idx in candidate_indices:
            image_flat = self.images[idx].flatten()

            if distance_metric == "euclidean":
                dist = self._euclidean_distance(query_image_flat, image_flat)
            elif distance_metric == "hamming":
                query_hash = self._hash(self.planes[0], query_image)
                candidate_hash = self._hash(self.planes[0], self.images[idx])
                dist = self._hamming_distance(list(query_hash), list(candidate_hash))
            elif distance_metric == "cosine":
                dist = self._cosine_distance(query_image_flat, image_flat)
            else:
                raise ValueError("Unsupported distance metric. Use 'euclidean', 'hamming', or 'cosine'.")

            distances.append((idx, dist))

        return sorted(distances, key=lambda x: x[1])[:n_neighbors]

##### Testing

lsh = LSH(15,10,images = images)


query_image = images[4]
result = lsh.query(query_image,5,"cosine")


fig, axes = plt.subplots(1, 6, figsize=(20, 5))

axes[0].imshow(query_image)
axes[0].set_title('Query Image')
axes[0].axis('off')

for i, (neighbor_id, distance) in enumerate(result):
    neighbor_image = images[neighbor_id].reshape(32,32,3)
    axes[i+1].imshow(neighbor_image)
    axes[i+1].set_title(f'Neighbor {i+1}\nDistance: {distance:.2f}')
    axes[i+1].axis('off')


plt.tight_layout()
plt.show()
