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


class SpectralHashing:
    def __init__(
        self,
        data_source,
        num_bits=64,
        k_neighbors=10,
        n_components=None,
        batch_size=128,
        device=None,
    ):
        """
        Initializes the SpectralHashing class.

        Parameters:
            data_source (DataLoader or str): A PyTorch DataLoader or a path to a folder with images.
            num_bits (int): The number of hash bits.
            k_neighbors (int): The number of nearest neighbors for graph construction.
            n_components (int): The number of PCA components. Must be >= num_bits.
            batch_size (int): The batch size for processing images.
            device (str): Device to use ('cuda' or 'cpu'). If None, will use 'cuda' if available.
        """
        self.num_bits = num_bits
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_components = n_components
        self.labels = None

        if self.n_components is not None and self.n_components < self.num_bits:
            raise ValueError(
                "Number of PCA components must be greater than or equal to number of hash bits."
            )

        # Set up transformation
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Normalization parameters for ImageNet
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Load pre-trained model
        self._load_model()

        # Load and preprocess images
        self._load_and_preprocess_images(data_source)

        # Dimensionality reduction with PCA
        self._perform_pca()

        # Build similarity graph
        self._build_similarity_graph()

        # Compute Laplacian and eigenvectors
        self._compute_laplacian_eigenvectors()

        # Learn hash functions
        self._learn_hash_functions()

        # Clean up
        gc.collect()

    def _load_model(self):
        # Load a pre-trained ResNet18 model
        self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg = self.vgg.to(self.device)
        self.vgg.eval()  # Set to evaluation mode

        # Remove the final fully connected layer to get features
        self.feature_extractor = torch.nn.Sequential(*list(self.vgg.features))

    def _load_and_preprocess_images(self, data_source):
        print("Loading and preprocessing images...")
        if isinstance(data_source, DataLoader):
            dataloader = data_source
        elif isinstance(data_source, str):
            image_paths = [
                os.path.join(data_source, fname)
                for fname in os.listdir(data_source)
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
            ]
            dataset = ImageFolderDataset(image_paths, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError(
                "data_source must be a DataLoader or a path to a folder with images."
            )

        features_list = []
        self.image_tensors = []
        labels_list = []

        batch_count = 0
        max_images = 10000  # Adjust this based on your RAM capacity
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                if isinstance(data_source, DataLoader):
                    images, labels = batch
                    labels = labels.cpu().numpy()
                else:
                    images = batch
                    labels = None

                images = images.to(self.device)
                features = self.feature_extractor(images)
                features = features.view(features.size(0), -1)

                features_list.append(features.cpu().numpy())
                self.image_tensors.append(images.cpu())
                if labels is not None:
                    labels_list.append(labels)

                batch_count += images.size(0)
                if batch_count >= max_images:
                    break

        self.features = np.vstack(features_list)[:max_images]
        self.image_tensors = torch.cat(self.image_tensors)[:max_images]
        if labels_list:
            self.labels = np.concatenate(labels_list)[:max_images]
        print(f"Extracted features for {self.features.shape[0]} images.")

    def _perform_pca(self):
        print("Reducing dimensionality with PCA...")
        if self.n_components is None:
            self.n_components = max(self.num_bits, 100)

        if self.n_components < self.num_bits:
            raise ValueError(
                "Number of PCA components must be greater than or equal to number of hash bits."
            )

        self.pca = IncrementalPCA(n_components=self.n_components, whiten=True)

        batch_size = 1000
        n_samples = self.features.shape[0]

        for i in tqdm(range(0, n_samples, batch_size), desc="Fitting Incremental PCA"):
            batch_features = self.features[i : i + batch_size]
            self.pca.partial_fit(batch_features)

        features_pca_list = []
        for i in tqdm(range(0, n_samples, batch_size), desc="Transforming features"):
            batch_features = self.features[i : i + batch_size]
            batch_features_pca = self.pca.transform(batch_features)
            features_pca_list.append(batch_features_pca)

        self.features_pca = np.vstack(features_pca_list)

        print(
            f"Reduced features to {self.n_components} dimensions using PCA with whitening."
        )

    def _build_similarity_graph(self):
        print("Building k-NN graph...")
        k = self.k_neighbors  # Number of nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(
            self.features_pca
        )
        distances, indices = nbrs.kneighbors(self.features_pca)

        # Exclude the point itself in the neighbors
        distances = distances[:, 1:]  # Exclude the first column (distance to self)
        indices = indices[:, 1:]  # Exclude the first column (index of self)

        # Compute the affinity matrix W using a Gaussian kernel
        sigma = np.mean(distances)
        print(f"Using sigma={sigma:.4f} for Gaussian kernel.")
        affinities = np.exp(-(distances**2) / (2 * sigma**2))
        rows = np.repeat(np.arange(self.features_pca.shape[0]), k)
        cols = indices.flatten()
        data = affinities.flatten()
        W = csr_matrix(
            (data, (rows, cols)),
            shape=(self.features_pca.shape[0], self.features_pca.shape[0]),
        )

        # Symmetrize the affinity matrix
        self.W = 0.5 * (W + W.T)
        print("Constructed symmetric affinity matrix using Gaussian kernel.")

    def _compute_laplacian_eigenvectors(self):
        print("Computing normalized Laplacian matrix...")
        from scipy.sparse import csgraph

        L = csgraph.laplacian(self.W, normed=True)
        print("Computed the normalized Laplacian matrix.")

        print("Computing eigenvectors of the Laplacian...")
        num_bits = self.num_bits
        eigenvalues, eigenvectors = eigsh(L, k=num_bits + 1, which="SM")

        # Exclude the first eigenvector (corresponding to eigenvalue zero)
        self.eigenvectors = eigenvectors[:, 1 : num_bits + 1]
        print(f"Computed the top {num_bits} non-trivial eigenvectors of the Laplacian.")

        # Binarize the eigenvectors to get hash codes
        self.binary_codes = (self.eigenvectors > 0).astype(int)
        print(f"Generated binary codes with {num_bits} bits.")

    def _learn_hash_functions(self):
        print("Learning hash functions using Ridge Regression...")
        self.hash_functions = Ridge(alpha=1.0)
        self.hash_functions.fit(self.features_pca, self.eigenvectors)
        print("Learned hash functions for all bits jointly.")

    def _compute_hash_codes(self, features):
        projected = self.hash_functions.predict(features)
        binary_codes = (projected > 0).astype(int)
        return binary_codes

    def query(self, image, num_neighbors=10, visualize=False):
        """
        Queries the dataset with a new image.

        Parameters:
            image (PIL.Image or similar): The query image (not preprocessed).
            num_neighbors (int): The number of nearest neighbors to retrieve.
            visualize (bool): Whether to display the query and retrieved images.

        Returns:
            retrieved_images (numpy array): Array of retrieved image tensors.
            retrieved_distances (numpy array): Hamming distances of retrieved images.
        """
        # Preprocess the query image
        query_image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            query_feature = self.feature_extractor(query_image_tensor)
            query_feature = query_feature.view(1, -1).cpu().numpy()

        # Apply PCA transformation
        query_feature_pca = self.pca.transform(query_feature)

        # Compute hash code
        query_binary_code = self._compute_hash_codes(query_feature_pca)

        # Retrieve similar images
        hamming_distances = np.sum(
            np.bitwise_xor(self.binary_codes, query_binary_code), axis=1
        )
        sorted_indices = np.argsort(hamming_distances)
        top_k_indices = sorted_indices[:num_neighbors]
        retrieved_images = self.image_tensors[top_k_indices]
        retrieved_distances = hamming_distances[top_k_indices]

        if visualize:
            self._visualize_results(
                query_image_tensor.cpu(), retrieved_images.cpu(), retrieved_distances
            )

        return retrieved_images, retrieved_distances

    def _visualize_results(
        self, query_image_tensor, retrieved_images, retrieved_distances
    ):
        # Unnormalize the query image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        query_image_vis = query_image_tensor.squeeze(0).numpy()
        query_image_vis = query_image_vis.transpose(1, 2, 0)  # Convert to HWC
        query_image_vis = std * query_image_vis + mean
        query_image_vis = np.clip(query_image_vis, 0, 1)

        retrieved_images_vis = []
        for img in retrieved_images:
            img = img.numpy().transpose(1, 2, 0)  # Convert to HWC
            img = std * img + mean
            img = np.clip(img, 0, 1)
            retrieved_images_vis.append(img)

        # Display the query image and retrieved images
        num_neighbors = len(retrieved_images_vis)
        cols = min(num_neighbors + 1, 6)
        plt.figure(figsize=(15, 5))
        plt.subplot(2, 6, 1)
        plt.imshow(query_image_vis)
        plt.title("Query Image")
        plt.axis("off")

        for i, (img, dist) in enumerate(zip(retrieved_images_vis, retrieved_distances)):
            plt.subplot(2, 6, i + 2)
            plt.imshow(img)
            plt.title(f"Hamming Distance: {dist}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def compute_map(self, sample_size=None):
        if self.labels is None:
            raise ValueError("Labels are required to compute mAP")

        n_samples = self.binary_codes.shape[0]

        if sample_size is None or sample_size > n_samples:
            sample_size = n_samples

        indices = np.arange(n_samples)
        query_indices = indices[:sample_size]

        APs = []

        for query_idx in tqdm(query_indices, desc="Computing mAP"):
            query_code = self.binary_codes[query_idx]
            query_label = self.labels[query_idx]

            hamming_distances = np.sum(
                np.bitwise_xor(self.binary_codes, query_code), axis=1
            )

            hamming_distances[query_idx] = np.max(hamming_distances) + 1

            ranking_list = np.argsort(hamming_distances)
            ranking_labels = self.labels[ranking_list]

            relevant = (ranking_labels == query_label).astype(int)
            cum = np.cumsum(relevant)
            precision_at_k = cum / (np.arange(len(relevant)) + 1)

            AP = np.sum(precision_at_k * relevant) / np.sum(relevant)
            APs.append(AP)

        mean_AP = np.mean(APs)
        print(f"Mean Average Precision (mAP): {mean_AP:.4f}")
        return mean_AP


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


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

subset_indices = list(range(10000))
cifar10_subset = Subset(cifar10_dataset, subset_indices)

batch_size = 128
train_loader = DataLoader(cifar10_subset, batch_size=batch_size, shuffle=False)

spectral_hash = SpectralHashing(
    data_source=train_loader,
    num_bits=32,
    k_neighbors=100,
    n_components=None,
    batch_size=batch_size,
    device=None,
)


query_image_path = "spectral-hashing\\frog2.jpg"
if not os.path.isfile(query_image_path):
    print(f"Query image not found: {query_image_path}")
else:
    spectral_hash.compute_map()
    query_image = Image.open(query_image_path).convert("RGB")

    retrieved_images, retrieved_distances = spectral_hash.query(
        image=query_image,
        num_neighbors=10,
        visualize=True,
    )
