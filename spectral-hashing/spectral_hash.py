import gc
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
from torchvision.models import ResNet18_Weights


class SpectralHashing:
    """
    Spectral Hashing class for generating binary hash codes using spectral hashing with
    graph-based feature representation. This class preprocesses image data, reduces
    dimensionality, constructs a similarity graph, and generates binary codes for fast
    approximate nearest neighbor search.
    """

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
        """
        Loads a pre-trained VGG16 model and prepares it as a feature extractor by removing
        the final classifier layers.
        """

        # Load a pre-trained ResNet18 model with the most up-to-date weights
        self.vgg = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.vgg = self.vgg.to(self.device)
        self.vgg.eval()  # Set to evaluation mode

        # Remove the final fully connected layer to get features
        self.feature_extractor = torch.nn.Sequential(*list(self.vgg.children())[:-1])

        # Optionally, add a flatten layer if needed
        self.feature_extractor.add_module("flatten", torch.nn.Flatten())

    def _load_and_preprocess_images(self, data_source):
        """
        Loads images from the specified data source, preprocesses them, and extracts
        feature vectors using the feature extractor.

        Parameters:
            data_source (DataLoader or str): Source of the data, either a DataLoader or a path.
        """
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
        max_images = 30000  # Adjust this based on your RAM capacity
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
        """
        Performs dimensionality reduction on image features using Incremental PCA.
        Reduces the feature dimensionality to the specified number of components (n_components).
        """
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

    # FEATURS AND LABELS
    # def _build_similarity_graph(self):
    #     """
    #     Constructs a k-Nearest Neighbor (k-NN) graph using a combination of features and labels for similarity.
    #     """
    #     print("Building k-NN graph with features and labels...")
    #     k = self.k_neighbors  # Number of nearest neighbors
    #     nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(
    #         self.features_pca
    #     )
    #     distances, indices = nbrs.kneighbors(self.features_pca)

    #     # Exclude the point itself in the neighbors
    #     indices = indices[:, 1:]  # Exclude the first column (index of self)

    #     # Initialize combined similarity
    #     combined_affinity = np.zeros((self.features_pca.shape[0], k))
    #     if self.labels is not None:
    #         for i, neighbors in enumerate(indices):
    #             for j, neighbor in enumerate(neighbors):
    #                 feature_similarity = np.exp(-(distances[i, j] ** 2))
    #                 label_similarity = (
    #                     1 if self.labels[i] == self.labels[neighbor] else 0
    #                 )
    #                 combined_affinity[i, j] = feature_similarity + label_similarity

    #     # Build sparse affinity matrix W
    #     rows = np.repeat(np.arange(self.features_pca.shape[0]), k)
    #     cols = indices.flatten()
    #     data = combined_affinity.flatten()
    #     W = csr_matrix(
    #         (data, (rows, cols)),
    #         shape=(self.features_pca.shape[0], self.features_pca.shape[0]),
    #     )

    #     # Symmetrize the affinity matrix
    #     self.W = 0.5 * (W + W.T)
    #     print("Constructed symmetric affinity matrix using features and labels.")

    # LABELS ONLY
    def _build_similarity_graph(self):
        """
        Constructs a k-Nearest Neighbor (k-NN) graph using labels for similarity,
        enhancing similarity scores between same-class samples.
        """
        print("Building k-NN graph with labels...")
        k = self.k_neighbors  # Number of nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(
            self.features_pca
        )
        distances, indices = nbrs.kneighbors(self.features_pca)

        # Exclude the point itself in the neighbors
        indices = indices[:, 1:]  # Exclude the first column (index of self)

        # Initialize label-based similarity (1 if labels match, 0 otherwise)
        label_affinity = np.zeros((self.features_pca.shape[0], k))
        if self.labels is not None:
            for i, neighbors in enumerate(indices):
                for j, neighbor in enumerate(neighbors):
                    if self.labels[i] == self.labels[neighbor]:
                        label_affinity[i, j] = 1

        # Build sparse affinity matrix W
        rows = np.repeat(np.arange(self.features_pca.shape[0]), k)
        cols = indices.flatten()
        data = label_affinity.flatten()
        W = csr_matrix(
            (data, (rows, cols)),
            shape=(self.features_pca.shape[0], self.features_pca.shape[0]),
        )

        # Symmetrize the affinity matrix
        self.W = 0.5 * (W + W.T)
        print("Constructed symmetric affinity matrix considering labels.")

    def _compute_laplacian_eigenvectors(self):
        """
        Computes the normalized Laplacian matrix and calculates its eigenvectors. These
        eigenvectors serve as the basis for generating binary hash codes, potentially
        influenced by label distributions.
        """
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
        """
        Learns hash functions using Ridge Regression to map reduced features to eigenvectors.
        This step enables generalization to unseen data.
        """
        print("Learning hash functions using Ridge Regression...")
        self.hash_functions = Ridge(alpha=1.0)
        self.hash_functions.fit(self.features_pca, self.eigenvectors)
        print("Learned hash functions for all bits jointly.")

    def _compute_hash_codes(self, features):
        """
        Computes binary hash codes for given features by projecting them through learned
        hash functions.

        Parameters:
            features (numpy array): The input feature vectors.

        Returns:
            numpy array: Binary hash codes.
        """
        projected = self.hash_functions.predict(features)
        binary_codes = (projected > 0).astype(int)
        return binary_codes

    def query(self, image, num_neighbors=10, visualize=False):
        """
        Queries the dataset with a new image, returning the most similar images and their
        distances.

        Parameters:
            image (PIL.Image, torch.Tensor, or similar): The query image. If in tensor format, assumes it's already preprocessed.
            num_neighbors (int): The number of nearest neighbors to retrieve.
            visualize (bool): Whether to display the query and retrieved images.

        Returns:
            retrieved_images (numpy array): Array of retrieved image tensors.
            retrieved_distances (numpy array): Hamming distances of retrieved images.
        """
        # Check if the input is a PIL image or already a tensor
        if isinstance(image, Image.Image):
            # Convert to tensor using transform
            query_image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            # Assume tensor is already in the correct format and preprocessed
            query_image_tensor = image.unsqueeze(0).to(self.device)
        else:
            raise ValueError(
                "Image must be a PIL.Image or a preprocessed torch.Tensor."
            )

        with torch.no_grad():
            query_feature = self.feature_extractor(query_image_tensor)
            query_feature = query_feature.view(1, -1).cpu().numpy()

        query_feature_pca = self.pca.transform(query_feature)
        query_binary_code = self._compute_hash_codes(query_feature_pca)

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
        """
        Visualizes the query image and the retrieved images along with their Hamming
        distances to the query.

        Parameters:
            query_image_tensor (torch.Tensor): The processed query image tensor.
            retrieved_images (list): List of retrieved image tensors.
            retrieved_distances (list): List of Hamming distances of the retrieved images.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Process query image for visualization
        query_image_vis = query_image_tensor.squeeze(0).numpy()
        query_image_vis = query_image_vis.transpose(1, 2, 0)
        query_image_vis = std * query_image_vis + mean
        query_image_vis = np.clip(query_image_vis, 0, 1)

        # Process retrieved images for visualization
        retrieved_images_vis = []
        for img in retrieved_images:
            img = img.numpy().transpose(1, 2, 0)
            img = std * img + mean
            img = np.clip(img, 0, 1)
            retrieved_images_vis.append(img)

        num_retrieved = len(retrieved_images)
        num_cols = min(
            num_retrieved + 1, 6
        )  # Limit to 6 columns for better visualization
        num_rows = (num_retrieved + 1) // num_cols + 1

        fig = plt.figure(figsize=(15, 5 * num_rows))
        gs = GridSpec(num_rows, num_cols, figure=fig)

        # Display the query image
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(query_image_vis)
        ax.set_title("Query Image")
        ax.axis("off")

        # Display the retrieved images with their Hamming distances
        for i, (img, dist) in enumerate(zip(retrieved_images_vis, retrieved_distances)):
            row = (i + 1) // num_cols
            col = (i + 1) % num_cols
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.set_title(f"Hamming Distance: {dist}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def compute_map(self, sample_size=None):
        """
        Computes the mean average precision (mAP) for retrieval accuracy, measuring how
        accurately similar images are retrieved for various queries.

        Parameters:
            sample_size (int or None): Number of samples for mAP calculation. If None,
            uses the entire dataset.

        Returns:
            float: The mean average precision score.
        """
        if self.labels is None:
            raise ValueError("Labels are required to compute mAP.")

        n_samples = self.binary_codes.shape[0]
        sample_size = sample_size or n_samples
        query_indices = np.arange(n_samples)[:sample_size]
        aps = []

        for query_idx in tqdm(query_indices, desc="Computing mAP"):
            query_code = self.binary_codes[query_idx]
            query_label = self.labels[query_idx]

            hamming_distances = np.sum(
                np.bitwise_xor(self.binary_codes, query_code), axis=1
            )
            hamming_distances[query_idx] = np.max(hamming_distances) + 1

            sorted_indices = np.argsort(hamming_distances)
            sorted_labels = self.labels[sorted_indices]
            relevant = (sorted_labels == query_label).astype(int)
            precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = np.sum(precision_at_k * relevant) / np.sum(relevant)
            aps.append(ap)

        mean_ap = np.mean(aps)
        print(f"Mean Average Precision (mAP): {mean_ap:.3f}")
        return mean_ap


class ImageFolderDataset(Dataset):
    """
    Custom Dataset class for loading images from a folder.

    Parameters:
        image_paths (list): List of paths to the images.
        transform (callable, optional): Transformations to apply to each image.
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and returns an image from the dataset at the specified index after applying
        transformations.

        Parameters:
            idx (int): Index of the image to load.

        Returns:
            torch.Tensor: Transformed image.
        """
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

subset_indices = list(range(30000))
cifar10_subset = Subset(cifar10_dataset, subset_indices)

batch_size = 128
train_loader = DataLoader(cifar10_subset, batch_size=batch_size, shuffle=False)

spectral_hash = SpectralHashing(
    data_source=train_loader,
    num_bits=8,
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
