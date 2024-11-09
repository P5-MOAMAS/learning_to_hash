from sklearn.decomposition import IncrementalPCA
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LSH:
    def __init__(
        self, images, labels, num_tables=5, num_bits_per_table=10, pca_components=None
    ):
        """
        Initialize the LSH (Locality-Sensitive Hashing) class and execute the LSH process.

        Parameters:
            images (np.ndarray): The dataset of images to index.
            labels (np.ndarray): The labels corresponding to the images.
            num_tables (int): Number of hash tables to use.
            num_bits_per_table (int): Number of bits (hash functions) per table.
            pca_components (int, optional): Number of principal components for PCA dimensionality reduction.
        """
        self.images = images
        self.labels = labels
        self.num_tables = num_tables
        self.num_bits_per_table = num_bits_per_table
        self.pca_components = pca_components or min(images.shape[0], images.shape[1])
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        self.hash_functions = []  # List of lists of hash functions per table
        self.pca = None

        # Execute the LSH process during initialization
        self.preprocess_data()
        self.hash_images()
        self.calculate_map()

    def preprocess_data(self):
        """
        Preprocess the data by flattening the images and performing incremental PCA
        for dimensionality reduction.
        """
        print("Preprocessing data...")
        # Flatten the images into a 2D array: (num_samples, num_features)
        self.flattened_images = self.images.reshape(self.images.shape[0], -1)
        batch_size = 256
        total_samples = self.flattened_images.shape[0]

        # Ensure the batch size is at least equal to the number of PCA components
        if batch_size < self.pca_components:
            batch_size = self.pca_components

        # Adjust total_samples to avoid a small last batch
        num_batches = total_samples // batch_size
        total_samples = num_batches * batch_size
        self.flattened_images = self.flattened_images[:total_samples]
        self.labels = self.labels[:total_samples]

        # Initialize Incremental PCA
        self.pca = IncrementalPCA(
            n_components=self.pca_components, batch_size=batch_size
        )

        # Fit the PCA incrementally on batches
        for i in tqdm(
            range(0, total_samples, batch_size),
            desc="Fitting IncrementalPCA",
        ):
            batch = self.flattened_images[i : i + batch_size]
            self.pca.partial_fit(batch)

        # Transform the data incrementally
        self.reduced_images = []
        for i in tqdm(
            range(0, total_samples, batch_size),
            desc="Transforming data with PCA",
        ):
            batch = self.flattened_images[i : i + batch_size]
            reduced_batch = self.pca.transform(batch)
            self.reduced_images.append(reduced_batch)
        self.reduced_images = np.vstack(self.reduced_images)

    def create_hash_function(self):
        """
        Create a single hash function using a random hyperplane.

        Returns:
            function: A hash function that takes an input vector x and returns 0 or 1.
        """
        # Generate a random hyperplane by sampling a random vector from a normal distribution
        random_vector = np.random.randn(self.reduced_images.shape[1])
        # The hash function computes the sign of the dot product between input vector and random vector
        # If the dot product is positive, returns 1; otherwise, returns 0
        return lambda x: int(np.dot(x, random_vector) > 0)

    def hash_images(self):
        """
        Hash the images into multiple hash tables using the generated hash functions.
        """
        print("Hashing images...")
        self.hash_functions = []
        for t in range(self.num_tables):
            # Create a list of hash functions (random hyperplanes) for each table
            table_hash_functions = [
                self.create_hash_function() for _ in range(self.num_bits_per_table)
            ]
            self.hash_functions.append(table_hash_functions)
        for idx, image in tqdm(
            enumerate(self.reduced_images),
            total=len(self.reduced_images),
            desc="Hashing images into tables",
        ):
            for t in range(self.num_tables):
                # Compute the hash key for the current image in table t
                # The hash key is a tuple of bits obtained by applying each hash function
                hash_key = tuple(hf(image) for hf in self.hash_functions[t])
                # Store the index of the image in the hash table under the computed hash key
                self.hash_tables[t][hash_key].append(idx)

    def calculate_map(self):
        """
        Calculate the Mean Average Precision (MAP) of the LSH retrieval.

        Returns:
            float: The MAP score.
        """
        print("Calculating Mean Average Precision (MAP)...")
        average_precisions = []
        for idx, image in tqdm(
            enumerate(self.reduced_images),
            total=len(self.reduced_images),
            desc="Calculating MAP",
        ):
            candidate_indices = set()
            for t in range(self.num_tables):
                # Compute the hash key for the query image in table t
                query_hash = tuple(hf(image) for hf in self.hash_functions[t])
                # Retrieve indices of images with the same hash key
                candidate_indices.update(self.hash_tables[t].get(query_hash, []))
            candidate_indices.discard(idx)  # Remove the query image itself
            if not candidate_indices:
                # No candidates found, average precision is zero
                average_precisions.append(0)
                continue
            candidate_indices_list = list(candidate_indices)
            candidate_images = self.reduced_images[candidate_indices_list]
            # Compute Euclidean distances between the query image and candidate images
            distances = np.linalg.norm(candidate_images - image, axis=1)
            # Sort candidates by distance (ascending)
            sorted_indices = np.argsort(distances)
            similar_images = [candidate_indices_list[i] for i in sorted_indices]
            # Count the number of relevant items (same label as the query image)
            relevant_items = sum(
                1 for i in similar_images if self.labels[i] == self.labels[idx]
            )
            if relevant_items == 0:
                average_precisions.append(0)
                continue
            precision_at_k = []
            num_relevant = 0
            for rank, i in enumerate(similar_images):
                if self.labels[i] == self.labels[idx]:
                    num_relevant += 1
                    precision = num_relevant / (rank + 1)
                    precision_at_k.append(precision)
            if precision_at_k:
                # Average precision for this query is the mean of precisions at each relevant retrieval
                average_precisions.append(np.mean(precision_at_k))
            else:
                average_precisions.append(0)
        # Mean Average Precision over all queries
        self.map_score = np.mean(average_precisions)
        print(f"Mean Average Precision (MAP): {self.map_score}")
        return self.map_score

    def query(self, image, k=5):
        """
        Query the LSH index with a new image and retrieve the top k similar images.

        Parameters:
            image (np.ndarray): The query image.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            list: List of indices of the top k similar images.
        """
        # Flatten and reduce the query image
        image_flattened = image.reshape(1, -1)
        image_reduced = self.pca.transform(image_flattened)
        candidate_indices = set()
        for t in range(self.num_tables):
            # Compute the hash key for the query image in table t
            hash_key = tuple(hf(image_reduced[0]) for hf in self.hash_functions[t])
            # Retrieve indices of images with the same hash key
            candidate_indices.update(self.hash_tables[t].get(hash_key, []))
        if not candidate_indices:
            print("No candidates found for the query image.")
            return []
        candidate_indices_list = list(candidate_indices)
        candidate_images = self.reduced_images[candidate_indices_list]
        # Compute Euclidean distances between the query image and candidate images
        distances = np.linalg.norm(candidate_images - image_reduced, axis=1)
        # Get indices of the top k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        return [candidate_indices_list[i] for i in nearest_indices]

    def visualize_images(self, indices, query_image=None):
        """
        Visualize the images corresponding to the given indices.

        Parameters:
            indices (list): List of image indices to visualize.
            query_image (np.ndarray, optional): The query image to display alongside results.
        """
        num_images = len(indices) + (1 if query_image is not None else 0)
        plt.figure(figsize=(15, 5))
        if query_image is not None:
            plt.subplot(1, num_images, 1)
            plt.imshow(query_image)
            plt.title("Query Image")
            plt.axis("off")
            start_idx = 2
        else:
            start_idx = 1
        for i, idx in enumerate(indices, start=start_idx):
            plt.subplot(1, num_images, i)
            plt.imshow(self.images[idx])
            plt.title(f"Label: {self.labels[idx]}")
            plt.axis("off")
        plt.show()
