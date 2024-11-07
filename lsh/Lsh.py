from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class LSH:
    def __init__(self, data, num_tables=5, num_bits_per_table=10, pca_components=None):
        """
        Initialize the LSH (Locality-Sensitive Hashing) class and execute the LSH process.

        Parameters:
            data (np.ndarray): The dataset of features to index (2D array, shape: (num_images, feature_length)).
            num_tables (int): Number of hash tables to use.
            num_bits_per_table (int): Number of bits (hash functions) per table.
            pca_components (int, optional): Number of principal components for PCA dimensionality reduction.
        """
        self.data = data
        self.num_tables = num_tables
        self.num_bits_per_table = num_bits_per_table
        self.pca_components = pca_components or min(data.shape[0], data.shape[1])
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
        self.hash_functions = []  # List of lists of hash functions per table
        self.pca = PCA(n_components=self.pca_components)

        # Apply PCA and hash the data
        self.apply_pca()
        self.hash_data()

    def apply_pca(self):
        """
        Apply PCA to the data for dimensionality reduction.
        """
        print("Applying PCA to data...")
        # Fit and transform the data with PCA in a single step
        self.reduced_data = self.pca.fit_transform(self.data)

    def create_hash_function(self):
        """
        Create a single hash function using a random hyperplane.

        Returns:
            function: A hash function that takes an input vector x and returns 0 or 1.
        """
        random_vector = np.random.randn(self.reduced_data.shape[1])
        return lambda x: int(np.dot(x, random_vector) > 0)

    def hash_data(self):
        """
        Hash the data into multiple hash tables using the generated hash functions.
        """
        print("Hashing data...")
        self.hash_functions = []
        for t in range(self.num_tables):
            # Create a list of hash functions (random hyperplanes) for each table
            table_hash_functions = [
                self.create_hash_function() for _ in range(self.num_bits_per_table)
            ]
            self.hash_functions.append(table_hash_functions)
        for idx, datum in tqdm(
            enumerate(self.reduced_data),
            total=len(self.reduced_data),
            desc="Hashing data into tables",
        ):
            for t in range(self.num_tables):
                # Compute the hash key for the current data point in table t
                hash_key = tuple(hf(datum) for hf in self.hash_functions[t])
                # Store the index of the data point in the hash table under the computed hash key
                self.hash_tables[t][hash_key].append(idx)

    def query(self, query_feature):
        """
        Compute and return the hash codes for a given query feature across all hash tables.

        Parameters:
            query_feature (np.ndarray): The feature vector of the query image (1D array, length: any feature length).

        Returns:
            list of tuples: A list of hash codes, where each hash code is a tuple of bits for a specific table.
        """
        # Reduce the dimensionality of the query feature
        query_feature_reduced = self.pca.transform(query_feature.reshape(1, -1))

        # Compute the hash code for each table and return the list of hash codes
        hash_codes = []
        for t in range(self.num_tables):
            # Compute the hash key for the query feature in table t
            hash_key = tuple(hf(query_feature_reduced[0]) for hf in self.hash_functions[t])
            hash_codes.append(hash_key)

        return hash_codes
