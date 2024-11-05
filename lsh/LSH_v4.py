from sklearn.decomposition import IncrementalPCA
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class LSH:
    def __init__(self, data, num_tables=5, num_bits_per_table=10, pca_components=None):
        """
        Initialize the LSH (Locality-Sensitive Hashing) class and execute the LSH process.

        Parameters:
            data (np.ndarray): The dataset of features to index.
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
        self.pca = None

        # Execute the LSH process during initialization
        self.preprocess_data()
        self.hash_data()

    def preprocess_data(self):
        """
        Preprocess the data by performing incremental PCA for dimensionality reduction.
        """
        print("Preprocessing data...")
        batch_size = 256
        total_samples = self.data.shape[0]

        # Ensure the batch size is at least equal to the number of PCA components
        if batch_size < self.pca_components:
            batch_size = self.pca_components

        # Adjust total_samples to avoid a small last batch
        num_batches = total_samples // batch_size
        total_samples = num_batches * batch_size
        self.data = self.data[:total_samples]

        # Initialize Incremental PCA
        self.pca = IncrementalPCA(
            n_components=self.pca_components, batch_size=batch_size
        )

        # Fit the PCA incrementally on batches
        for i in tqdm(
            range(0, total_samples, batch_size),
            desc="Fitting IncrementalPCA",
        ):
            batch = self.data[i : i + batch_size]
            self.pca.partial_fit(batch)

        # Transform the data incrementally
        self.reduced_data = []
        for i in tqdm(
            range(0, total_samples, batch_size),
            desc="Transforming data with PCA",
        ):
            batch = self.data[i : i + batch_size]
            reduced_batch = self.pca.transform(batch)
            self.reduced_data.append(reduced_batch)
        self.reduced_data = np.vstack(self.reduced_data)

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

    def query(self, query_data):
        """
        Compute and return the hash codes for the given query data across all hash tables.

        Parameters:
            query_data (np.ndarray): The query data point.

        Returns:
            list of tuples: A list of hash codes, where each hash code is a tuple of bits for a specific table.
        """
        # Reduce the dimensionality of the query data
        query_data_reduced = self.pca.transform(query_data.reshape(1, -1))

        # Compute the hash code for each table and return the list of hash codes
        hash_codes = []
        for t in range(self.num_tables):
            # Compute the hash key for the query data in table t
            hash_key = tuple(hf(query_data_reduced[0]) for hf in self.hash_functions[t])
            hash_codes.append(hash_key)

        return hash_codes
