from collections.abc import Callable
from typing import List

import numpy as np
from torchvision.transforms import Compose
from tqdm import tqdm, trange


class Database:
    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels


class MetricsFramework:
    def __init__(self, query_func: Callable, data: List[List[int]], labels: List[int], query_size: int, trans: Compose = None, multi_encoder: bool = False):
        self.query_func = query_func

        self.transform = trans
        self.database = self.create_database(data[query_size:], labels[query_size:], multi_encoder)
        self.queries = self.create_database(data[:query_size], labels[:query_size], multi_encoder)


    """
    Creates a database in format of the class Database, using the data given.
    Supports query functions that can encode multiple at once.
    """
    def create_database(self, data: List[List[int]], labels: List[int], use_multi: bool = False) -> Database:
        if not data or not labels:
            raise RuntimeError("Dataset or labels are empty.")
        if len(data) != len(labels):
            raise ValueError(f"Mismatch between dataset size ", len(data), "and label size",len(labels))

        print("Creating database of", str(len(data)), "images...")
        if use_multi:
            codes = self.encode_multi(data)
        else:
            codes = self.encode_single(data)

        # Assuming labels range from 0 to 9, might have to altered for nus dataset
        labels_one_hot = np.zeros((len(labels), 10))
        for i in trange(len(labels), desc="Creating one-hot encoded labels and normalizing hash-codes"):
            # Update the code and handle any -1 values
            code = np.asarray(codes[i]).flatten()
            code[code == -1] = 0
            codes[i] = code

            # Set the corresponding label in the one-hot encoded vector
            labels_one_hot[i, labels[i]] = 1

        return Database(np.asarray(codes), labels_one_hot)

    """
    Uses the query function to encode images into hash-codes in batches of 10000
    """
    def encode_multi(self, data: List[List[int]]):
        batch_size = 10000
        codes = []

        # Process images in batches of 10000 as to not run out of memory
        print("Multi encoding is used, processing images in batches of", batch_size)
        for i in tqdm(range(0, len(data), batch_size), desc="Encoding images"):
            batch = data[i:i + batch_size]

            if self.transform is not None:
                batch = [self.transform(feature) for feature in batch]

            batch_codes = self.query_func(batch)
            codes.extend(batch_codes)

        return codes


    """
    Uses the query function to encode images one at a time
    """
    def encode_single(self, data: List[List[int]]):
        codes = []
        for feature in tqdm(data, desc="Encoding images"):
            if self.transform is not None:
                feature = self.transform(feature)
            codes.append(self.query_func(feature))
        return codes


    """
    Returns the precision of a given query for a maximum of total_predictions
    """
    def precision(self, query: (int, List[int], int), total_predictions: int) -> float:
        pass


    """
    Returns the recall of a given query for a maximum of total_predictions
    """
    def recall(self, query: (int, List[int], int), total_predictions: int) -> float:
        pass


    def calculate_precision_recall(self):
        pass


    """
    Calculates the hamming distance between 2 hash-codes
    """
    @staticmethod
    def calculate_hamming_distance(hash_code_1, hash_code_2):
        # Get the number of bits in the binary codes
        num_bits = hash_code_2.shape[1]

        # Calculate the Hamming distance using the dot product
        # The expression np.dot(binary_code_1, binary_code_2.transpose()) calculates the number of matching bits
        # Subtracting this from the total number of bits gives the number of differing bits (Hamming distance)
        hamming_distances = 0.5 * (num_bits - np.dot(hash_code_1, hash_code_2.transpose()))

        return hamming_distances


    """
    This method is based on the implementation used in hashnet
    """
    def calculate_top_k_mean_average_precision(self, top_k):
        num_queries = self.queries.labels.shape[0]
        total_mean_average_precision = 0

        print()
        for query_index in trange(num_queries, desc="Calculating mean average precision"):
            # Calculate ground truth relevance by checking if the dot product between
            # the query and retrieval labels is greater than zero (relevant items have a positive score)
            ground_truth_relevance = (np.dot(self.queries.labels[query_index, :], self.database.labels.transpose()) > 0).astype(np.float32)

            hamming_distances = self.calculate_hamming_distance(self.queries.codes[query_index, :], self.database.codes)

            # Sort the Hamming distances to rank retrieval items (ascending order: closer items first)
            sorted_indices = np.argsort(hamming_distances)

            # Reorder the ground truth relevance according to the sorted Hamming distances
            sorted_ground_truth_relevance = ground_truth_relevance[sorted_indices]

            # Select the top-k relevant ground truth items for this query
            top_k_ground_truth_relevance = sorted_ground_truth_relevance[:top_k]

            # Calculate the number of relevant items in the top-k
            total_relevant_in_top_k = np.sum(top_k_ground_truth_relevance).astype(int)

            # If no relevant items in the top-k, skip this query
            if total_relevant_in_top_k == 0:
                continue

            # Create a linear sequence from 1 to the number of relevant items in top-k (used for weighting)
            relevance_count = np.linspace(1, total_relevant_in_top_k, total_relevant_in_top_k)

            # Get the indices of relevant items in the top-k
            relevant_item_indices = np.asarray(np.where(top_k_ground_truth_relevance == 1)) + 1.0

            # Calculate the Average Precision for this query by averaging the relevance weighted by rank
            total_mean_average_precision += np.mean(relevance_count / relevant_item_indices)

        mean_average_precision = total_mean_average_precision / num_queries

        return mean_average_precision


    def calculate_metrics(self, total_predictions: int) -> float:
        mAP = self.calculate_top_k_mean_average_precision(total_predictions)
        print("Mean Average Precision:", mAP)
        return mAP