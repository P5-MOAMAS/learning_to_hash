from collections.abc import Callable
from typing import List, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
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


    def get_relevant_in_top_k(self, query_index: int, top_k: int) -> Tuple[int, np.ndarray, np.ndarray]:
        # Calculate ground truth relevance by checking if the dot product between
        # the query and retrieval labels is greater than zero (relevant items have a positive score)
        ground_truth_relevance = (
                    np.dot(self.queries.labels[query_index, :], self.database.labels.transpose()) > 0).astype(
            np.float32)

        hamming_distances = self.calculate_hamming_distance(self.queries.codes[query_index, :], self.database.codes)

        # Sort the Hamming distances to rank retrieval items (ascending order: closer items first)
        sorted_indices = np.argsort(hamming_distances)

        # Reorder the ground truth relevance according to the sorted Hamming distances
        sorted_ground_truth_relevance = ground_truth_relevance[sorted_indices]

        # Select the top-k relevant ground truth items for this query
        top_k_ground_truth_relevance = sorted_ground_truth_relevance[:top_k]

        # Calculate the number of relevant items in the top-k
        total_relevant_in_top_k = np.sum(top_k_ground_truth_relevance).astype(int)

        return total_relevant_in_top_k, ground_truth_relevance, top_k_ground_truth_relevance


    """
    Returns the precision of a given query for a maximum of top_k
    """
    def precision(self, query_index: int, top_k: int) -> float:
        total_relevant_in_top_k, _, _ = self.get_relevant_in_top_k(query_index, top_k)
        return total_relevant_in_top_k / top_k


    """
    Returns the recall of a given query for a maximum of top_k
    """
    def recall(self, query_index: int, top_k: int) -> float:
        total_relevant_in_top_k, ground_truth_relevance, _ = self.get_relevant_in_top_k(query_index, top_k)
        return total_relevant_in_top_k / np.sum(ground_truth_relevance)


    def create_precision_recall_curve(self, max_top_k: int, min_k: int = 10):
        if min_k < 1:
            raise ValueError("min_k must be greater than or equal to 1")
        if min_k > max_top_k:
            raise ValueError("min_k must be less than or equal to max_top_k")

        precision_values = []
        recall_values = []
        num_queries = self.queries.labels.shape[0]

        for query_index in trange(num_queries, desc="Calculating precision-recall curve"):
            query_precision_values = []
            query_recall_values = []

            _, ground_truth_relevance, top_k_ground_truth_relevance = self.get_relevant_in_top_k(query_index, max_top_k)

            for top_k in range(min_k, max_top_k + 1):
                total_relevant_at_k = np.sum(top_k_ground_truth_relevance[:top_k]).astype(int)
                precision_value = total_relevant_at_k / top_k
                recall_value = total_relevant_at_k / np.sum(ground_truth_relevance)

                query_precision_values.append(precision_value)
                query_recall_values.append(recall_value)

            precision_values.append(query_precision_values)
            recall_values.append(query_recall_values)

        avg_precision = np.mean(precision_values, axis=0)
        avg_recall = np.mean(recall_values, axis=0)

        plt.plot(avg_recall, avg_precision, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve k: ' + str(min_k) + "-" + str(max_top_k) + " queries: " + str(num_queries))
        plt.grid(True)
        plt.savefig("precision_recall_curve.png")

        return avg_precision, avg_recall


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
    def calculate_top_k_mean_average_precision(self, top_k: int):
        num_queries = self.queries.labels.shape[0]
        total_mean_average_precision = 0

        print()
        for query_index in trange(num_queries, desc="Calculating mean average precision"):
            total_relevant_in_top_k, _, top_k_ground_truth_relevance = self.get_relevant_in_top_k(query_index, top_k)

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


    def calculate_metrics(self, top_k: int) -> float:
        if top_k > self.database.labels.shape[0]:
            raise RuntimeError("Top k(", top_k,")is larger than the database(", self.database.labels.shape[0],")")

        mAP = self.calculate_top_k_mean_average_precision(top_k)

        self.create_precision_recall_curve(top_k)

        print("Mean Average Precision:", mAP)
        return mAP