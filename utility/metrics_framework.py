import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict

import numpy as np
import torch
from torchvision.transforms import Compose
from tqdm import tqdm, trange

from utility.data_loader import Dataloader
from utility.feature_loader import FeatureLoader


class Database:
    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels


class MetricsFramework:
    def __init__(self, query_func: Callable, data: Dataloader | FeatureLoader, query_size: int, trans: Compose = None,
                 multi_encoder: bool = False):
        self.query_func = query_func

        self.transform = trans
        self.database = self.create_database(data, query_size, False, multi_encoder)
        self.queries = self.create_database(data, query_size, True, multi_encoder)

    """
    Creates a database in format of the class Database, using the data given.
    Supports query functions that can encode multiple at once.
    """

    def create_database(self, data: Dataloader | FeatureLoader, query_size: int, is_queries: bool,
                        use_multi: bool = False) -> Database:
        if not data:
            raise RuntimeError("Dataset is empty.")
        if len(data.data) != len(data.labels):
            raise ValueError(f"Mismatch between dataset size ", len(data.data), "and label size", len(data.labels))

        # Hashcode generation
        if use_multi:
            codes = self.encode_multi(data, query_size, is_queries)
        else:
            codes = self.encode_single(data, query_size, is_queries)

        # Ensure the generated codes are using {0, 1} and isn't a 2d array
        for i in trange(len(codes)):
            # Update the code and handle any -1 values
            code = np.asarray(codes[i]).flatten()
            code[code == -1] = 0
            codes[i] = code

        # Label generation
        # Split the data into the correct chunk
        labels = data.labels[:query_size] if is_queries else data.labels[query_size:]

        # Create one hot label if label isn't already one-hot (Nuswide is already one hot)
        if isinstance(data.labels[0], np.ndarray) | isinstance(data.labels[0], list):
            labels_one_hot = np.asarray(labels)
        else:
            labels_one_hot = self.create_one_hot_labels(labels)

        return Database(np.asarray(codes), labels_one_hot)

    """
    Creates one-hot version of the given labels. This is specifically done for Cifar-10 and Mnist.
    """

    @staticmethod
    def create_one_hot_labels(labels):
        labels_one_hot = np.zeros((len(labels), 10))
        for i in trange(len(labels), desc="Creating one-hot encoded labels"):
            # Set the corresponding label in the one-hot encoded vector
            labels_one_hot[i, labels[i]] = 1
        return labels_one_hot

    """
    Uses the query function to encode images into hash-codes in batches of 1500
    """

    def encode_multi(self, data: Dataloader | FeatureLoader, query_size: int, is_queries: bool):
        batch_size = 1500
        codes = []

        # Process images in batches of 1500 as to not run out of memory
        print("Multi encoding is used, processing images in batches of", batch_size)

        last_index, first_index = (query_size, 0) if is_queries else (len(data), query_size)
        data_range = range(first_index, last_index, batch_size)

        for i in tqdm(data_range, desc="Encoding images"):
            # Ensure the current batch is within the expected range
            batch_last_index = min(i + batch_size, last_index)

            # Create the batch and transform it if a transform is given
            batch = [data[index] for index in range(i, batch_last_index)]
            if self.transform is not None:
                batch = [self.transform(feature) for feature in batch]
                batch = torch.stack(batch)

            batch_codes = self.query_func(batch)
            codes.extend(batch_codes)

        return codes

    """
    Uses the query function to encode images one at a time
    """

    def encode_single(self, data: Dataloader | FeatureLoader, query_size: int, is_queries: bool):
        codes = []
        data_range = range(0, query_size) if is_queries else range(query_size, len(data))
        for i in tqdm(data_range, desc="Encoding images"):
            feature = data[i]
            if self.transform is not None:
                feature = self.transform(feature)
            codes.append(self.query_func(feature))
        return codes

    """
    Finds all relevant items in the top k and ranks them by hamming distance
    """

    def get_relevant_in_top_k(self, query_index: int, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate ground truth relevance by checking if the dot product between
        # the query and retrieval labels is greater than zero (relevant items have a positive score)
        ground_truth_relevance = (
                np.dot(self.queries.labels[query_index, :], self.database.labels.transpose()) > 0).astype(
            np.float32)

        hamming_distances = self.calculate_hamming_distance(self.queries.codes[query_index, :], self.database.codes)

        # Sort the indices based on hamming distance in ascending order
        sorted_indices = np.argsort(hamming_distances)

        # Reorder the ground truth relevance according to the sorted Hamming distances
        sorted_ground_truth_relevance = ground_truth_relevance[sorted_indices]

        # Select the top-k relevant ground truth items for this query
        top_k_ground_truth_relevance = sorted_ground_truth_relevance[:top_k]

        return ground_truth_relevance, top_k_ground_truth_relevance

    def create_precision_recall_curve(self, max_top_k: int, min_k: int = 100, max_queries: int = 10000):
        if min_k < 1:
            raise ValueError("min_k must be greater than or equal to 1")
        if min_k > max_top_k:
            raise ValueError("min_k must be less than or equal to max_top_k")

        # The function used to calculate the precision recall curve for each query
        def compute_precision_recall(query_index):
            # Compute the relevant objects op to the max k value
            ground_truth_relevance, top_k_ground_truth_relevance = self.get_relevant_in_top_k(query_index, max_top_k)

            # Compute precision for all k values op to the max k value
            relevant_at_k = np.cumsum(top_k_ground_truth_relevance)
            computed_precision_values = relevant_at_k / np.arange(1, max_top_k + 1)

            # Compute recall for all k values op to the max k value
            total_relevant = np.sum(ground_truth_relevance)
            computed_recall_values = relevant_at_k / total_relevant

            # Return precision and recall in the range min k to top k
            return computed_precision_values[min_k - 1:max_top_k], computed_recall_values[min_k - 1:max_top_k]

        precision_values = []
        recall_values = []
        num_queries = self.queries.labels.shape[0]
        if num_queries > max_queries:
            num_queries = max_queries

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(compute_precision_recall, range(num_queries)), total=num_queries,
                                desc="Calculating precision-recall"))

        # Collect results from the threads
        for query_precision_values, query_recall_values in results:
            precision_values.append(query_precision_values)
            recall_values.append(query_recall_values)

        # Calculate the average over all queries for each k value
        avg_precision = np.mean(precision_values, axis=0)
        avg_recall = np.mean(recall_values, axis=0)

        return avg_precision.tolist(), avg_recall.tolist()

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

    def calculate_top_k_mean_average_precision(self, top_k: List[int]):
        num_queries = self.queries.codes.shape[0]
        average_precision = [0 for _ in range(len(top_k))]
        highest_k = max(top_k)
        for query_index in trange(num_queries, desc="Calculating mean average precision"):
            # Find the relevant items op to highest k value
            _, top_k_ground_truth_relevance = self.get_relevant_in_top_k(query_index, highest_k)

            # Calculate the average precision for each k value
            for k_index in range(len(top_k)):
                # Get the relevance of items op to the current k value
                top_k_ground_truth_relevance_at_k = top_k_ground_truth_relevance[:top_k[k_index]]
                total_relevant_in_top_k_at_k = np.sum(top_k_ground_truth_relevance[:top_k[k_index]]).astype(int)

                if total_relevant_in_top_k_at_k == 0:
                    continue

                # Create a linear sequence from 1 to the number of relevant items in top-k. Essentially: [1, 2, ..., total_relevant_in_top_k_at_k]
                relevance_count = np.linspace(1, total_relevant_in_top_k_at_k, total_relevant_in_top_k_at_k)

                # Get the indices of relevant items in the top-k
                relevant_item_indices = np.asarray(np.where(top_k_ground_truth_relevance_at_k == 1)) + 1.0

                if len(relevant_item_indices) == 0:
                    continue

                # Calculate the Average Precision for this query by averaging the relevance weighted by rank
                average_precision[k_index] += np.mean(relevance_count / relevant_item_indices)

        # Calculate the mean average precision at each k and return them as an array
        mean_average_precision = [average_precision[i] / num_queries for i in range(len(top_k))]

        return mean_average_precision

    def calculate_metrics(self, name: str, top_k: List[int]) -> dict[
        int | str, float | np.floating | np.complexfloating]:
        metrics = {}

        # Calculate the mAP at each k value
        map_at_ks = self.calculate_top_k_mean_average_precision(top_k)
        for k in range(len(top_k)):
            metrics[top_k[k]] = map_at_ks[k]
            print("Mean Average Precision at " + str(top_k[k]) + ":", metrics[top_k[k]])

        # Calculate the precision recall curve
        avg_precision, avg_recall = self.create_precision_recall_curve(len(self.database.codes), max_queries=2500)
        metrics["precision"] = avg_precision
        metrics["recall"] = avg_recall

        # Save the results to a json file for future use
        save_results(metrics, name + "_metrics")

        return metrics


def save_results(data: Dict, name: str):
    os.makedirs("results", exist_ok=True)
    results_file = "results/" + name + ".json"
    print("Saving results to " + results_file)
    with open(results_file, 'w') as f:
        json.dump(data, f)


def calculate_encoding_time(encode_length: int, data: FeatureLoader | Dataloader, trans: Compose = None,
                            query_gpu: Callable = None,
                            query_cpu: Callable = None):
    encoding_times_gpu = []
    processing_times = []
    encoding_times_cpu = []

    # Warmup so the results are more consistent
    for i in range(100):
        picture = data.data[1000 + i]
        if trans is not None:
            picture = trans(picture)
        if query_gpu is not None:
            query_gpu(picture)
        if query_cpu is not None:
            query_cpu(picture)

    if trans is not None:
        pictures = []
        for i in trange(1000, desc="Preprocessing images"):
            picture = data.data[i]
            processing_time_start = time.time_ns()
            picture = trans(picture)
            processing_times.append(time.time_ns() - processing_time_start)
            pictures.append(picture)
    else:
        pictures = data.data

    if query_gpu is not None:
        for i in trange(1000, desc="Calculating gpu times"):
            picture = pictures[i]
            encoding_gpu_time_start = time.time_ns()
            query_gpu(picture)
            encoding_times_gpu.append(time.time_ns() - encoding_gpu_time_start)

    if query_cpu is not None:
        for i in trange(1000, desc="Calculating cpu times"):
            picture = pictures[i]
            encoding_cpu_time_start = time.time_ns()
            query_cpu(picture)
            encoding_times_cpu.append(time.time_ns() - encoding_cpu_time_start)

    gpu_mean = np.mean(encoding_times_gpu) * math.pow(10, -6) if query_gpu is not None else 0
    cpu_mean = np.mean(encoding_times_cpu) * math.pow(10, -6) if query_cpu is not None else 0
    processing_mean = np.mean(processing_times) * math.pow(10, -6) if trans is not None else 0

    print(
        f"Encoding time at {encode_length} bits\n",
        f"\tGPU: {gpu_mean}ms\n" if query_gpu is not None else "",
        f"\tCPU: {cpu_mean}ms\n" if query_cpu is not None else "",
        f"\tProcessing time: {processing_mean}ms\n"
        f"Total time GPU: {gpu_mean + processing_mean}ms, CPU: {cpu_mean + processing_mean}\n" if trans is not None else "",
    )

    return gpu_mean, cpu_mean, processing_mean
