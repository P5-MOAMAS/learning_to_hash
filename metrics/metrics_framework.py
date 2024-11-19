from collections.abc import Callable
from typing import List, Tuple

import numpy as np
import torch

class MetricsFramework:
    def __init__(self, query_func: Callable, dataset:  List[Tuple[int, List[int], int]]):
        self.query_func = query_func
        self.database = self.create_database(dataset)
        self.hit_miss = []


    """
    Creates a database, a list of tuples containing (id, hash-code, label) for each item in the dataset
    """
    def create_database(self, dataset: List[Tuple[int, List[int], int]]) -> List[Tuple[int, List[int], int]]:
        if len(dataset) == 0:
            RuntimeError("Dataset is empty")

        database = []
        for i, (idx, feature, label) in enumerate(dataset):
            code = self.query_func(feature)
            # Ensure code ís 1D and uses 0 instead of -1
            code = np.asarray(code).flatten()
            code[code == -1] = 0
            database.append((idx, code, label))

        return database

    """
    Returns a list of length total_predictions, containing tuples of (id, hash-code, hamming distance, if labels are equal)
    """
    def find_nearest_neighbours(self, query: (int, List[int], int), total_predictions: int) -> List[Tuple[int, List[int], int, bool]]:
        query_id, query_code, query_label = query
        query_code = np.array(query_code)

        database_codes = np.array([code for _, code, _ in self.database])
        database_labels = np.array([label for _, _, label in self.database])

        # Calculate the Hamming distance for all entries in the database
        distances = np.sum(database_codes != query_code, axis=1)

        # Compute the relevance
        is_relevant = (database_labels == query_label)

        # Find the indices of the top N nearest neighbors based on their distance to the query
        top_indices = np.argsort(distances)[:total_predictions]

        # Create the structure list for the top predictions
        predictions = [
            (self.database[i][0], self.database[i][1], distances[i], is_relevant[i]) for i in top_indices
        ]

        return predictions


    """
    Returns the precision of a given query for a maximum of total_predictions
    """
    def precision(self, query: (int, List[int], int), total_predictions: int) -> float:
        neighbours = self.find_nearest_neighbours(query, total_predictions)
        correct_predictions = 0
        for (idx, code, distance, is_relevant) in neighbours:
            if is_relevant:
                correct_predictions += 1

        return correct_predictions / total_predictions

    """
    Returns the recall of a given query for a maximum of total_predictions
    """
    def recall(self, query: (int, List[int], int), total_predictions: int) -> float:
        neighbours = self.find_nearest_neighbours(query, total_predictions)
        correct_predictions = 0
        for (idx, code, distance, is_relevant) in neighbours:
            if is_relevant:
                correct_predictions += 1
        total_relevant = sum([1 for (_, _, label) in self.database if query[2] == label])
        return correct_predictions / total_relevant


    def calculate_precision_recall(self):
        pass


    def average_precision(self, query: (int, List[int], int), total_predictions: int) -> float:
        neighbours = self.find_nearest_neighbours(query, total_predictions)
        correct_predictions = 0
        average_precision = 0
        for k, (idx, code, distance, is_relevant) in enumerate(neighbours, start=1):
            if is_relevant:
                correct_predictions += 1
                average_precision += correct_predictions / k

        # Ensure we divide only be total amount of relevant documents in the top k
        total_relevant = sum([1 for (_, _, label) in self.database if query[2] == label])
        if total_relevant > total_predictions:
            total_relevant = total_predictions
        self.hit_miss.append(correct_predictions / total_relevant)
        return average_precision / total_relevant


    def calculate_map(self, queries: List[Tuple[int, List[int], int]], total_predictions: int) -> float:
        mean_ap= 0
        for i, query in enumerate(queries):
            mean_ap += self.average_precision(query, total_predictions)
            print("Calculating mAP for", i, "/", len(queries), end="\r", flush=True)

        return mean_ap / len(queries)

    def mean_average_precision_sh(
            self,
            query_code,
            database_code,
            query_labels,
            database_labels,
            device,
            topk=None,
    ):
        """
        Calculate mean average precision(map).

        Args:
            query_code (torch.Tensor): Query data hash code.
            database_code (torch.Tensor): Database data hash code.
            query_labels (torch.Tensor): Query data targets, one-hot
            database_labels (torch.Tensor): Database data targets, one-host
            device (torch.device): Using CPU or GPU.
            topk (int): Calculate top k data map.

        Returns:
            meanAP (float): Mean Average Precision.
        """
        num_query = query_labels.shape[0]
        mean_AP = 0.0

        for i in range(num_query):
            # Retrieve images from database
            retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

            # Calculate hamming distance
            hamming_dist = 0.5 * (
                    database_code.shape[1] - query_code[i, :] @ database_code.t()
            )

            # Arrange position according to hamming distance
            retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

            # Retrieval count
            retrieval_cnt = retrieval.sum().int().item()

            # Can not retrieve images
            if retrieval_cnt == 0:
                continue

            # Generate score for every position
            score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

            # Acquire index
            index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float().to(device)

            mean_AP += (score / index).mean()

        mean_AP = mean_AP / num_query
        return mean_AP


    def calculate_metrics(self, queries: List[Tuple[int, List[int], int]], total_predictions: int) -> None:
        query_db = self.create_database(queries)
        print("Mean Average Precision:", self.calculate_map(query_db, total_predictions))
        print("Hit miss: " + str(round(sum(self.hit_miss) / len(self.hit_miss) * 100, 2)) + "%")


        database_code = []
        database_labels = []
        for (idx, code, label) in self.database:
            label_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label_array[label] = 1
            database_code.append(code)
            database_labels.append(label_array)

        query_code = []
        query_labels = []
        for (idx, code, label) in query_db:
            label_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label_array[label] = 1
            query_code.append(code)
            query_labels.append(label_array)


        print("Mean Average Precision SH:", self.mean_average_precision_sh(
            torch.tensor(query_code),
            torch.tensor(database_code),
            torch.tensor(query_labels),
            torch.tensor(database_labels),
            "cuda",
            total_predictions
        ))


if __name__ == "__main__":
    def foo_query(query: List[int]) -> List[int]:
        return query
    d = [
        (0, [1, 1, 0, 0], 1),
        (1, [1, 0, 0, 0], 2),
        (2, [0, 0, 0, 0], 3),
        (3, [1, 1, 1, 0], 2),
        (4, [1, 1, 1, 1], 1),
        (5, [0, 1, 0, 1], 2),
    ]
    m = MetricsFramework(foo_query, d)
    n = m.find_nearest_neighbours((6, [1, 0, 1, 0], 2), 6)
    print("Nearest Neighbours:")
    for l in n:
        print(l)

    print("Precision:", m.precision((6, [1, 0, 1, 0], 2), 10))
    print("Recall:", m.recall((6, [1, 0, 1, 0], 2), 10))
    print("Average Precision:", m.average_precision((6, [1, 0, 1, 0], 2), 10))
    print("Mean Average Precision:", m.calculate_map(d, 10))

