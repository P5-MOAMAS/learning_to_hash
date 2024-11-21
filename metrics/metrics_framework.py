from collections.abc import Callable
from typing import List, Tuple

import numpy as np


class MetricsFramework:
    def __init__(self, query_func: Callable, dataset:  List[Tuple[int, List[int], int]], query_size: int):
        self.query_func = query_func

        self.hit_miss = []
        self.relevant = 0

        self.database = self.create_database(dataset)
        self.queries = self.create_database(dataset[:query_size])


    """
    Creates a database, a list of tuples containing (id, hash-code, label) for each item in the dataset
    """
    def create_database(self, dataset: List[Tuple[int, List[int], int]]) -> List[Tuple[int, List[int], int]]:
        if len(dataset) == 0:
            RuntimeError("Dataset is empty")

        database = []
        for i, (idx, feature, label) in enumerate(dataset, start=1):
            code = self.query_func(feature)
            # Ensure code ís 1D and uses 0 instead of -1
            code = np.asarray(code).flatten()
            code[code == -1] = 0
            database.append((idx, code, label))
            print("Creating database, image", i, "of", len(dataset), end="\r", flush=True)
        print()
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
        self.relevant = total_relevant
        self.hit_miss.append(correct_predictions / total_relevant)
        return average_precision / total_relevant


    def calculate_map(self, total_predictions: int) -> float:
        mean_ap= 0
        for i, query in enumerate(self.queries):
            mean_ap += self.average_precision(query, total_predictions)
            print("Calculating mAP for", i, "/", len(self.queries), end="\r", flush=True)

        return mean_ap / len(self.queries)


    def calculate_metrics(self, total_predictions: int) -> float:
        mAP = self.calculate_map(total_predictions)
        print("Mean Average Precision:", mAP)
        correct_ratio = sum(self.hit_miss) / len(self.hit_miss)
        print("Avg. correct img: " + str(round(correct_ratio * 100, 2)) + "% " + str(round(correct_ratio * self.relevant, 3)) + "/" + str(self.relevant))

        return mAP


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

