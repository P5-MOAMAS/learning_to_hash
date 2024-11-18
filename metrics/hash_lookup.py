from typing import Callable

import numpy as np
import torch


class Database:
    def __init__(self):
        # List of hash-codes (id of image, hashcode)
        self.hash_codes = {}


    def __getitem__(self, key) -> list | None:
        return self.hash_codes[key]


    def __len__(self):
        return len(self.hash_codes)


    def get_hash_code_index(self, key) -> int | None:
        for i, hashcode in enumerate(self.hash_codes.values()):
            if np.array_equal(hashcode, key):
                return i
        return None

    def hash_codes_as_np_array(self):
        return np.array([code for code in self.hash_codes.values()])

    def hash_codes_as_tensor(self):
        return torch.tensor([code for code in self.hash_codes.values()])


    def __setitem__(self, key, value) -> None:
        self.hash_codes[value] = np.array(key).flatten().tolist()

    """
    Finds all neighbors in the Database within a given distance

    query                   - Hash-code to find neighbors for
    max_distance: int (5)   - The maximum distance to consider a neighbor

    returns: dict - A dictionary where keys are hash-codes and values are the distance
    """
    def get_neighbors(self, query, max_distance: int = 5) -> dict:
        neighbors = {}
        for i, item in enumerate(self.hash_codes.items()):
            distance = self.hamming_distance(query, item[1])
            if distance <= max_distance:
                neighbors[item[0]] = distance
        return neighbors

    """
    Calculates the distance between two hash-codes
    
    bin1         - Hash-code 1
    bin2         - Hash-code 2
    
    returns: int - The distance between two hash-codes
    """
    @staticmethod
    def hamming_distance(bin1, bin2) -> int:
        return np.count_nonzero(bin1 != bin2)


"""
Generates hash-codes for an entire dataset

model_query: Callable   - A function that creates a prediction from a model given a tensor
dataset: Numpy array    - The dataset to generate hash codes for
db: Database            - An optional data set that can be extended with new codes

returns - A Database containing all generated hash codes
"""
def pre_gen_hash_codes(model_query: Callable, dataset: list[tuple[int, list, int]], db: Database | None = None) -> Database:
    db = Database() if db is None else db

    for count, e in enumerate(dataset):
        index, feature, label = e
        print("Generating hash codes:", count + 1, "out of", len(dataset), "   ", sep=" ", end="\r") # Fuck progress bar
        key = model_query(feature)
        db[key] = count
    print()

    return db
