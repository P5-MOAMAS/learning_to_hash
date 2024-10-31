from typing import Callable

import numpy as np

from util.progressbar.src.progressbar import progressbar


class Database:
    def __init__(self):
        # Key hashcode - value array of ids of picture that has that hash-code
        self.database = {}


    def __getitem__(self, key) -> list | None:
        return self.database[key] if key in self.database else None


    def __setitem__(self, key, value) -> None:
        if key in self.database:
            self.database[key].append(value)
        else:
            self.database[key] = [value]


    """
    Finds all neighbors in the Database within a given distance

    query                   - Hash-code to find neighbors for
    max_distance: int (5)   - The maximum distance to consider a neighbor

    returns: dict - A dictionary where keys are hash-codes and values are the distance
    """
    def get_neighbors(self, query, max_distance: int = 5) -> dict:
        neighbors = {}
        for key in self.database.keys():
            distance = self.hamming_distance(query, key)
            if distance <= max_distance:
                neighbors[key] = distance
        return neighbors


    """
    Calculates the distance between two hash-codes
    
    bin1         - Hash-code 1
    bin2         - Hash-code 2
    
    returns: int - The distance between two hash-codes
    """
    def hamming_distance(self, bin1, bin2) -> int:
        return np.count_nonzero(bin1 != bin2)


"""
Generates hash-codes for an entire dataset

model_query: Callable   - A function that creates a prediction from a model given a tensor
dataset: Numpy array    - The dataset to generate hash codes for
db: Database            - An optional data set that can be extended with new codes

returns - A Database containing all generated hash codes
"""
def pre_gen_hash_codes(model_query: Callable, dataset: np.ndarray, db: Database = None) -> Database:
    db = Database() if db is None else db

    for index in progressbar(range(len(dataset)), "Generating hash codes"):
        key = model_query(dataset[index])
        db[key] = index

    return db