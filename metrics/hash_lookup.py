import pickle
import sys
from typing import Callable

import numpy as np


class Database:
    def __init__(self):
        # List of all images in buckets, the hashcode for a given bucket is the same index in the hashcode table
        self.database = []
        # List of hash-codes
        self.hashcodes = []


    def __getitem__(self, key) -> list | None:
        index = self.get_hash_code_index(key)
        return self.database[index] if index is not None else None


    def get_image_hash_code(self, id: int):
        for i in range(len(self.database)):
            for j in range(len(self.database[i])):
                if self.database[i][j] == id:
                    return self.hashcodes[i]
        return None

    def get_hash_code_index(self, key) -> int | None:
        for i, hashcode in enumerate(self.hashcodes):
            if np.array_equal(hashcode, key):
                return i
        return None

    def get_images_with_distance_sorted(self, query, max_distance):
        dist_to_indexes = [None] * (max_distance + 1)

        for i, code in enumerate(self.hashcodes):
            distance = self.hamming_distance(query, code)
            if distance <= max_distance:
                if distance in dist_to_indexes:
                    dist_to_indexes[distance].append(i)
                else:
                    dist_to_indexes[distance] = [i]

        sorted = []
        for xs in dist_to_indexes:
            if xs == None:
                continue
            sorted += xs

        return sorted


    def __setitem__(self, key, value) -> None:
        index = self.get_hash_code_index(key)
        if index is None:
            # Converts all -1 in hashcode to 0
            key[key == -1] = 0
            self.database.append([value])
            self.hashcodes.append(key)
        else:
            self.database[index].append(value)

    """
    Finds all neighbors in the Database within a given distance

    query                   - Hash-code to find neighbors for
    max_distance: int (5)   - The maximum distance to consider a neighbor

    returns: dict - A dictionary where keys are hash-codes and values are the distance
    """
    def get_neighbors(self, query, max_distance: int = 5) -> dict:
        neighbors = {}
        for i, code in enumerate(self.hashcodes):
            distance = self.hamming_distance(query, code)
            if distance <= max_distance:
                neighbors[i] = distance
        return neighbors


    def get_nearest_neighbors(self, query, amount: int = 10) -> list:
        neighbors = []
        # Get all neighbors
        neighbors_distance = self.get_neighbors(query, max_distance=sys.maxsize)

        # Sorts by distance first then index in the array
        distance_sorted = dict(sorted(neighbors_distance.items(), key=lambda item: (item[1], item[0])))

        for key in distance_sorted.keys():
            images = self.database[key]
            for image in images:
                if len(neighbors) < amount:
                    neighbors.append(image)
                else:
                    return neighbors
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

# Test function please delete
def generate_binary_hash(input_data, hash_size=4):
    hash_value = hash(input_data.tobytes())
    binary_hash = bin(hash_value & int('1' * hash_size, 2))[2:].zfill(hash_size)
    return np.array([int(bit) for bit in binary_hash], dtype=int)

if __name__ == "__main__":
    with open("cifar-10-batches-py/data_batch_1", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')[b'data'][:100]
    db = pre_gen_hash_codes(generate_binary_hash, data)
    neighbors = db.get_nearest_neighbors(generate_binary_hash(data[0]))
    print("Images ids of the", len(neighbors), "nearest neighbors:", neighbors)
