import random
from typing import Callable

from torch import nn
from feature_loader import FeatureLoader
from gen_cifar_simmat import CifarSimilarityMatrix
from label_loader import LabelLoader
from hash_lookup import pre_gen_hash_codes

import torch
import gc

"""
Takes a function that generates an image hash code, given an image.
function: function from image to hash code.
dataset_name: name of the dataset to use.

Generate useful metrics for measuring how well the function approximates
a hash code for the given image.
"""
def calculate_metrics(function: Callable, dataset_name, is_deep = True):
    pass

def calculate_recall(dataset_name: str, id: int, image_ids: list[int]):
    sim_matrix = None
    match dataset_name:
        case "cifar-10":
            try:
                sim_matrix = CifarSimilarityMatrix.load()
            except:
                CifarSimilarityMatrix.create_matrix().save()
                sim_matrix = CifarSimilarityMatrix.load()
        case _:
            raise LookupError("No similarity matrix can be found for: " + dataset_name)

    related = sim_matrix.get_related(id)
    count = sum([1 if x in related else 0 for x in image_ids])

    if len(related) == 0:
        return 0
    else:
        return count / len(related)


def calculate_precision(dataset_name: str, id: int, image_ids: list[int]):
    sim_matrix = None
    match dataset_name:
        case "cifar-10":
            try:
                sim_matrix = CifarSimilarityMatrix.load()
            except:
                CifarSimilarityMatrix.create_matrix().save()
                sim_matrix = CifarSimilarityMatrix.load()
        case _:
            raise LookupError("No similarity matrix can be found for: " + dataset_name)

    related = sim_matrix.get_related(id)
    count = sum([1 if x in related else 0 for x in image_ids])
    
    if len(image_ids) == 0:
        return 0
    else:
        return count / len(image_ids)

def get_similarity_matrix(dataset_name: str):
    sim_matrix = None
    match dataset_name:
        case "cifar-10":
            try:
                sim_matrix = CifarSimilarityMatrix.load()
            except:
                CifarSimilarityMatrix.create_matrix().save()
                sim_matrix = CifarSimilarityMatrix.load()
        case _:
            raise LookupError("No similarity matrix can be found for: " + dataset_name)
    return sim_matrix


"""
https://builtin.com/articles/mean-average-precision
Used the above for reference
"""
def average_precision_at_n(dataset_name: str, query_image_idx: int, image_ids: list[int]):
    sim_matrix = get_similarity_matrix(dataset_name)
    gtp_set = sim_matrix.get_related(query_image_idx) # Ground truth positives

    g_prime = []
    count = 0
    tp = 0

    for idx in image_ids:
        count += 1
        if idx in gtp_set:
            tp += 1
            g_prime.append(tp / count)

    if len(g_prime) > 0:
        return sum(g_prime) / len(gtp_set)
    else:
        return 0


def calc_mean_average_precision(dataset_name: str, queries: list[int], image_query_results: dict[int, list[int]]):
    aps = [average_precision_at_n(dataset_name, query, image_query_results[query]) for query in image_query_results]
    s = sum(aps)
    for i, ap in enumerate(aps):
        print("Average precision ("+ str(i) +"):" + str(ap))
    return s / len(queries)


def calculate_precision_recall_curve(dataset_name: str, query_image_idx: int, image_ids: list[int]):
    sim_matrix = get_similarity_matrix(dataset_name)

    points = []
    related = sim_matrix.get_related(query_image_idx)
    fp = 0
    tp = 0
    fn = len(related)
    for idx in image_ids:
        if idx in related:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        points.append((precision, recall))

    return points


"""
The class picks a random subset from the dataset specified

self.model: 
    The model being evaluated.

self.sample:
    The dataset the model is evaluated on.

self.switch_sample:
    Should switch to a new dataset. This includes setting self.sample and self.labels
    This functions should also use random to pick a random subset of the data.
    Maybe use del to make sure there is no memory living longer than needed.

self.query_model:
    Should call the model given the features
    This function should make sure the model is in the correct 'state',
    such as model.eval and pytorch.nograd
"""
class ModelEvaluator:
    def __init__(self, dataset_name: str, model: nn.Module):
        self.database = None
        self.model = model
        self.sample = None
        self.labels = None
        self.dataset_name = dataset_name
        self.switch_sample(dataset_name)

    """
    id is the image identity
    image_idxs is the list of all image identities retrieved
    This function calculates the recall, meaning of only related images
    retrieved what percentage do these make up of all related images.
    """
    def calculate_recall(self, id: int, image_ids: list[int]):
        return calculate_recall(self.dataset_name, id, image_ids)


    """
    id is the image identity
    image_idxs is the list of all image identities retrieved
    This function calculates the precision, meaning of the images
    retrieved what percentage of these are related.
    """
    def calculate_precision(self, id: int, image_ids: list[int]):
        return calculate_recall(self.dataset_name, id, image_ids)


    def calculate_metrics_for_id(self, id: int):
        self.prepare_dataset()
        neighbors = self.database.get_neighbors(id)
        ids = []
        for neighbor in neighbors:
            ids.extend(self.database.get_images(neighbor))

        print("Recall: ", calculate_recall(self.dataset_name, id, ids))
        print("Precision: ", calculate_precision(self.dataset_name, id, ids))


    """
    Pre-gens all hash codes for the current dataset
    """
    def prepare_dataset(self):
        feature_loader = FeatureLoader(self.dataset_name)
        for batch in feature_loader:
            self.database = pre_gen_hash_codes(self.query_model, batch, self.database)


    def query_model(self, features: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(features)
        return pred

    """
    Separate function in case sample should be selected across all batches
    """
    def get_sample(self, sample_size: int, batch_id: int = 1):
        data = FeatureLoader(self.dataset_name)[batch_id]
        labels = LabelLoader(self.dataset_name)[batch_id]

        selected_ids = random.sample(range(len(data)), sample_size)
        selected_data = [data[x] for x in selected_ids]
        selected_labels = [labels[x] for x in selected_ids]

        del data, labels
        gc.collect()

        return selected_data, selected_labels


    def switch_sample(self, dataset_name: str, sample_size: int = 1000):
        self.sample, self.labels = self.get_sample(sample_size, dataset_name)


if __name__ == '__main__':
    queries = [x for x in range(100)]
    query_results = {}
    for x in queries:
        query_results[x] = [i for i in range(0, 50000)]

    avg = calc_mean_average_precision("cifar-10", queries, query_results)
    print(avg)
