from typing import Callable
from metrics.feature_loader import FeatureLoader
from metrics.gen_cifar_simmat import SimilarityMatrix
from metrics.hash_lookup import pre_gen_hash_codes

"""
Takes a function that generates an image hash code, given an image.
function: function from image to hash code.
dataset_name: name of the dataset to use.

Generate useful metrics for measuring how well the function approximates
a hash code for the given image.
"""
def calculate_metrics(function: Callable, dataset: list[tuple[int, list, int]], is_deep = True):
    
    """
    TODO: Brug hash_lookup og pre_gen_hash_codes til at skabe db'en. Signaturen er ændret så den passer bedre ind.
    Db'en generere en list per query, så brug den til fx, udregning af precision osv.
    DB kommer ikke ud sorteret!!! Så sorter query således 'tætteste' kommer først.
    """

    db = pre_gen_hash_codes(function, dataset) # Create database.

    sim_matrix = SimilarityMatrix.create_matrix(dataset) # Generate similarity matrix with this dataset

    # Calculate average precision for each query
    aps = []
    for index, tmp in enumerate(dataset):
        id, feature, label = tmp
        print("Calculating average precisions:", index + 1, "of", len(dataset), "       ", sep=" ", end="\r")

        digest = function(feature)
        query_result = db.get_images_with_distance_sorted(digest, 2) # Is it good with max distance = 2?
        ap = average_precision(sim_matrix, id, query_result)
        aps.append(ap)
    print()

    # Calculate mean average precision
    mAP = sum(aps) / len(aps)

    # Temp print of mAP
    print("mAP: ", mAP)
    pass


"""
https://builtin.com/articles/mean-average-precision
Used the above for reference
"""
def average_precision(sim_matrix: SimilarityMatrix, query_image_idx: int, image_ids: list[int]):
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


def calc_mean_average_precision(sim_matrix: SimilarityMatrix, image_query_results: dict[int, list[int]]):
    aps = [average_precision(sim_matrix, query, image_query_results[query]) for query in image_query_results]
    s = sum(aps)
    for i, ap in enumerate(aps):
        print("Average precision ("+ str(i) +"):" + str(ap))
    return s / len(image_query_results)


def calculate_precision_recall_curve(sim_matrix: SimilarityMatrix, query_image_idx: int, image_ids: list[int]):
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


if __name__ == '__main__':
    query_results = {}
    for x in [x for x in range(100)]:
        query_results[x] = [i for i in range(0, 50000)]
    
    fl = FeatureLoader("cifar-10")
    
    if fl.training == None:
        raise Exception("Training subset was null")

    avg = calc_mean_average_precision(SimilarityMatrix.create_matrix(fl.training), query_results)
    print(avg)
