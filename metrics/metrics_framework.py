from typing import Callable
from feature_loader import FeatureLoader
from gen_cifar_simmat import SimilarityMatrix
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
    """
    db = pre_gen_hash_codes(function, dataset)
    sim_matrix = SimilarityMatrix.create_matrix(dataset)

    for id, feature, label in dataset: # ???? Vel ikke på alting? Hvor meget data til det her?
        pass

def calculate_recall(sim_matrix: SimilarityMatrix, id: int, image_ids: list[int]):
    related = sim_matrix.get_related(id)
    count = sum([1 if x in related else 0 for x in image_ids])

    if len(related) == 0:
        return 0
    else:
        return count / len(related)


def calculate_precision(sim_matrix: SimilarityMatrix, id: int, image_ids: list[int]):
    related = sim_matrix.get_related(id)
    count = sum([1 if x in related else 0 for x in image_ids])
    
    if len(image_ids) == 0:
        return 0
    else:
        return count / len(image_ids)


"""
https://builtin.com/articles/mean-average-precision
Used the above for reference
"""
def average_precision_at_n(sim_matrix: SimilarityMatrix, query_image_idx: int, image_ids: list[int]):
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


def calc_mean_average_precision(sim_matrix: SimilarityMatrix, queries: list[int], image_query_results: dict[int, list[int]]):
    aps = [average_precision_at_n(sim_matrix, query, image_query_results[query]) for query in image_query_results]
    s = sum(aps)
    for i, ap in enumerate(aps):
        print("Average precision ("+ str(i) +"):" + str(ap))
    return s / len(queries)


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
    queries = [x for x in range(100)]
    query_results = {}
    for x in queries:
        query_results[x] = [i for i in range(0, 50000)]
    
    fl = FeatureLoader("cifar-10")
    
    if fl.training == None:
        raise Exception("Training subset was null")

    avg = calc_mean_average_precision(SimilarityMatrix.create_matrix(fl.training), queries, query_results)
    print(avg)
