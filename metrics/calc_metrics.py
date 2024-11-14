from typing import Callable, List, Tuple

from metrics.hash_lookup import Database, pre_gen_hash_codes

def average_precision(database: Database, query_idx: int, dataset: List[Tuple[int, list, int]]) -> float:
    """
    Compute the Average Precision for a single query
    """
    # Compute Hamming distances for all other images in the dataset
    neighbors_distance = database.get_neighbors(database.get_image_hash_code(query_idx), 99999)

    # Sort by Hamming distance
    distance_sorted = dict(sorted(neighbors_distance.items(), key=lambda item: (item[1], item[0])))

    query_label = dataset[query_idx][2]

    relevant_count = 0
    total_relevant = sum([1 for (_, _, label) in dataset if label == query_label])

    # Calculate precision at each rank (rank is essentially based on the hamming distance)
    ap = 0
    for rank, key in enumerate(distance_sorted.keys(), start=1):
        if query_label == dataset[key][2]:
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            ap += precision_at_rank


    # Normalize AP by the number of relevant items
    if total_relevant > 0:
        ap = ap / total_relevant

    return ap

def calculate_metrics(function: Callable, dataset: List[Tuple[int, list, int]], is_deep=True) -> float:
    """
    Calculate the mAP a dataset using a model query function
    """

    db = pre_gen_hash_codes(function, dataset)
    num_queries = len(dataset)

    # Calculate average precision for each query
    total_ap = 0.0
    for query_idx in range(num_queries):
        total_ap += average_precision(db, query_idx, dataset)

        progress = round((query_idx)/num_queries * 15)
        print(f"Calculating mAP: |{("█" * progress)}{("-" * (15 - progress))}| {query_idx + 1}/{num_queries}", end='\r', flush=True)
    print()

    # Calculate mAP
    mAP = total_ap / num_queries
    print("Calculated mAP: ", mAP)
    return mAP
