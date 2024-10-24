"""Evaluating the performance of model.

Evaluate metrics include:
    - mean average precision
    - precision-recall curve

Using {-1, 1} instead of {0, 1} as binary encoding.
"""
import numpy as np


def hamming_distance(query, database):
    """Compute the hamming distance between `query` and `database`.

    # Parameters:
        query: array, shape = (n_q, k), belongs to {-1, 1}.
        database: array, shape = (n, k), belongs to {-1, 1}.
    # Returns:
        hamming_dist: array (dtype: int), shape = (n_q, n)
    """
    if len(query.shape) == 1:
        query = np.expand_dims(query, axis=0)
    encode_len = query.shape[1]
    hamming_dist = 1 / 2 * (encode_len - np.matmul(query, database.T))
    return hamming_dist.astype(int)


def mean_average_precision(query, query_label,
                           database, database_label,
                           top_k=None):
    """Compute mAP(mean average precision).

    # Parameters:
        query: array, shape = (n_q, k), belongs to {-1, 1}.
        query_label: array, shape = (n_q,).
        database: array, shape = (n, k), belongs to {-1, 1}.
        database_label: array, shape = (n,).
        top_k: int (default=None).
    # Returns:
        mAp: float.
    """

    assert query.shape[0] == query_label.shape[0]
    assert database.shape[0] == database_label.shape[0]

    # shape: (n_q, n)
    hamming_dist = hamming_distance(query, database)
    returned_idx = np.argsort(hamming_dist, axis=1)
    # shape: (n_q, top_k)
    returned_label = database_label[returned_idx[:, :top_k]]
    is_match = query_label.reshape(-1, 1) == returned_label
    precisions = (np.cumsum(is_match, axis=1) /
                  np.arange(1, is_match.shape[1] + 1))
    total_sum = np.sum(is_match, axis=1)
    total_sum[total_sum == 0] = 1  # avoid division by 0
    average_precision = np.sum(is_match * precisions, axis=1) / total_sum
    return np.mean(average_precision)


def precision_recall(query, query_label,
                     database, database_label):
    """Computer precision-recall pairs under different hamming radii.

        precision = #good / #returned
        recall    = #good / #ground-truth
        good = returned & ground_truth

    # Parameters:
        query: array, shape = (n_q, k), belongs to {-1, 1}.
        query_label: array, shape = (n_q,).
        database: array, shape = (n, k), belongs to {-1, 1}.
        database_label: array, shape = (n,).
    # Returns:
        precision: array, shape = (max(hamming_dist),)
        recall: array, shape = (max(hamming_dist),)
    """

    assert query.shape[0] == query_label.shape[0]
    assert database.shape[0] == database_label.shape[0]

    # shape: (n_q, n)
    hamming_dist = hamming_distance(query, database)
    precision, recall = [], []
    max_radius = np.max(hamming_dist)
    for radius in range(max_radius + 1):

        # shape: (n_q, n)
        returned = hamming_dist <= radius
        # shape: (n_q,)
        n_returned = np.sum(returned, axis=1)
        n_returned[n_returned == 0] = 1  # avoid division by 0

        # shape: (n_q, n)
        ground_truth = query_label.reshape(-1, 1) == database_label
        # shape: (n_q,)
        n_ground_truth = np.sum(ground_truth, axis=1)

        # shape: (n_q,)
        n_good = np.sum(returned & ground_truth, axis=1)
        precision.append(np.mean(n_good / n_returned))
        recall.append(np.mean(n_good / n_ground_truth))
    return np.array(precision), np.array(recall)
