import numpy as np


def load_mnist_gist(root):
    """load 512D GIST descriptor of MNIST and split it.

        query: 100 per class                  (total: 1000)
        train: 600 per class                  (total: 6000)
        database: 6900 (in average) per class (total: 69000)

    # Parameters:
        root: str, directory of MNIST GIST descriptor (.npz).

    # Returns:
        query_data: tuple[array], (features, labels).
        train_data: tuple[array], (features, labels).
        database_data: tuple[array], (features, labels).
    """
    mnist_gist = np.load(root)
    features, labels = mnist_gist['features'], mnist_gist['labels']
    assert features.shape == (70000, 512)
    assert labels.shape == (70000,)
    query_index, train_index, database_index = [], [], []
    for digit in range(10):
        digit_idx = np.flatnonzero(labels == digit)
        digit_idx = np.random.permutation(digit_idx)
        query_index.extend(digit_idx[:100])
        train_index.extend(digit_idx[100:700])
        database_index.extend(digit_idx[100:])
    query_data = features[query_index], labels[query_index]
    train_data = features[train_index], labels[train_index]
    database_data = features[database_index], labels[database_index]

    assert query_data[0].shape[0] == 1000
    assert train_data[0].shape[0] == 6000
    assert database_data[0].shape[0] == 69000

    return query_data, train_data, database_data
