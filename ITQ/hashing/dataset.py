import os
import torch
import numpy as np

def load_cifar10_deep(root):
    all_features = []
    all_labels = []
    root = os.path.normpath(root)

    class_labels = [digit for digit in range(10) for _ in range(500)]  # 500 per class for 10 classes

    for i in range(1, 6):
        batch_filename = f'cifar-10-{i}-features'
        batch_path = os.path.join(root, batch_filename)

        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"File not found: {batch_path}")

        try:
            features = torch.load(batch_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load {batch_path}. Error: {str(e)}")

        if isinstance(features, torch.Tensor):
            all_features.append(features)
            start_index = (i - 1) * 1000
            end_index = i * 1000
            all_labels.extend(class_labels[start_index:end_index])
        else:
            print(f"Warning: Expected a tensor but got {type(features)} from {batch_filename}")

    if not all_features or not all_labels:
        raise ValueError("No features or labels collected.")

    features = torch.cat(all_features, dim=0).numpy()
    labels = np.array(all_labels)

    query_index, train_index, database_index = [], [], []

    for digit in range(10):
        digit_idx = np.flatnonzero(labels == digit)
        digit_idx = np.random.permutation(digit_idx)
        query_index.extend(digit_idx[:100])
        train_index.extend(digit_idx[100:700])
        database_index.extend(digit_idx[100:])

    query_data = (features[query_index], labels[query_index])
    train_data = (features[train_index], labels[train_index])
    database_data = (features[database_index], labels[database_index])

    return query_data, train_data, database_data


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
