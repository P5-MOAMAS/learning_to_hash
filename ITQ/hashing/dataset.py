import os
import zipfile
import torch
import numpy as np
from scipy.io import loadmat

def load_cifar10_deep(root):
    all_features = []

    # Initialize the labels for each class (0-9) with 5000 instances each
    class_labels = np.array([digit for digit in range(10) for _ in range(5000)])

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
        else:
            print(f"Warning: Expected a tensor but got {type(features)} from {batch_filename}")

    # Concatenate all features into a single tensor
    if not all_features:
        raise ValueError("No features collected.")

    features = torch.cat(all_features, dim=0).numpy()  # Shape should be (50000, 25088)

    # Create labels corresponding to the collected features
    all_labels = np.repeat(np.arange(10), 5000)  # 10 classes with 5000 samples each

    # Check the shapes
    assert features.shape == (50000, 25088), f"Expected shape (50000, 25088) but got {features.shape}"
    assert len(all_labels) == 50000, f"Expected 50000 labels but got {len(all_labels)}"

    # Indices for query, train, and database sets
    query_index, train_index, database_index = [], [], []

    for digit in range(10):
        digit_idx = np.flatnonzero(all_labels == digit)
        digit_idx = np.random.permutation(digit_idx)

        # Adjust these numbers according to the desired split
        query_index.extend(digit_idx[:100])  # 100 per class for the query
        database_index.extend(digit_idx[100:])  # 4900 per class for database

    # Extract query, train, and database data
    query_data = (features[query_index], all_labels[query_index])
    database_data = (features[database_index], all_labels[database_index])

    return query_data, database_data

def load_cifar10(root):
    all_features = []

    batch_filename = f'cifar-10-1-features'
    batch_path = os.path.join(root, batch_filename)

    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"File not found: {batch_path}")

    try:
        features = torch.load(batch_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load {batch_path}. Error: {str(e)}")

    if isinstance(features, torch.Tensor):
        all_features.append(features)
    else:
        print(f"Warning: Expected a tensor but got {type(features)} from {batch_filename}")


    if not all_features:
        raise ValueError("No features collected.")

    features = torch.cat(all_features, dim=0).numpy()  # Shape should be (50000, 25088)

    # Create labels corresponding to the collected features
    all_labels = np.repeat(np.arange(10), 1000)  # 10 classes with 5000 samples each

    # Indices for query, train, and database sets
    query_index, train_index, database_index = [], [], []

    for digit in range(10):
        digit_idx = np.flatnonzero(all_labels == digit)
        digit_idx = np.random.permutation(digit_idx)

        # Adjust these numbers according to the desired split
        query_index.extend(digit_idx[:100])  # 100 per class for the query
        database_index.extend(digit_idx[100:])  # 4900 per class for database

    # Extract query, train, and database data
    query_data = (features[query_index], all_labels[query_index])
    database_data = (features[database_index], all_labels[database_index])

    return query_data, database_data

def load_cifar10_gist(root):
    database_path = os.path.join(root, 'cifar10_gist512_train.mat')
    query_path = os.path.join(root, 'cifar10_gist512_test.mat')

    database_dict = loadmat(database_path, squeeze_me=True)
    query_dict = loadmat(query_path, squeeze_me=True)

    database_features, database_labels = database_dict['train_features'], database_dict['train_labels']
    query_features, query_labels = query_dict['test_features'], query_dict['test_labels']

    return database_features, database_labels, query_features, query_labels

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
