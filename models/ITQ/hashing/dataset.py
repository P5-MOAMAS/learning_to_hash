import os
import torch
import numpy as np
from scipy.io import loadmat

def load_imagenet_deep(root):
    """
    Load ImageNet with deep features extracted using a pre-trained model (e.g., VGG16).

    Parameters:
        root: str, the directory where ImageNet deep feature files are stored.

    Returns:
        query_data: tuple[array], (features, labels)
        database_data: tuple[array], (features, labels)
    """
    all_features = []

    for i in range(1, 55):
        batch_filename = f'image-net-{i}-features'
        batch_path = os.path.join(root, batch_filename)

        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"File not found: {batch_path}")

        try:
            # Load features
            features = torch.load(batch_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load {batch_path}. Error: {str(e)}")

            # Check if features are in tensor format, if yes append to all_features
        if isinstance(features, torch.Tensor):
            all_features.append(features)
        else:
            print(f"Warning: Expected a tensor but got {type(features)} from {batch_filename}")

            # Check if not all_features have been collected
        if not all_features:
            raise ValueError("No features collected.")

        features = torch.cat(all_features, dim=0).numpy()

        num_samples = features.shape[0]  # Number of samples in `features`
        num_classes = 1000

        # Adjusted label array to match `features`
        all_labels = np.repeat(np.arange(num_classes), num_samples // num_classes)

        # TODO: check the shapes
        assert features.shape == (10000, 512), f"Expected shape (50000, 512) but got {features.shape}"
        assert len(all_labels) == 10000, f"Expected 50000 labels but got {len(all_labels)}"

        # Indices for query and database sets
        query_index, database_index = [], []

        for digit in range(1000):
            digit_idx = np.flatnonzero(all_labels == digit)
            digit_idx = np.random.permutation(digit_idx)

            if len(digit_idx) < 1000:
                raise ValueError(f"Not enough samples for class {digit}. Got only {len(digit_idx)} samples.")

            # Split the data into query and database - Adjust these numbers according to the desired split
            query_index.extend(digit_idx[:1000])  # 100 per class for the query
            database_index.extend(digit_idx[1000:])  # remaining for database

        # Extract query and database data
        query_data = (features[query_index], all_labels[query_index])
        database_data = (features[database_index], all_labels[database_index])

        return query_data, database_data

def load_mnist_deep(root):
    all_features = []

    # Load features from each batch (1-5)
    for i in range(1, 7):
        batch_filename = f'cifar-10-{i}-features'
        batch_path = os.path.join(root, batch_filename)

        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"File not found: {batch_path}")

        try:
            # Load features
            features = torch.load(batch_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load {batch_path}. Error: {str(e)}")

        # Check if features are in tensor format, if yes append to all_features
        if isinstance(features, torch.Tensor):
            all_features.append(features)
        else:
            print(f"Warning: Expected a tensor but got {type(features)} from {batch_filename}")

    # Check if not all_features have been collected
    if not all_features:
        raise ValueError("No features collected.")

    features = torch.cat(all_features, dim=0).numpy()  # Shape should be (50000, 25088)

    # Create labels corresponding to the collected features
    all_labels = np.repeat(np.arange(10), 5000)  # 10 classes with 5000 samples each

    # Check the shapes
    assert features.shape == (50000, 512), f"Expected shape (50000, 512) but got {features.shape}"
    assert len(all_labels) == 50000, f"Expected 50000 labels but got {len(all_labels)}"

    # Indices for query and database sets
    query_index, database_index = [], []

    for digit in range(10):
        digit_idx = np.flatnonzero(all_labels == digit)
        digit_idx = np.random.permutation(digit_idx)

        # Split the data into query and database - Adjust these numbers according to the desired split
        query_index.extend(digit_idx[:100])  # 100 per class for the query
        database_index.extend(digit_idx[100:])  # remaining for database

    # Extract query and database data
    query_data = (features[query_index], all_labels[query_index])
    database_data = (features[database_index], all_labels[database_index])

    return query_data, database_data

def load_cifar10_deep(root):
    """
       Load CIFAR-10 with deep features extracted using a VGG16 model

       Parameters:
           root: str, the directory where CIFAR-10 deep feature files are stored.

       Returns:
           query_data: tuple[array], (features, labels)
           database_data: tuple[array], (features, labels)
       """
    all_features = []

    # Load features from each batch (1-5)
    for i in range(1, 6):
        batch_filename = f'cifar-10-{i}-features'
        batch_path = os.path.join(root, batch_filename)

        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"File not found: {batch_path}")

        try:
            # Load features
            features = torch.load(batch_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load {batch_path}. Error: {str(e)}")

        # Check if features are in tensor format, if yes append to all_features
        if isinstance(features, torch.Tensor):
            all_features.append(features)
        else:
            print(f"Warning: Expected a tensor but got {type(features)} from {batch_filename}")

    # Check if not all_features have been collected
    if not all_features:
        raise ValueError("No features collected.")

    features = torch.cat(all_features, dim=0).numpy()  # Shape should be (50000, 25088)

    # Create labels corresponding to the collected features
    all_labels = np.repeat(np.arange(10), 5000)  # 10 classes with 5000 samples each

    # Check the shapes
    assert features.shape == (50000, 512), f"Expected shape (50000, 512) but got {features.shape}"
    assert len(all_labels) == 50000, f"Expected 50000 labels but got {len(all_labels)}"

    # Indices for query and database sets
    query_index, database_index = [], []

    for digit in range(10):
        digit_idx = np.flatnonzero(all_labels == digit)
        digit_idx = np.random.permutation(digit_idx)

        # Split the data into query and database - Adjust these numbers according to the desired split
        query_index.extend(digit_idx[:100])  # 100 per class for the query
        database_index.extend(digit_idx[100:])  # remaining for database

    # Extract query and database data
    query_data = (features[query_index], all_labels[query_index])
    database_data = (features[database_index], all_labels[database_index])

    print("Sample of query data labels (first 10 labels):", query_data[1][:10])
    print("Counts of labels in query data:", {digit: np.sum(query_data[1] == digit) for digit in range(10)})

    return query_data, database_data

def load_cifar10(root):
    """
       Load CIFAR-10 deep features from a single batch.

       Parameters:
           root: str, the directory where CIFAR-10 features are stored.

       Returns:
           query_data: tuple[array], (features, labels)
           database_data: tuple[array], (features, labels)
       """

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

    features = torch.cat(all_features, dim=0).numpy()  # Shape should be (10000, 25088)

    # Create labels corresponding to the collected features
    all_labels = np.repeat(np.arange(10), 1000)  # 10 classes with 1000

    # Indices for query and database sets
    query_index, database_index = [], []

    for digit in range(10):
        digit_idx = np.flatnonzero(all_labels == digit)
        digit_idx = np.random.permutation(digit_idx)

        # Adjust these numbers according to the desired split
        query_index.extend(digit_idx[:100])  # 100 per class for the query
        database_index.extend(digit_idx[100:])  # Remaining for database

    # Extract query and database data
    query_data = (features[query_index], all_labels[query_index])
    database_data = (features[database_index], all_labels[database_index])

    return query_data, database_data

def load_cifar10_gist(root):
    """
    Load CIFAR-10 GIST dataset

    Parameters:
        root: str, directory where the CIFAR-10 GIST feature files are stored.

    Returns:
        database_features: array, database features.
        database_labels: array, database labels.
        query_features: array, query features.
        query_labels: array, query labels.
    """

    database_path = os.path.join(root, 'cifar10_gist512_train.mat')
    query_path = os.path.join(root, 'cifar10_gist512_test.mat')

    # Load .mat files containing GIST features
    database_dict = loadmat(database_path, squeeze_me=True)
    query_dict = loadmat(query_path, squeeze_me=True)

    # Retrieve feature and label data
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
        query_index.extend(digit_idx[:100]) # 100 per class for query
        train_index.extend(digit_idx[100:700]) # 600 per class for training
        database_index.extend(digit_idx[100:]) # Rest for database
    query_data = features[query_index], labels[query_index]
    train_data = features[train_index], labels[train_index]
    database_data = features[database_index], labels[database_index]

    assert query_data[0].shape[0] == 1000
    assert train_data[0].shape[0] == 6000
    assert database_data[0].shape[0] == 69000

    return query_data, train_data, database_data
