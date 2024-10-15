import torch
def load_feature(feature, root):
    """
    Load Features from Root.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
    """
    if feature == 'mnist' or feature == 'cifar10' or feature == 'imagenet':
        return _load_data(root)
    else:
        raise ValueError('Invalid Feature Name!')


def _load_data(root):
    """
    Load features extracted from root.

    Args
        root(str): Path of dataset.

    Returns
        train_data(torch.Tensor, 5000*4096): Training data.
        train_targets(torch.Tensor, 5000*10): One-hot training targets.
        query_data(torch.Tensor, 1000*4096): Query data.
        query_targets(torch.Tensor, 1000*10): One-hot query targets.
        retrieval_data(torch.Tensor, 59000*4096): Retrieval data.
        retrieval_targets(torch.Tensor, 59000*10): One-hot retrieval targets.
    """
    data = torch.load(root)
    train_data = data['train_features']
    train_targets = data['train_targets']
    query_data = data['query_features']
    query_targets = data['query_targets']
    retrieval_data = data['retrieval_features']
    retrieval_targets = data['retrieval_targets']

    # Normalization
    mean = retrieval_data.mean()
    std = retrieval_data.std()
    train_data = (train_data - mean) / std
    query_data = (query_data - mean) / std
    retrieval_data = (retrieval_data - mean) / std

    return train_data, train_targets, query_data, query_targets, retrieval_data, retrieval_targets

