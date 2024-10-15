import torch
import numpy as np

from sklearn.decomposition import PCA
from utils.evaluate import mean_average_precision, pr_curve


def train(
    train_data,
    query_data,
    query_targets,
    retrieval_data,
    retrieval_targets,
    bin_hashcode_length,
    max_iter,
    device,
    topk,
    quantization_errors=None):
    """
    Training model.

    Args
        train_data(torch.Tensor): Training data.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        max_iter(int): Number of iterations.
        device(torch.device): GPU or CPU.
        topk(int): Calculate top k data points map.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Initialization
    query_data, query_targets, retrieval_data, retrieval_targets = query_data.to(device), query_targets.to(device), retrieval_data.to(device), retrieval_targets.to(device)

    """
    R=Rotation Matrix is randomly rotated, SVD (Singular Value Decomposition) is used to make sure
    that when the data is rotated the distances are not distorted. SVD splits the Matrix into three additional matrices
    "U" columns are the left singular of R (used for the initial rotation)
    "_" Singular values of R (not used)
    "_" Right Singular values of R (Not used)
    """

    R = torch.randn(bin_hashcode_length, bin_hashcode_length).to(device)
    [U, _, _] = torch.svd(R)
    R = U[:, :bin_hashcode_length]

    """
    PCA (Principal Component Analysis) is used to reduce the dimensions of the training data to the Binary hash length.
    PCA is an imported package. 
    V is the reduced dimension of the training data.
    """
    pca = PCA(n_components=bin_hashcode_length)
    V = torch.from_numpy(pca.fit_transform(train_data.numpy())).to(device)

    """
    ITQ Process.
    V is rotated using R to balance variance
    B is the rotated data binarized -1 or 1 
    The Matrix is then updated with SVD Where the right and left singular vectors are needed. 
    R is updated with the W(Left singular) and transpose of U (Right Singular) to reduce quantization error.
    
    """
    for i in range(max_iter):
        V_tilde = V @ R
        B = V_tilde.sign()
        error = torch.norm(B - V @ R, p='fro') ** 2
        quantization_errors.append(error.item())

        [U, _, W] = torch.svd(B.t() @ V)
        R = (W @ U.t())

    """
    After training, we generate the binary codes for the query data and retrieval data using the learned rotation matrix R and PCA transformation.
    The generate_code function applies PCA to the data, rotates it using R, and binarizes it.
    """
    query_code = generate_code(query_data.cpu(), bin_hashcode_length, R, pca)
    retrieval_code = generate_code(retrieval_data.cpu(), bin_hashcode_length, R, pca)

    # Compute map
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
        topk,
    )

    # P-R curve
    P, Recall = pr_curve(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        device,
    )

    # Save checkpoint
    checkpoint = {
        'qB': query_code,
        'rB': retrieval_code,
        'qL': query_targets,
        'rL': retrieval_targets,
        'pca': pca,
        'rotation_matrix': R,
        'P': P,
        'R': Recall,
        'map': mAP,
    }

    return checkpoint


def generate_code(data, code_length, R, pca):
    """
    Generate hashing code.

    Args
        data(torch.Tensor): Data.
        code_length(int): Hashing code length.
        R(torch.Tensor): Rotation matrix.
        pca(callable): PCA function.

    Returns
        pca_data(torch.Tensor): PCA data.
    """
    return (torch.from_numpy(pca.transform(data.numpy())).to(R.device) @ R).sign()

