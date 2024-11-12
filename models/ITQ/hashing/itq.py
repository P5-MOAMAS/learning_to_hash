"""Iterative Quantization Hashing.

Gong, Y., & Lazebnik, S. (2011). Iterative quantization:
A procrustean approach to learning binary codes. CVPR 2011, 817–824.
https://doi.org/10.1109/CVPR.2011.5995432
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from model import Model  # Assuming this imports a base class for machine learning models

def sign(X):
    """Sign function.

    sign(x) = -1 if x < 0 and 1 if x >= 0. to ensure that it is on the axis

    # Parameters:
        X: array.
    # Returns:
        X_sign: array, same shape as `X`.
    """
    X_sign = np.ones(X.shape).astype(int)
    X_sign[X < 0] = -1
    return X_sign

class _ITQ_Base(Model):
    """Base class object for Iterative Quantization (ITQ)."""
    def __init__(self, encode_len):
        super().__init__(encode_len)

    def _itq(self, X):
        """Iterative quantization to find optimal rotation.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
        # Returns:
            R: array, shape = (encode_len, encode_len).
                Optimal rotation matrix.
        """
        # Perform dimension reduction to reduce data to desired encode length
        # shape: (n_samples, encode_len)
        V = self._project(X)

        # Initialize with a random orthogonal rotation matrix
        R = np.random.randn(self.encode_len, self.encode_len)
        R = np.linalg.svd(R)[0]  # Ensure orthogonality by using SVD

        # ITQ iterations to refine the rotation matrix
        for _ in range(self.n_iter):
            # Step 1: Fix R, compute binary code B based on V and R
            Z = np.matmul(V, R)
            B = sign(Z)  # Binary code 1 or -1

            # Step 2: Fix B, update R by solving for optimal rotation
            C = np.matmul(B.T, V)
            S, _, S_hat_T = np.linalg.svd(C)  # Singular value decomposition
            R = np.matmul(S, S_hat_T).T  # Update R with SVD results
        return R

    def encode(self, X):
        """Encode `X` to binary codes.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The data to encode.
        # Returns:
            B: array, shape = (n_samples, encode_len).
                Binary codes of X.
        """
        # Step 1: Perform dimensionality reduction
        V = self._project(X)
        # Step 2: Apply rotation to get binary code
        Z = np.matmul(V, self._R)
        B = sign(Z)
        return B


class ITQ(_ITQ_Base):
    def __init__(self, encode_len):
        """Iterative Quantization Hashing (with PCA for dimension reduction).

        # Parameters:
            encode_len: int.
                Encode length of binary codes.
        """
        super().__init__(encode_len)

    def fit(self, X, n_iter=3):
        """Compute principal components and learn rotation matrix.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            n_iter: int (default=50).
                Maximum number of iterations for ITQ.
        """
        self.n_iter = n_iter  # Set number of ITQ iterations
        self.pca = PCA(n_components=self.encode_len)  # Initialize PCA for dimensionality reduction
        self.pca.fit(X)  # Fit PCA to data
        self._project = self.pca.transform  # Set projection function to PCA transformation
        # Perform ITQ to find optimal rotation matrix
        self._R = self._itq(X)

    def save_model(self, filename):
        """Save the model parameters to a file."""
        # Print statements to show model parameters (for debug)
        print('rotation_matrix')
        print(self._R)
        print('pca')
        print(self.pca)

        # Save the model using PyTorch
        torch.save({
            'rotation_matrix': self._R,
            'encode_len': self.encode_len,
            'pca_components': self.pca.components_,  # PCA components matrix
            'pca_mean': self.pca.mean_,  # Mean of data used in PCA
            'pca_explained_variance': self.pca.explained_variance_,  # Explained variance by components
        }, filename)

    def load_model(self, filename):
        """Load model parameters from a file."""
        checkpoint = torch.load(filename)
        self._R = checkpoint['rotation_matrix']
        self.encode_len = checkpoint['encode_len']

        # Restore PCA model parameters
        self.pca = PCA(n_components=self.encode_len)
        self.pca.components_ = checkpoint['pca_components']
        self.pca.mean_ = checkpoint['pca_mean']
        self.pca.explained_variance_ = checkpoint['pca_explained_variance']
        self._project = lambda X: self.pca.transform(X)  # Restore PCA projection function
