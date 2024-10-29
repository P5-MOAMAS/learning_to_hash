"""Iterative Quantization Hashing.

Gong, Y., & Lazebnik, S. (2011). Iterative quantization:
A procrustean approach to learning binary codes. CVPR 2011, 817–824.
https://doi.org/10.1109/CVPR.2011.5995432
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from model import Model

def sign(X):
    """Sign function.

    sign(x) = -1 if x < 0 and 1 if x >= 0.

    # Parameters:
        X: array.
    # Returns:
        X_sign: array, same shape as `X`.
    """
    X_sign = np.ones(X.shape).astype(int)
    X_sign[X < 0] = -1
    return X_sign

class _ITQ_Base(Model):
    """Base class object for ITQ."""
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
        # Dimension reduction
        # shape: (n_samples, encode_len)
        V = self._project(X)

        # Initialize with a orthogonal random rotation
        R = np.random.randn(self.encode_len, self.encode_len)
        R = np.linalg.svd(R)[0]

        # ITQ to find optimal rotation
        for _ in range(self.n_iter):
            # Fix R and updata B:
            # shape: (n_samples, encode_len)
            Z = np.matmul(V, R)
            B = sign(Z)
            # Fix B and update R:
            # shape: (encode_len, encode_len)
            C = np.matmul(B.T, V)
            S, _, S_hat_T = np.linalg.svd(C)
            # R = S_hat @ S.T = (S @ S_hat_T).T
            R = np.matmul(S, S_hat_T).T
        return R

    def encode(self, X):
        """Encode `X` to binary codes.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The data.
        # Returns:
            B: array, shape = (n_samples, encode_len).
                Binary codes of X.
        """
        # Dimensionality Reduction
        V = self._project(X)
        # Rotate the data
        Z = np.matmul(V, self._R)
        B = sign(Z)
        return B


class ITQ(_ITQ_Base):
    def __init__(self, encode_len):
        """Iterative Quantization Hashing (with PCA).

        # Parameters:
            encode_len: int.
                Encode length of binary codes.
        # Returns:
            None.
        """
        super().__init__(encode_len)

    def fit(self, X, n_iter=50):
        """Get the principal component and learn the rotation matrix.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            n_iter: int (default=50).
                Max number of iterations, 50 is usually enough.
        # Returns:
                None.
        """
        self.n_iter = n_iter
        self.pca = None
        # PCA
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.encode_len)
        self.pca.fit(X)
        self._project = self.pca.transform
        # ITQ to find the best rotation matrix R
        self._R = self._itq(X)


    def save_model(self, filename):
        """Save the model parameters to a file."""
        print('rotation_matrix')
        print(self._R)
        print('pca')
        print(self.pca)

        # Save using torch
        torch.save({
            'rotation_matrix': self._R,
            'encode_len': self.encode_len,
            'pca_components': self.pca.components_,  # PCA components
            'pca_mean': self.pca.mean_,  # PCA mean
            'pca_explained_variance': self.pca.explained_variance_,  # explained_variance_
        }, filename)

    def load_model(self, filename):
        """Load the model parameters from a file."""
        checkpoint = torch.load(filename)
        self._R = checkpoint['rotation_matrix']
        self.encode_len = checkpoint['encode_len']

        # Restore PCA model
        self.pca = PCA(n_components=self.encode_len)
        self.pca.components_ = checkpoint['pca_components']
        self.pca.mean_ = checkpoint['pca_mean']
        self.pca.explained_variance_ = checkpoint['pca_explained_variance']
        self._project = lambda X: self.pca.transform(X)
