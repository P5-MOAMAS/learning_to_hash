"""Iterative Quantization Hashing.

Gong, Y., & Lazebnik, S. (2011). Iterative quantization:
A procrustean approach to learning binary codes. CVPR 2011, 817–824.
https://doi.org/10.1109/CVPR.2011.5995432
"""
import numpy as np

from model import Model
from utils import sign


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
        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.encode_len)
        pca.fit(X)
        self._project = pca.transform
        # ITQ to find the best rotation matrix R
        self._R = self._itq(X)


class ITQ_CCA(_ITQ_Base):
    def __init__(self, encode_len):
        """Iterative Quantization Hashing (with CCA).

        # Parameters:
            encode_len: int.
                Encode length of binary codes.
        # Returns:
            None.
        # Examples:
            Given X (shape=[n_samples, n_features]) and y (shape=[n_samples,]):
            >>> from hashing.model import ITQ_CCA
            >>> from hashing.utils import one_hot_encoding
            >>> itq_cca = ITQ_CCA(encode_len=32)
            >>> y = one_hot_encoding(y, n_classes=10)
            >>> itq_cca.fit(X, y)
        """
        super().__init__(encode_len)

    def fit(self, X, y, n_iter=50):
        """Get the projection matrix and learn the rotation matrix.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            y: array, shape = (n_samples, n_targets).
                The multi-class labels of the data.
            n_iter: int (default=50).
                Max number of iterations, 50 is usually enough.
        # Returns:
                None.
        """
        self.n_iter = n_iter
        # CCA
        from sklearn.cross_decomposition import CCA
        cca = CCA(n_components=self.encode_len)
        cca.fit(X, y)
        self._project = cca.transform
        # ITQ to find the best rotation matrix R
        self._R = self._itq(X)
