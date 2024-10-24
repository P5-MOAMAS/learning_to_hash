from .model import Model
from ..utils import sign

import numpy as np


class LSH(Model):
    def __init__(self, encode_len):
        """Locality Sensitive Hashing(LSH) based on random projection.

        # Parameters:
            encode_len: int (default=16).
                Encode length of binary codes.
        # Returns:
            None.
        """
        super().__init__(encode_len)

    def fit(self, X_train, zero_centered=True):
        """Generate parameters for LSH.

        # Parameters:
            X_train: array, shape = (n_samples, n_features).
                The data.
            zero_centered: bool (default=True).
                Whether to center the data.
        # Returns:
            None.
        """
        self.zero_centered = zero_centered
        n_features = X_train.shape[1]
        if self.zero_centered:
            # shape: (n_features,)
            self._mean = np.mean(X_train, axis=0)
        self.W = np.random.randn(n_features, self.encode_len)

    def encode(self, X):
        """Encode `X` to binary codes.

        # Parameters:
            X: array, shape = (n_samples, n_features).
        # Returns:
            B: array, shape = (n_samples, encode_len).
                Binary codes of X.
        """
        if self.zero_centered:
            X = X - self._mean
        Z = np.matmul(X, self.W)
        # covert to {-1, 1} binary encoding
        B = sign(Z)
        return B
