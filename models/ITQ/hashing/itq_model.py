import numpy as np
from sklearn.decomposition import PCA
"""
this Implementation uses a modified version of ITQ from https://github.com/wen-zhi/hashing 
"""
class Model:
    "Hashing model base."

    def __init__(self, encode_len):
        self.encode_len = encode_len

    def fit(self, X):
        raise NotImplementedError

    def encode(self, X):
        raise NotImplementedError


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


class ITQ(Model):
    """Iterative Quantization (ITQ) with PCA for dimension reduction."""

    def __init__(self, encode_len):
        super().__init__(encode_len)

    def fit(self, X, n_iter=50):
        """Compute principal components and learn rotation matrix.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            n_iter: int (default=50).
                Maximum number of iterations for ITQ.
        """
        self.n_iter = n_iter
        self.pca = PCA(n_components=self.encode_len)
        self.pca.fit(X)
        self._project = self.pca.transform
        self._R = self._itq(X)

    def _itq(self, X):
        """Iterative quantization to find optimal rotation.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
        # Returns:
            R: array, shape = (encode_len, encode_len).
                Optimal rotation matrix.
        """
        V = self._project(X)
        R = np.random.randn(self.encode_len, self.encode_len)
        R = np.linalg.svd(R)[0]

        for _ in range(self.n_iter):
            # Fix R and update B:
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
                The data to encode.
        # Returns:
            B: array, shape = (n_samples, encode_len).
                Binary codes of X.
        """
        V = self._project(X)
        Z = np.matmul(V, self._R)
        B = sign(Z)
        return B

    def encode_single(self, query_feature):
        """Encode a single feature vector to binary code.

        # Parameters:
            query_feature: array, shape = (n_features,).
            The single data point to encode.
        # Returns:
            B: array, shape = (1, encode_len).
                Binary code of the input feature, with values 0 or 1.
        """
        V = self._project(query_feature.reshape(1, -1))
        Z = np.matmul(V, self._R)
        B = sign(Z).astype(int)
        B[B == -1] = 0
        return B