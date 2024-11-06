import numpy as np
import torch
from sklearn.decomposition import PCA

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

    def fit(self, X, n_iter=3):
        """Compute principal components and learn rotation matrix.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            n_iter: int (default=3).
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
            Z = np.matmul(V, R)
            B = sign(Z)
            C = np.matmul(B.T, V)
            S, _, S_hat_T = np.linalg.svd(C)
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

    def save_model(self, filename):
        """Save the model parameters to a file."""
        torch.save({
            'rotation_matrix': self._R,
            'encode_len': self.encode_len,
            'pca_components': self.pca.components_,
            'pca_mean': self.pca.mean_,
            'pca_explained_variance': self.pca.explained_variance_,
        }, filename)

    def load_model(self, filename):
        """Load model parameters from a file."""
        checkpoint = torch.load(filename)
        self._R = checkpoint['rotation_matrix']
        self.encode_len = checkpoint['encode_len']
        self.pca = PCA(n_components=self.encode_len)
        self.pca.components_ = checkpoint['pca_components']
        self.pca.mean_ = checkpoint['pca_mean']
        self.pca.explained_variance_ = checkpoint['pca_explained_variance']
        self._project = lambda X: self.pca.transform(X)

    def query_image(self, image):
        """Convert a query image to a binary tensor using ITQ.

        # Parameters:
            image: array, shape = (n_features,).
                The query image features.
        # Returns:
            binary_tensor: torch.Tensor.
                The binary tensor representation of the image.
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        binary_code = self.encode(image.reshape(1, -1))[0]
        binary_tensor = torch.tensor((binary_code > 0).astype(int), dtype=torch.int32)
        return binary_tensor
