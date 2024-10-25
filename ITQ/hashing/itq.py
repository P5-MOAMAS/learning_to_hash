import numpy as np
import torch
from sklearn.decomposition import PCA
from model import Model  # Ensure you have a base model class defined

def sign(x):
    """Return the sign of the input tensor, converting values to -1 or 1."""
    return torch.where(x > 0, torch.tensor(1.0), torch.tensor(-1.0)).float()

class _ITQ_Base(Model):
    """Base class for ITQ."""
    def __init__(self, encode_len):
        super().__init__(encode_len)

    def _itq(self, X):
        """Iterative quantization to find optimal rotation."""
        V = self._project(X)
        R = np.random.randn(self.encode_len, self.encode_len)
        R, _ = np.linalg.qr(R)  # Orthogonal initialization

        for _ in range(self.n_iter):
            Z = np.dot(V, R)
            B = sign(torch.tensor(Z))  # Ensure B is a tensor
            C = np.dot(B.T, V)
            S, _, S_hat_T = np.linalg.svd(C)
            R = np.dot(S, S_hat_T).T
        return R

    def encode(self, X):
        """Encode `X` to binary codes."""
        V = self._project(X)
        Z = np.dot(V, self._R)
        B = sign(torch.tensor(Z))  # Ensure B is a tensor
        return B.numpy()  # Ensure it returns a NumPy array


class ITQ(_ITQ_Base):
    def __init__(self, encode_len):
        super().__init__(encode_len)

    def fit(self, X, n_iter=50):
        """Fit the model to the data."""
        self.n_iter = n_iter
        pca = PCA(n_components=self.encode_len)
        V = pca.fit_transform(X)
        self._project = lambda X: pca.transform(X)
        self._R = self._itq(X)
