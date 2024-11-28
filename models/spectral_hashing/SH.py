from sklearn.decomposition import PCA
import pickle

import torch
import numpy as np
from sklearn.decomposition import PCA


class SpectralHashing:
    def __init__(self, nbits):
        """
        Initialize the Spectral Hashing model with the desired number of bits.
        Args:
            nbits (int): Number of bits for hashing.
        """
        self.nbits = nbits
        self.model_components = None

    def fit(self, X):
        """
        Train the Spectral Hashing model.
        Args:
          X (np.ndarray): Feature matrix [Nsamples, Nfeatures].
        Returns:
          None: The model components are stored in the instance for querying.
        """
        # PCA
        pca = PCA(n_components=self.nbits)
        X_pca = pca.fit_transform(X)

        # Fit uniform distribution
        eps = np.finfo(float).eps
        mn = X_pca.min(0) - eps
        mx = X_pca.max(0) + eps

        # Eigenfunction enumeration
        R = mx - mn
        max_mode = np.ceil((self.nbits + 1) * R / R.max()).astype(np.int64)
        n_modes = max_mode.sum() - len(max_mode) + 1
        modes = np.ones([n_modes, self.nbits])
        m = 0
        for i in range(self.nbits):
            modes[m + 1 : m + max_mode[i], i] = np.arange(1, max_mode[i]) + 1
            m = m + max_mode[i] - 1
        modes -= 1
        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(n_modes, 0)

        # Eigenvalue sorting
        eig_val = -(omegas**2).sum(1)
        ii = (-eig_val).argsort()
        modes = modes[ii[1 : self.nbits + 1], :]

        # Store trained components
        self.model_components = {"pca": pca, "mn": mn, "R": R, "modes": modes}

    def generate_code(self, data):
        """
        Generate hashing code for given data using the trained model components.
        Args:
            data (torch.Tensor): Data.
        Returns:
            torch.Tensor: Hash codes.
        """
        if self.model_components is None:
            raise ValueError("Model not trained. Please call 'train' first.")

        pca = self.model_components["pca"]
        mn = self.model_components["mn"]
        R = self.model_components["R"]
        modes = self.model_components["modes"]

        # PCA transform and preprocessing
        data_pca = pca.transform(data) - mn.reshape(1, -1)

        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1)
        U = np.zeros((len(data), self.nbits))

        # Hashing code generation
        for i in range(self.nbits):
            omegai = omegas[i, :]
            ys = np.sin(data_pca * omegai + np.pi / 2)
            yi = np.prod(ys, 1)
            U[:, i] = yi

        return torch.from_numpy(np.sign(U))

    def query(self, features):
        """
        Convert a query image to a binary tensor using the trained SH model.
        Args:
            features (torch.Tensor): Query image features.
        Returns:
            torch.Tensor: Binary hash code for the query image.
        """
        return self.generate_code(torch.as_tensor(features).unsqueeze(0))
