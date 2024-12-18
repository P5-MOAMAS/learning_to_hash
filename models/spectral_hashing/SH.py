# Reference: https://github.com/TreezzZ/SH_PyTorch

from sklearn.decomposition import PCA
import torch
import numpy as np


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
        # PCA: Reduce the dimensionality of the input dataset to 'nbits' dimensions
        pca = PCA(n_components=self.nbits)
        X_pca = pca.fit_transform(X)

        # Fit uniform distribution: Compute the range of the PCA-transformed data
        eps = np.finfo(float).eps  # Small value to ensure numerical stability
        mn = X_pca.min(0) - eps
        mx = X_pca.max(0) + eps

        # Eigenfunction enumeration
        R = mx - mn  # Range for each PCA dimension

        # Compute the maximum number of modes for each dimension based on scaling
        max_mode = np.ceil((self.nbits + 1) * R / R.max()).astype(np.int64)

        # Total number of modes across all dimensions
        n_modes = max_mode.sum() - len(max_mode) + 1

        # Initialize mode matrix: Each row represents a specific mode combination
        modes = np.ones([n_modes, self.nbits])
        m = 0
        for i in range(self.nbits):
            # Populate the modes for the current dimension
            modes[m + 1 : m + max_mode[i], i] = np.arange(1, max_mode[i]) + 1
            m = m + max_mode[i] - 1  # Update index for the next dimension
        modes -= 1  # Adjust modes to start at 0

        # Compute fundamental frequencies for each PCA dimension
        omega0 = np.pi / R
        # Scale frequencies by the mode matrix to get all combinations
        omegas = modes * omega0.reshape(1, -1).repeat(n_modes, 0)

        # Eigenvalue sorting
        
        # Compute eigenvalues (negative squared norms of omega)
        eig_val = -(omegas**2).sum(1)
        # Sort modes by eigenvalues in descending order
        ii = (-eig_val).argsort()
        # Select the top 'nbits' modes for hashing
        modes = modes[ii[1 : self.nbits + 1], :]


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

        # Apply the PCA transformation and center the data using the training range and the already computed PCA model with its principal
        # compoments
        data_pca = pca.transform(data) - mn.reshape(1, -1)

        # Compute scaled frequencies for eigenfunctions
        omega0 = np.pi / R # Fundamental frequqncy
        omegas = modes * omega0.reshape(1, -1)
        U = np.zeros((len(data), self.nbits))

        # Hashing code generation: Compute the binary codes for each bit
        for i in range(self.nbits):
            omegai = omegas[i, :]  # Frequencies for the current bit
            ys = np.sin(data_pca * omegai + np.pi / 2)  # Apply sinusoidal eigenfunction
            yi = np.prod(ys, 1)  # Combine results across dimensions
            U[:, i] = yi  # Store the combined value for this bit

        # Return the sign of the computed values as binary hash codes
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
