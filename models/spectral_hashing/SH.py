import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle


class SpectralHashing:
    def __init__(self, code_length):
        """
        Initialize the Spectral Hashing model.

        Args:
            code_length (int): Length of the hash codes to generate.
        """
        self.code_length = code_length
        self.pca = None  # PCA model
        self.mn = None  # Minimum values after PCA
        self.R = None  # Range of values after PCA
        self.modes = None  # Eigenfunctions modes
        self.omegas = None  # Omega values for modes
        self.omega0 = None  # Base omega values

    def fit(self, train_data):
        """
        Train the Spectral Hashing model by performing PCA, fitting the uniform distribution,
        and enumerating eigenfunctions.

        Args:
            train_data (torch.Tensor or np.ndarray): Training data.
        """
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.numpy()

        # Step 1: Perform PCA on the training data
        X = self.perform_pca(train_data)

        # Step 2: Fit a uniform distribution to the PCA-transformed data
        self.fit_uniform_distribution(X)

        # Step 3: Compute omega0 (base frequencies)
        self.compute_omega0()

        # Step 4: Compute modes (eigenfunctions indices)
        modes = self.compute_modes()

        # Step 5: Compute omega values for modes
        omegas = self.compute_omegas(modes)

        # Step 6: Compute eigenvalues for the modes
        eig_vals = self.compute_eigenvalues(omegas)

        # Step 7: Select top modes based on eigenvalues
        self.select_top_modes(modes, omegas, eig_vals)

    def perform_pca(self, data):
        """
        Perform PCA on the training data to reduce dimensionality.

        Args:
            data (np.ndarray): Training data.

        Returns:
            np.ndarray: Data transformed by PCA.
        """
        self.pca = PCA(n_components=self.code_length)
        X = self.pca.fit_transform(data)
        return X

    def fit_uniform_distribution(self, X):
        """
        Fit a uniform distribution to the PCA-transformed data by calculating the minimum and
        maximum values, then computing the range.

        Args:
            X (np.ndarray): PCA-transformed data.
        """
        eps = np.finfo(float).eps  # Machine epsilon to prevent division by zero
        self.mn = X.min(axis=0) - eps
        mx = X.max(axis=0) + eps
        self.R = mx - self.mn  # Range of values

    def compute_omega0(self):
        """
        Compute the base omega values (frequencies) for each dimension.
        """
        self.omega0 = np.pi / self.R

    def compute_modes(self):
        """
        Compute the modes (indices of the eigenfunctions) for spectral hashing.

        Returns:
            np.ndarray: Modes matrix.
        """
        # Calculate the maximum number of modes for each dimension
        max_mode = np.ceil((self.code_length + 1) * self.R / self.R.max()).astype(int)
        n_modes = max_mode.sum() - len(max_mode) + 1

        # Initialize modes matrix
        modes = np.ones((n_modes, self.code_length))
        m = 0
        for i in range(self.code_length):
            # Assign modes for each dimension
            modes[m + 1 : m + max_mode[i], i] = np.arange(1, max_mode[i]) + 1
            m = m + max_mode[i] - 1

        modes -= 1  # Adjust modes by subtracting 1
        return modes

    def compute_omegas(self, modes):
        """
        Compute the omega values for each mode.

        Args:
            modes (np.ndarray): Modes matrix.

        Returns:
            np.ndarray: Omegas corresponding to the modes.
        """
        omegas = modes * self.omega0.reshape(1, -1)
        return omegas

    def compute_eigenvalues(self, omegas):
        """
        Compute the eigenvalues for the given omegas.

        Args:
            omegas (np.ndarray): Omegas corresponding to the modes.

        Returns:
            np.ndarray: Eigenvalues for each mode.
        """
        eig_vals = -np.sum(omegas**2, axis=1)
        return eig_vals

    def select_top_modes(self, modes, omegas, eig_vals):
        """
        Select the top modes based on eigenvalues to use for hashing.

        Args:
            modes (np.ndarray): Modes matrix.
            omegas (np.ndarray): Omegas corresponding to the modes.
            eig_vals (np.ndarray): Eigenvalues for each mode.
        """
        # Sort modes based on eigenvalues in descending order
        sorted_indices = np.argsort(-eig_vals)
        # Exclude the first index (constant eigenfunction)
        indices = sorted_indices[1 : self.code_length + 1]
        # Select top modes and their omegas
        self.modes = modes[indices, :]
        self.omegas = omegas[indices, :]

    def query(self, data):
        """
        Generate hash codes for the given data using the trained spectral hashing model.

        Args:
            data (torch.Tensor or np.ndarray): Data to generate hash codes for.

        Returns:
            np.ndarray: Generated hash codes.
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # Transform data using PCA and adjust by the minimum values
        data = data.reshape(1, -1)
        data = self.pca.transform(data) - self.mn.reshape(1, -1)

        # Initialize the hash code matrix
        U = np.zeros((len(data), self.code_length))
        for i in range(self.code_length):
            # Compute the i-th hash function
            omegai = self.omegas[i, :]
            ys = np.sin(data * omegai + np.pi / 2)
            yi = np.prod(ys, axis=1)
            U[:, i] = yi

        # Return the sign of U as the binary hash codes
        return np.sign(U)

    def save_model(self, filepath):
        """
        Save the Spectral Hashing model to a file.

        Args:
            filepath (str): Path to the file where the model will be saved.
        """
        # Prepare the model parameters to save
        model_params = {
            "code_length": self.code_length,
            "pca": self.pca,
            "mn": self.mn,
            "R": self.R,
            "modes": self.modes,
            "omegas": self.omegas,
            "omega0": self.omega0,
        }

        # Save the model parameters using pickle
        with open(filepath, "wb") as f:
            pickle.dump(model_params, f)

    @classmethod
    def load_model(cls, filepath):
        """
        Load the Spectral Hashing model from a file.

        Args:
            filepath (str): Path to the file where the model is saved.

        Returns:
            SpectralHashing: The loaded Spectral Hashing model.
        """
        # Load the model parameters using pickle
        with open(filepath, "rb") as f:
            model_params = pickle.load(f)

        # Create a new instance of SpectralHashing
        model = cls(model_params["code_length"])

        # Set the loaded parameters
        model.pca = model_params["pca"]
        model.mn = model_params["mn"]
        model.R = model_params["R"]
        model.modes = model_params["modes"]
        model.omegas = model_params["omegas"]
        model.omega0 = model_params["omega0"]
