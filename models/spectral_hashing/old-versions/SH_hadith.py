import os
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.io import loadmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from sklearn.decomposition import PCA


class SH:
    def __init__(
        self, K=8, Z_spec=None, random_state=42, init="random", n_components=None
    ):
        """
        Initialize the Spectral Hashing (SH) model with PCA.

        :param K: Number of bits for each binary code.
        :param Z_spec: Dictionary specifying how to compute the anchor-based representation.
        :param random_state: Random seed for reproducibility.
        :param init: Initialization method for anchor points ('random' or 'kmeans').
        :param n_components: Number of components for PCA.
        """
        self.K = K
        self.Z_spec = Z_spec
        self.random_state = random_state
        self.init = init
        self.n_components = n_components
        self.pca = None
        self.anchors = None
        self.W = None

    def fit(self, X_train):
        """
        Fit the Spectral Hashing model on the training data.

        :param X_train: Training data with shape (n_samples, n_features).
        """
        # Step 1: Apply PCA for dimensionality reduction
        if self.n_components:
            self.pca = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
            X_train = self.pca.fit_transform(X_train)

        # Step 2: Compute the anchor-based representation and binary codes
        self.Z_train, self.anchors = self._compute_affinity(X_train)
        self.B_train = self._compute_binary_codes(self.Z_train)

        # Step 3: Compute the projection matrix for out-of-sample extension
        self.W = self._projection_matrix(self.B_train, self.Z_train)

    def query(self, X_test):
        """
        Generate binary codes for new (unseen) samples.

        :param X_test: Test data with shape (n_test, n_features).
        :return: Binary codes with shape (n_test, K).
        """
        # Apply PCA to the test data if PCA was used in training
        if self.pca:
            X_test = self.pca.transform(X_test)

        # Generate anchor-based representation and binary codes for test data
        Z_test = self._to_Z(X_test, self.anchors)
        B_test = Z_test @ self.W.T
        return np.sign(B_test)

    def _compute_affinity(self, data):
        n_anchors = self.Z_spec["n_anchors"]
        if self.init.lower() == "random":
            R = np.random.RandomState(self.random_state)
            anchors = data[R.choice(data.shape[0], size=n_anchors, replace=False), :]
        elif self.init.lower() == "kmeans":
            kmeans = KMeans(n_clusters=n_anchors, random_state=self.random_state).fit(
                data
            )
            anchors = kmeans.cluster_centers_
        Z = self._to_Z(data, anchors)
        return Z, anchors

    def _to_Z(self, inputs, anchors):
        s = self.Z_spec["s"]
        sigma = self.Z_spec["sigma"]
        Dis = np.float32(cdist(inputs, anchors, metric=self.Z_spec["metric"]))
        min_val, min_pos = (
            np.zeros((inputs.shape[0], s), dtype="float32"),
            np.zeros((inputs.shape[0], s), dtype="int"),
        )

        for i in range(s):
            min_pos[:, i] = np.argmin(Dis, axis=1)
            min_val[:, i] = Dis[np.arange(inputs.shape[0]), min_pos[:, i]]
            Dis[np.arange(inputs.shape[0]), min_pos[:, i]] = np.inf

        if sigma is None:
            sigma = np.mean(min_val[:, -1])
        min_val = np.exp(-((min_val / sigma) ** 2))
        min_val /= np.sum(min_val, axis=1, keepdims=True)
        Z = np.zeros((inputs.shape[0], anchors.shape[0]), dtype="float32")

        for i in range(s):
            Z[np.arange(inputs.shape[0]), min_pos[:, i]] = min_val[:, i]

        return Z

    def _compute_binary_codes(self, Z):
        Z = csr_matrix(Z, dtype="float32")
        D = np.array(1 / np.sqrt(np.sum(Z, axis=0)), dtype="float32")
        D = diags(D.flatten(), dtype="float32")
        M = (D @ Z.T @ Z @ D).toarray()
        lambdas, eigen_vecs = np.linalg.eigh(M)
        idx_eig = np.argsort(-lambdas)
        lambdas, eigen_vecs = (
            lambdas[idx_eig[1 : self.K + 1]],
            eigen_vecs[:, idx_eig[1 : self.K + 1]],
        )
        S = np.diag(1 / np.sqrt(lambdas))
        B = Z @ D @ eigen_vecs @ S
        B = np.sqrt(Z.shape[0]) * B
        return np.sign(B)

    def _projection_matrix(self, B, Z):
        D = np.array(1 / (np.sum(Z, axis=0)), dtype="float32")
        D = np.diag(D.flatten())
        return B.T @ Z @ D


def cifar10_gist(path, one_hot=True, **kwargs):
    """
    Inputs:
        path: The path containing cifar10_gist512 features
        one_hot: If True, return one hoted labels.
    Outputs:
        train features, train labels, test features, and test labels.
    """
    train_path = os.path.join(path, "cifar10_gist512_train.mat")
    test_path = os.path.join(path, "cifar10_gist512_test.mat")

    train_dict = loadmat(train_path, squeeze_me=True)
    test_dict = loadmat(test_path, squeeze_me=True)

    train_features, train_labels = (
        train_dict["train_features"],
        train_dict["train_labels"],
    )
    test_features, test_labels = test_dict["test_features"], test_dict["test_labels"]

    return train_features, train_labels, test_features, test_labels


# map dataset names to dataset loaders
Dataset_maps = {"cifar10_gist512": cifar10_gist}


# loader function
def load_dataset(name, path=".", **kwargs):
    """
    the name of dataset. It can be one of 'cifar10_gist512', 'cifar10_vggfc7',
        'mnist_gist512', 'labelme_vggfc7', 'nuswide_vgg', 'colorectal_efficientnet',
        or 'sun397'.
    path: the path containing dataset files (Do not include the filenames of
        dataset)
    **kwargs are passed to the function that loads name dataset.
    """
    dataset_loader = Dataset_maps[name.lower()]
    train_features, train_labels, test_features, test_labels = dataset_loader(
        path, **kwargs
    )
    return train_features, train_labels, test_features, test_labels


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Load data
path = "./spectral-hashing/cifar10_gist512"  # Adjust this path as needed
dataset_name = "cifar10_gist512"
train_features, train_labels, test_features, test_labels = load_dataset(
    dataset_name, path=path, one_hot=False
)

# Normalize data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features.astype("float32"))
test_features = scaler.transform(test_features.astype("float32"))

# SH parameters
K = 16
Z_spec = {"n_anchors": 300, "s": 5, "sigma": None, "metric": "euclidean"}

# Initialize and train SH model
sh_model = SH(K=K, Z_spec=Z_spec, init="kmeans", random_state=42, n_components=100)
sh_model.fit(train_features)

# Generate binary codes for test data
test_codes = sh_model.query(test_features)
