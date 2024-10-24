"""Supervised Discrete Hashing.

Shen, F., Shen, C., Liu, W., & Tao Shen, H. (2015).
Supervised Discrete Hashing. CVPR, 37–45.
"""
import numpy as np

from .model import Model
from ..utils import sign


class SDH(Model):
    def __init__(self, encode_len):
        """Supervised Discrete Hashing (with l2 loss for G-step).

        # Parameters:
            encode_len: int.
                Encode length of binary codes.
        # Returns:
            None.
        """
        super().__init__(encode_len)

    def fit(self,
            X, Y,
            n_anchors,
            n_iter=5,
            F_lambda=1e-2, F_nu=1e-5,
            G_lambda=1.,
            tol=1e-5):
        """Fit model to data.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            Y: array, shape = (n_samples, n_targets).
                The labels of the training data.
            n_anchors: int.
                Numer of anchor points.
            n_iter: int (default=5).
                Maximum iteration number.
            F_lambda: float (default=1e-2).
                Regularization parameter of Fmap (F-Step).
            F_nu: float (default=1e-5).
                Penalty Parameter of F(·).
            G_lambda: float (default=1.).
                Regularization Parameter of Gmap (G-Step).
            tol: float (default=1e-5).
                Tolerance for termination.
        # Returns:
            None.
        # Examples:
            Given X (shape=[n_samples, n_features]) and y (shape=[n_samples,]):
            >>> from hashing.model import SDH
            >>> from hashing.utils import one_hot_encoding
            >>> sdh = SDH(encode_len=32)
            >>> y = one_hot_encoding(y, n_classes=10)
            >>> sdh.fit(X, Y, n_anchors=1000)
        """
        self.n_anchor = n_anchors
        self.n_iter = n_iter
        self.F_lambda = F_lambda
        self.F_nu = F_nu
        self.G_lambda = G_lambda
        self.tol = tol
        self.anchors = None

        n_samples = X.shape[0]
        # shape: (n_samples, n_anchors)
        Phi_X = self._rbf_kernel(X)
        # add dull 1 for linear regression bias
        # shape: (n_samples, n_anchors + 1)
        Phi_X = np.hstack([Phi_X, np.ones((n_samples, 1))])
        # optimize the model
        # shape: (n_anchors + 1, encode_len)
        self._Wf = self._sdh(Phi_X, Y)

    def encode(self, X):
        """Encode `X` to binary codes.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The data.
        # Returns:
            B: array, shape = (n_samples, encode_len).
                Binary codes of X.
        """
        # shape: (n_samples, n_anchors)
        Phi_X = self._rbf_kernel(X)
        # add dull 1 for linear regression bias
        # shape: (n_samples, n_anchors + 1)
        Phi_X = np.hstack([Phi_X, np.ones((Phi_X.shape[0], 1))])
        # shape: (n_samples, encode_len)
        Z = np.matmul(Phi_X, self._Wf)
        B = sign(Z)
        return B

    def _rbf_kernel(self, X, sigma=0.4):
        """The RBF kernel mapping.

        K(x, x') = exp(-||x - x'||^2 / (2 * sigma^2)).

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The data.
            sigma: float.
                The kernel width.
        # Returns:
            X_out: array, shape = (n_samples, n_anchors).
                The mapped data.
        """
        from scipy.spatial.distance import cdist
        if self.anchors is None:
            n_samples = X.shape[0]
            anchor_idx = np.random.permutation(
                np.arange(n_samples))[:self.n_anchor]
            self.anchors = X[anchor_idx]
        # shape: (n_samples, n_anchors)
        X_out = cdist(X, self.anchors, metric='sqeuclidean')
        X_out = np.exp(-1 * X_out / (2 * sigma ** 2))
        return X_out

    def _sdh(self, X, Y):
        """Joint learning with l2 loss by discrete cyclic coordinate
        descent (DCC) method.

        # Parameters:
            X: array, shape = (n_samples, n_features).
                The training data.
            Y: array, shape = (n_samples, n_targets).
                The labels of the training data.
        # Returns:
            Wf: array, shape = (n_features, encode_len).
                Projection matrix of Fmap (F-step).
        """
        from sklearn.linear_model import Ridge
        from numpy.linalg import norm

        n_samples = X.shape[0]
        # init B
        # shape: (n_samples, encode_len)
        Z = np.random.randn(n_samples, self.encode_len)
        B = sign(Z)

        # G-step
        G_ridge_model = Ridge(alpha=self.G_lambda, fit_intercept=False)
        G_ridge_model.fit(B, Y)
        # shape: (encode_len, n_targets)
        Wg = G_ridge_model.coef_.T

        # F-step
        F_ridge_model = Ridge(alpha=self.F_lambda, fit_intercept=False)
        F_ridge_model.fit(X, B)
        # shape: (n_features, encode_len)
        Wf = F_ridge_model.coef_.T

        for _ in range(self.n_iter):

            # B-step
            # shape: (n_samples, encode_len)
            Q = np.matmul(Y, Wg.T) + self.F_nu * np.matmul(X, Wf)
            for _ in range(10):
                B_prev = B.copy()
                # learn bit by bit
                for k in range(self.encode_len):
                    # shape: (n_samples, encode_len - 1)
                    B_no_k = np.delete(B, k, axis=1)
                    # shape: (encode_len - 1, n_targets)
                    Wg_no_k = np.delete(Wg, k, axis=0)
                    v = Wg[k]
                    # shape: (n_samples,)
                    z = Q[:, k] - np.matmul(np.matmul(B_no_k, Wg_no_k), v.T)
                    B[:, k] = sign(z)
                if norm(B - B_prev, 'fro') < self.tol * norm(B_prev, 'fro'):
                    break

            # G-step
            G_ridge_model.fit(B, Y)
            # shape: (encode_len, n_targets)
            Wg = G_ridge_model.coef_.T

            # F-step
            Wf_prev = Wf.copy()
            F_ridge_model.fit(X, B)
            # shape: (n_features, encode_len)
            Wf = F_ridge_model.coef_.T

            if norm(B - np.matmul(X, Wf), 'fro') < self.tol * norm(B, 'fro'):
                break
            if norm(Wf - Wf_prev, 'fro') < self.tol * norm(Wf_prev):
                break

        return Wf
