"""Utilities used in hashing model."""
import numpy as np


def sign(X):
    """Sign function.

    sign(x) = -1 if x < 0 and 1 if x >= 0.

    # Parameters:
        X: array.
    # Returns:
        X_sign: array, same shape as `X`.
    """
    X_sign = np.ones(X.shape).astype(int)
    X_sign[X < 0] = -1
    return X_sign


def one_hot_encoding(y, n_classes):
    """Convert a class vector to binary class matrix.

    # Parameters:
        y: array, shape = (n_samples,).
            A class vector(integers).
        n_classes: int.
            Number of classes.
    # Returns:
        y_out: array, shape = (n_samples, n_classes).
            One-hot encoding representation of the input.
    """
    y_out = np.eye(n_classes)[y]
    assert y_out.shape == (y.shape[0], n_classes)
    return y_out
