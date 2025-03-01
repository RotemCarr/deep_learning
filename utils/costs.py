"""Utility module for cost functions."""

import numpy as np


def quad_cost(predicted: np.ndarray, expected: np.ndarray):
    """Quadratic cost function."""
    return np.multiply(2, np.square(np.subtract(predicted, expected)))


def cross_entropy_cost(predicted: np.ndarray, expected: np.ndarray):
    """Cross entropy cost function."""
    epsilon = 1e-12  # Small value to avoid log(0)
    predicted = np.clip(predicted, epsilon, 1.0 - epsilon)  # Ensure no log(0)
    # Use average cost over the batch
    return -np.sum(expected * np.log(predicted)) / expected.shape[1]
