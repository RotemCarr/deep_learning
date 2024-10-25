import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    pos_mask = (z >= 0)
    neg_mask = ~pos_mask
    result = np.zeros_like(z)

    result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    result[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
    return result


def sigmoid_prime(z: np.ndarray):
    return np.subtract(sigmoid(z), np.square(sigmoid(z)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_prime(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z))  # Stability improvement
    return exp_z / exp_z.sum(axis=0, keepdims=True)