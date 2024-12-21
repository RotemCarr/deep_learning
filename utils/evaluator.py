"""Utility module for evaluating Neural networks."""
import numpy as np


def evaluate(_network, _test_data):
    """Evaluate a Neural Network's performence."""
    # Count the number of correct predictions
    correct_predictions = sum(
        int(np.argmax(_network.feedforward(x)) == np.argmax(y))
        for x, y in _test_data
    )
    return correct_predictions
