"""
Utility module for preproccesing data for
Neural Networks training.
"""

import numpy as np


def preprocess_data(data):
    """Preproccess csv data format."""
    # Shuffle the data first
    data = data.sample(frac=1).reset_index(drop=True)

    # Extract labels (first column) and pixel values (remaining columns)
    labels = data.iloc[:, 0].values  # The first column contains the labels
    pixels = data.iloc[:, 1:].values  # The remaining columns contain the pixel values

    # Normalize pixel values to be between 0 and 1
    pixels = pixels / 255.0

    # Reshape pixels to have shape (784, 1) for each image
    pixels = [np.reshape(p, (784, 1)) for p in pixels]

    # One-hot encode the labels
    def one_hot_encode(y, num_classes=10):
        encoded = np.zeros((num_classes, 1))
        encoded[y] = 1.0
        return encoded

    # Convert labels to one-hot encoded format
    labels = [one_hot_encode(label) for label in labels]

    # Return the data as a list of tuples (input, output)
    return list(zip(pixels, labels))
