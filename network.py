"""Module defining the Neural Network functionallity and structure."""
import ast
import pathlib
import random
import re
import time
from typing import List

import numpy as np
from utils.activations import relu, sigmoid, softmax, relu_prime


class Network:
    """Class defining the structure of each layer and
     the interaction between layers in the Neural Network."""

    def __init__(self, layer_sizes: List[int]):
        self.cost = None
        self.layer_sizes = layer_sizes
        self.weights = self._synoptic_weights()
        self.biases = self._synoptic_biases()

    def _synoptic_weights(self):
        return np.array([
            np.random.randn(y, x) * np.sqrt(2 / x)
            for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ], dtype=object)

    def _synoptic_biases(self):
        return np.array([np.random.randn(y, 1) for y in self.layer_sizes[1:]], dtype=object)

    def feedforward(self, activations: np.ndarray):
        """Feeding the Network forward for training and using the model."""
        for index, _ in enumerate(self.layer_sizes[1:]):
            z = np.dot(self.weights[index], activations) + self.biases[index]
            activations = relu(z) if \
                index < len(self.layer_sizes) - 2 else sigmoid(z)
        return activations

    def train(self, training_data, epochs, batch_size, learning_rate):
        """
        Training the network using Stochastic Gradient Descent.

        Args:
            training_data
            epochs (int)
            batch_size (int)
            learning_rate (float)
        """
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[j:j + batch_size] for j in range(0, n, batch_size)
            ]

            for batch in batches:
                self.update_batches(batch, learning_rate)
                print(f"Cost: {self.cost}")

            print(f"Epoch {i + 1} complete.")
            time.sleep(1)

    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        """Propagate backwards for the network to learn."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Feedforward
        activation = x
        activations = [x]  # List to store all activations layer by layer
        zs = []  # List to store all z vectors layer by layer

        for index, _ in enumerate(self.weights):
            z = np.dot(self.weights[index], activation) + self.biases[index]
            zs.append(z)
            activation = relu(z) if index < len(self.weights) - 1 else softmax(z)
            activations.append(activation)

        # Backward pass
        # Output layer delta (cross-entropy derivative)
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        self.cost = delta.sum()

        # Propagate error backward through the network
        for layer in range(2, len(self.layer_sizes)):
            z = zs[-layer]
            sp = relu_prime(z)  # Derivative of ReLU
            delta = np.dot(self.weights[-layer + 1].T, delta) * sp  # Propagate delta back
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].T)

        return nabla_w, nabla_b

    def update_batches(self, batch, learning_rate):
        """Update the mini batches after backpropagation."""
        # Initialize gradient sums for batch
        sum_nabla_w = [np.zeros(w.shape) for w in self.weights]
        sum_nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Accumulate gradients for each training example in the batch
        for x, y in batch:
            nabla_w, nabla_b = self.backpropagation(x, y)
            sum_nabla_w = [nw + dnw for nw, dnw in zip(sum_nabla_w, nabla_w)]
            sum_nabla_b = [nb + dnb for nb, dnb in zip(sum_nabla_b, nabla_b)]

        # Update weights and biases using average gradients
        self.weights = [w - (learning_rate / len(batch)) * nw
                        for w, nw in zip(self.weights, sum_nabla_w)]

        self.biases = [b - (learning_rate / len(batch)) * nb
                       for b, nb in zip(self.biases, sum_nabla_b)]

    def compile(self, path: str = "./trained_models"):
        """Compile a Neural Network to a file with the '.model' extension."""
        files = [f.name for f in pathlib.Path(path).iterdir() if f.is_file()]
        file_numbers = [
            int(m.group(1)) for file in files
            if (m := re.match(r"network_(\d+)\.model", file))
        ]

        new_model_number = np.max(file_numbers) + 1 if file_numbers else 1
        new_filename = pathlib.Path(f"{path}/network_{new_model_number}.model")
        while new_filename.exists():
            new_model_number += 1
            new_filename = pathlib.Path(
                f"{path}/network_{new_model_number}.model")

        with open(new_filename, "x", encoding="utf-8") as f:
            f.write(f"""{[self.layer_sizes,
                          [w.tolist() for w in self.weights],
                          [b.tolist() for b in self.biases]]}""")
            f.close()


def load(path: str):
    """Load a Neural Network from a given file."""
    with open(path, encoding="utf-8") as f:
        model = ast.literal_eval(f.read())
        f.close()

    network = Network(model[0])
    network.weights = [np.array(w) for w in model[1]]
    network.biases = [np.array(b) for b in model[2]]
    return network
