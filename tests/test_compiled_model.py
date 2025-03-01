import os
import tempfile
from logging import getLogger

import pandas as pd
from network import Network, load
from utils.data_loader import preprocess_data
from utils.evaluator import evaluate

logger = getLogger("network_compiler")
mnist_train = pd.read_csv("./data/mnist_train.csv")
mnist_test = pd.read_csv("./data/mnist_test.csv")
training_data = preprocess_data(mnist_train)
test_data = preprocess_data(mnist_test)


def test_network_compile():
    logger.info("Starting network training...")
    network = Network([784, 128, 64, 10])
    network.train(
        training_data=training_data,
        epochs=3,
        batch_size=5,
        learning_rate=0.03,
    )
    temp_dir = tempfile.TemporaryDirectory()
    logger.info(
        f"Created temporary directory for network compilation: {temp_dir.name}..."
    )
    logger.info(f"Compiling network to {temp_dir.name}...")
    network.compile(temp_dir.name)
    logger.info("Checking to see if the network compiled successfully...")
    assert os.path.exists(f"{temp_dir.name}/network_1.model")
    temp_dir.cleanup()


def test_compiled_network_load():
    logger.info("Starting network training...")
    network = Network([784, 128, 64, 10])
    network.train(
        training_data=training_data,
        epochs=3,
        batch_size=5,
        learning_rate=0.03,
    )
    precompile_accuracy = evaluate(network, test_data)
    logger.info(f"Accuracy before compilation is: {precompile_accuracy}")
    temp_dir = tempfile.TemporaryDirectory()
    logger.info(
        f"Created temporary directory for network compilation: {temp_dir.name}..."
    )
    logger.info(f"Compiling network to {temp_dir.name}...")
    network.compile(temp_dir.name)
    logger.info(f"Loading compiled network from {temp_dir.name}...")
    network = load(f"{temp_dir.name}/network_1.model")
    logger.info("checking that accuracy is the same after compilation...")
    postcompile_accuracy = evaluate(network, test_data)
    assert postcompile_accuracy == precompile_accuracy
    temp_dir.cleanup()
