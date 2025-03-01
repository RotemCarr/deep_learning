import time
from logging import getLogger
import pandas as pd
from network import Network
from utils.data_loader import preprocess_data
from utils.evaluator import evaluate


logger = getLogger("network_training")
mnist_train = pd.read_csv('./data/mnist_train.csv')
mnist_test = pd.read_csv('./data/mnist_test.csv')
training_data = preprocess_data(mnist_train)
test_data = preprocess_data(mnist_test)


def test_model_training():
    logger.info("Starting network training...")
    network = Network([784, 128, 64, 10])
    network.train(
        training_data=training_data,
        epochs=3,
        batch_size=5,
        learning_rate=0.03,
    )

    accuracy = evaluate(network, test_data)
    logger.info(f" accuracy is {accuracy} / {len(test_data)}")
    assert accuracy / len(test_data) > 0.95


def test_training_time():
    logger.info("Starting network training...")
    start_time = time.time()
    network = Network([784, 128, 64, 10])
    network.train(
        training_data=training_data,
        epochs=3,
        batch_size=5,
        learning_rate=0.03,
    )
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f" training time is {duration} seconds.")
    assert end_time - start_time < 60
