from network import Network
import pandas as pd
from utils.evaluator import evaluate
from utils.data_loader import *


mnist_train = pd.read_csv('data/mnist_train.csv')
mnist_test = pd.read_csv('data/mnist_test.csv')
training_data = preprocess_data(mnist_train)
test_data = preprocess_data(mnist_test)


network = Network([784, 128, 64, 10])

network.train(
    training_data=training_data,
    epochs=3,
    batch_size=5,
    learning_rate=0.03,
)

accuracy = evaluate(network, test_data)
print(f"Accuracy on test data: {accuracy} / {len(test_data)}")

network.compile()
