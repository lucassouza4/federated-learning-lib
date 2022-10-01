"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""

import logging
import keras
import time
import json
import numpy as np
import tensorflow as tf

# if tf.__version__[0:4] != "1.15":
#     raise ImportError("KerasFLModel requires TensorFlow v1.15.")

from keras_preprocessing.image.numpy_array_iterator import NumpyArrayIterator
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import set_session

from ibmfl.util import config
from ibmfl.util import fl_metrics
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import FLException, LocalTrainingException, ModelException

logger = logging.getLogger(__name__)


# Define Flower client
class Client(FLModel):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"test_accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    model = utils.get_model()
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    (x_train, y_train), (x_test, y_test) = utils.load_data(args.partition)
    client = Client(model, x_train, y_train, x_test, y_test)
    
    fl.client.start_numpy_client(
        # server_address="10.182.0.7:8080",
        server_address="localhost:8080",
        client=client,
    )

if __name__ == "__main__":
    main()

