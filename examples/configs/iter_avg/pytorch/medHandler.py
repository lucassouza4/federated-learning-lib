import logging

import numpy as np
import tensorflow as tf

from ibmfl.data.data_handler import DataHandler
logger = logging.getLogger(__name__)

class PneumoniaMedmnistDataHandler(DataHandler):
    """
    Data handler for MNIST dataset.
    """

    def __init__(self, data_config=None, channels_first=False):
        super().__init__()

        self.file_name = None
        if data_config is not None:
            if 'npz_file' in data_config:
                self.file_name = data_config['npz_file']
        self.channels_first = channels_first

        # load the datasets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_dataset()
        # pre-process the datasets
        self.preprocess()
    
    def load_dataset(self, nb_points=500):
        """
        Loads the training and testing datasets from a given local path. \
        If no local path is provided, it will download the original MNIST \
        dataset online, and reduce the dataset size to contain \
        500 data points per training and testing dataset. \
        Because this method \
        is for testing it takes as input the number of datapoints, nb_points, \
        to be included in the training and testing set.

        :param nb_points: Number of data points to be included in each set if
        no local dataset is provided.
        :type nb_points: `int`
        :return: training and testing datasets
        :rtype: `tuple`
        """
        # if self.file_name is None:
        #     (x_train, y_train), (x_test, y_test) = load_mnist()
        #     # Reduce datapoints to make test faster
        #     x_train = x_train[:nb_points]
        #     y_train = y_train[:nb_points]
        #     x_test = x_test[:nb_points]
        #     y_test = y_test[:nb_points]
        # else:
        try:
            logger.info('Loaded training data from ' + str(self.file_name))
            data_train = np.load(self.file_name)
            x_train = data_train['train_images']
            y_train = data_train['train_labels']
            x_test = data_train['test_images']
            y_test = data_train['test_labels']
        except Exception:
            raise IOError('Unable to load training data from path '
                            'provided in config file: ' +
                            self.file_name)
        return (x_train, y_train), (x_test, y_test)

    def get_data(self):
        """
        Gets pre-processed mnist training and testing data. 

        :return: training and testing data
        :rtype: `tuple`
        """
        return ((self.x_train, self.y_train), (self.x_test, self.y_test))

    def preprocess(self):
        """
        Preprocesses the training and testing dataset, \
        e.g., reshape the images according to self.channels_first; \
        convert the labels to binary class matrices.

        :return: None
        """
        num_classes = 10
        img_rows, img_cols = 28, 28

        if self.channels_first:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
            
        self.y_train = np.eye(num_classes)[self.y_train]
        self.y_test = np.eye(num_classes)[self.y_test]
        self.y_train = self.y_train.reshape((-1,num_classes))
        self.y_test = self.y_test.reshape((-1,num_classes))