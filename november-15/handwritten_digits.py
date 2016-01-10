__author__ = 'Alok'

import time
import pickle
import zlib
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from alib.neural_net import NeuralNet
from alib.dataset import AbstractDataSet
from alib.mnist_loader import load_data_wrapper

from stats import write_dump, read_dump


class FileReader(AbstractDataSet):
    """
    TODO: Yet to be completed
    """
    def __init__(self, input_file, labels_file):
        self.input_file = input_file
        self.labels_file = labels_file

    def get_input(self, items):
        for chunk in self.read_from_file(self.input_file, items):
            yield self.process_file_chunk(chunk)

    def get_labels(self, items):
        for chunk in self.read_from_file(self.labels_file, items):
            yield self.process_file_chunk(chunk)

    def get(self, items, skip=None):
        for a, b in zip(self.get_input(items), self.get_labels(items)):
            yield a, b

    @staticmethod
    def process_file_chunk(chunk):
        return chunk

    @staticmethod
    def read_from_file(file_path, num_lines):
        with open(file_path, "r") as fp:
            while True:
                chunk = fp.read(num_lines)
                # check if we've reached EOF
                if chunk == "":
                    break

                yield chunk


class MNISTReader(AbstractDataSet):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None

    def load(self):
        self.training_data, self.validation_data, self.test_data = load_data_wrapper(self.file_path)

    def get(self, limit, skip=None, *args, **kwargs):
        """
        Returns a generator expression to retrieve data in chunks
        limit: number of data sets
        kwargs:
            mode: training/validation/test
        """

        # check if data is loaded yet
        if not (self.training_data or self.validation_data or self.test_data):
            raise ValueError('Please load data-sets using load() before calling get()')

        mode = kwargs.get('mode', 'training')

        # validate mode
        if mode not in ('training', 'validation', 'test'):
            raise AttributeError('invalid value for attribute mode; correct values: training/validation/test')

        self.shuffle()

        images, labels = getattr(self, mode+'_data')

        images = images[:5000]
        labels = labels[:5000]

        size = images.shape[0]

        # convert digits to bit-array
        # if mode == 'training':
        labels = self.digit_to_bits(labels)

        for i in range(0, size, limit):
            yield images[i:i + limit], labels[i:i + limit]

    def shuffle(self):
        """
        Shuffle the training dataset
        :return:
        """
        images, labels = self.training_data
        images = np.asarray(images)
        labels = np.asarray(labels)
        c = np.c_[images.reshape(len(images), -1), labels.reshape(len(labels), -1)]

        np.random.shuffle(c)
        images = c[:, :images.size//len(images)].reshape(images.shape)
        labels = c[:, images.size//len(images)].reshape(labels.shape)

        self.training_data = images, labels

    @staticmethod
    def digit_to_bits(labels):
        bits_array = np.zeros((labels.shape[0], 10))
        for i, label in enumerate(labels):
            bits_array[i][label] = 1.0

        return bits_array

    @staticmethod
    def bits_to_digit(arr):
        pass

    @staticmethod
    def view_image(image_array):
        # roll the 1d image_array vector to a 28/28 matrix
        # subtract the matrix from 1 to invert color leading to a dark handwriting on a light background
        arr = 1 - np.reshape(image_array, (28, 28))

        plt.imshow(arr, cmap=cm.Greys_r)
        plt.show()


def cost_tracker(costs):
    print(costs[-1])


def train(reader, neural_net, metadata, track_cost=False, thetas=None):
    data_stream = reader.get(metadata['batch_size'], mode='training')

    t = -time.time()

    if thetas:
        neural_net.thetas = thetas

    tracker = cost_tracker if track_cost else None

    costs = neural_net.mini_batch_train(data_stream, NeuralNet.gradient_descent, cost_tracker=tracker,
                                        learning_rate=metadata['learning_rate'], iterations=metadata['iterations'])

    t += time.time()

    return costs, t, neural_net.thetas


def validate(metadata, thetas):
    reader = MNISTReader(metadata['data_file'])
    reader.load()
    data_stream = reader.get(10, mode='validation')

    n = NeuralNet(metadata['layers'])
    n.thetas = thetas

    counter = 0
    matches = 0

    for x, y in data_stream:
        counter += x.shape[0]
        predicted = n.predict(x)
        p_digits = np.argmax(predicted, axis=1)

        print('predicted', p_digits)
        print('actual', y)
        return None

if __name__ == "__main__":
    metadata = {
        'batch_size': 10,
        'layers': [784, 25, 10],
        'mode': 'training',
        'data_file': 'data/mnist.pkl.gz',
        'learning_rate': 0.1,
        'lambda': 0,
        'iterations': 1,
        'program': 'MNIST-handwritten-digits'
    }

    # training
    reader = MNISTReader(metadata['data_file'])
    reader.load()

    n = NeuralNet(metadata['layers'], r_lambda=metadata['lambda'])

    # costs, time_taken, thetas = train(reader, n, metadata, track_cost=True)

    costs = []
    thetas = None
    test_images, test_labels = next(reader.get(5000, mode='test'))
    for i in range(20):
        _costs, time_taken, thetas = train(reader, n, metadata, thetas=thetas)
        testing_cost = n.cost(test_images, test_labels)
        costs.append(testing_cost)
        print(testing_cost)
        reader.shuffle()

    data = {'thetas': thetas, 'time_taken_in_seconds': round(time_taken), 'metadata': metadata, 'costs': costs}
    # write_dump(data)

    print(time_taken)

    x = np.arange(len(costs))
    y = costs
    plt.plot(x, y)
    plt.show()

    """
    training_images, training_labels = next(reader.get(5000, mode='train'))
    model = BernoulliRBM(30, learning_rate=3, verbose=True)
    model.fit(X=training_images, y=training_labels)
    """

    # validation
    '''
    data = read_dump('dumps/MNIST-handwritten-digits-1447021461-554144.ml')
    validate(metadata, data['thetas'])
    '''
