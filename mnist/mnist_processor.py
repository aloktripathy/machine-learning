__author__ = 'Alok'

import pickle
import gzip
import math
import time

import numpy as np

from matplotlib import pyplot as plt, cm
from alib.data_reader import AbstractDataSet


class MNISTReader(AbstractDataSet):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None

    def load(self, verbose=False):
        t = -time.time()
        self.training_data, self.validation_data, self.test_data = load_data_wrapper(self.file_path)
        t += time.time()
        if verbose:
            print('time taken to initialize load MNIST data from disk: {0} seconds'.
                  format(round(t, 2)))

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

        # images = images[:5000]
        # labels = labels[:5000]

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


"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""


def load_data(file_path):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    with gzip.open(file_path, 'rb') as f:
        # workaround: doesn't work without encoding attr
        training_data, validation_data, test_data = pickle.load(f)

    return training_data, validation_data, test_data


def load_data_wrapper(file_path):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data(file_path)

    # the following lines of codes are replaced below by Alok Tripathy
    """
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data
    """
    return tr_d, va_d, te_d


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e