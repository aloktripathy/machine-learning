import numpy as np
from numba import jit

class NeuralNetError(Exception):
    pass


class NeuralNet(object):
    def __init__(self, layers, lmda=0):
        """
        :param layers: list containing number of nodes in each layer of the network
        :param lmda: regularization factor
        """
        # validate layers
        if not (isinstance(layers, list) and len(layers) > 1):
            raise NeuralNetError('attribute `layers` should be of type list with more than'
                                 ' two elements')

        if len(layers) < 2 or not all([isinstance(i, int) and (i > 0) for i in layers]):
            raise NeuralNetError('elements in attribute `layers` should be non-negative integers')

        self.layers = layers
        self.lmda = lmda

        self.n_layers = len(layers)

        self.weights = None
        self.random_initialize_weights()

    def random_initialize_weights(self):
        """
        Initialize weights using gaussian distribution with mean 0 and variance 1
        :return: None
        """
        self.weights = [np.random.randn(self.layers[i + 1], j + 1)
                        for i, j in enumerate(self.layers[:-1])]

    def train(self, x, y, iterations, learning_rate, testing_data=None):
        self.gradient_descent(x, y, iterations, learning_rate)

        if testing_data:
            print(self.cost(testing_data[0], testing_data[1]))

    def gradient_descent(self, x, y, iterations, learning_rate):
        """

        """
        for i in range(iterations):
            gradients = self.back_prop(x, y)
            self.weights = [weight - learning_rate * gradient
                            for weight, gradient in zip(self.weights, gradients)]

    def feed_forward(self, x):
        """
        Compute activation values for nodes in each layer by using feed-forward algorithm
        :param x: input
        :return: activation values
        """
        activations = list()
        activations.append(self.insert_in_matrix(x, position=0, value=1, in_axis=1))

        for i in range(self.n_layers - 1):
            z = np.dot(activations[i], self.weights[i].transpose())
            a = self.sigmoid(z)

            '''add activation for bias unit for all but output layer'''
            if i != self.n_layers - 2:
                a = self.insert_in_matrix(a, position=0, value=1, in_axis=1)

            activations.append(a)

        return activations

    def back_prop(self, x, y):
        """ Perform back propagation algorithm on the model using the training data x, y.

        Args:
            x: input data
            y: output labels

        Returns:
            delta term
        """
        m = x.shape[0]
        activations = self.feed_forward(x)

        # use activations to compute errors in prediction
        errors = list()
        # error for output layer is predicted output minus actual output
        errors.append(activations[-1] - y)

        '''
        The following loop evaluated error matrices for all but input and output layer
        Examples:
            for a neural network with one hidden layer, 3 layers total, this loop will
            run only once, equivalent to-
            for i in [1]:
                ...
            similarly for another network with 3 hidden layers (5 total), the expression
             would be-
            for i in [3, 2, 1]:
                ...
        The idea is to evaluate error matrices for hidden layers in backward direction one
        by one
        '''
        for i in reversed(range(1, self.n_layers - 1)):
            e = errors[0]
            z = activations[i] * (1 - activations[i])   # sigmoid gradient evaluation

            error = np.dot(e, self.weights[i]) * z
            errors.insert(0, error[:, 1:])              # prepend the back-propagated error matrix

        # compute weight gradients or delta matrices
        weight_gradients = []

        for i, error in enumerate(errors):
            gradient = np.dot(error.transpose(), activations[i]) / m
            regularization_penalty = self.lmda / m * self.insert_in_matrix(self.weights[i][:, 1:],
                                                                           value=0, position=0,
                                                                           in_axis=1)
            weight_gradients.append(gradient + regularization_penalty)

        return weight_gradients

    def predict(self, x):
        """
        Computes the output for input matrix x
        :param x: input data matrix
        :return: predicted output
        """
        activations = self.feed_forward(x)
        return activations[-1]

    def cost(self, x, y, include_accuracy=False):
        """
        Computes cost of prediction on input matrix x against actual output y
        :param x: input data matrix
        :param y: actual output
        :param include_accuracy: if true, then cost function returns classification accuracy
        :return: cost of prediction
        """
        m = x.shape[0]
        hx = self.predict(x)

        entropy_cost = np.sum(-1 / m * (y * np.log(hx) + (1 - y) * np.log(1 - hx)))

        weights_squared_sum = sum(np.sum(np.power(weight[:, 1:], 2)) for weight in self.weights)
        regularization_penalty = self.lmda / (2 * m) * weights_squared_sum

        cost = entropy_cost + regularization_penalty

        if not include_accuracy:
            return cost
        else:
            p_digits = np.argmax(hx, axis=1)
            y_digits = np.argmax(y, axis=1)
            accuracy = sum([1 if i == j else 0 for i, j in zip(p_digits, y_digits)]) / len(p_digits)
            # print('predicted-', p_digits[1:30])
            # print('actual----', y_digits[1:30])
            return cost, accuracy

    @staticmethod
    def insert_in_matrix(array, position, value, in_axis):
        """
        Inserts a row/column in a matrix with given value to the given position in the given axis
        index. Acts as a wrapper around numpy's insert function for the sake of simplicity
        The arguments names are self-explanatory
        """
        return np.insert(array, position, values=value, axis=in_axis)

    @staticmethod
    def sigmoid(a):
        return 1.0/(1.0 + np.exp(-a))

    @staticmethod
    def feature_scale(a):
        # TODO: Complete this method when in need
        pass
