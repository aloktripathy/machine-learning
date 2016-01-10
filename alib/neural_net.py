import numpy as np


class NeuralNet:
    def __init__(self, layers, r_lambda=0):
        """ Initialize the NeuralNet object.

        Args:
            layers(list): number of neurons in each layer including the input and output layer
            r_lambda(float): regularization factor
        """
        self.layers = tuple(layers)
        self.n_layers = len(layers)
        self.r_lambda = r_lambda

        # randomly initialize L-1 thetas
        self.thetas = []
        for i in range(self.n_layers - 1):
            theta_dimension = (layers[i+1], layers[i]+1)
            theta = np.random.randn(*theta_dimension)
            self.thetas.append(theta)

    def mini_batch_train(self, data_stream, algorithm, learning_rate=0.01,
                         iterations=20, cost_tracker=None, apply_feature_scaling=False):
        """ Train a data set with labels using algorithm.

        Args:
            data_stream(generator): returns packets of  Input/output data
            algorithm(function): choice of algorithm to be used
            learning_rate(float): the learning rate (alpha)
            iterations(int):
            cost_tracker:
            apply_feature_scaling:

        Returns:

        """
        costs = []

        for x, y in data_stream:
            # check dimensions
            self._validate_input_layer(x.shape)
            self._validate_output_layer(y.shape)

            # normalize data
            if apply_feature_scaling:
                x = self.scale_features(x)

            # compute cost
            cost = self.cost(x, y)
            costs.append(cost)

            # broadcast cost tracker if present
            if callable(cost_tracker):
                cost_tracker(costs)

            # perform training while tracking costs
            algorithm(self, x, y, learning_rate, iterations)

        return costs

    def gradient_descent(self, x, y, learning_rate, iterations):

        for i in range(iterations):
            # evaluate gradients using back-propagation
            delta_thetas = self._backprop(x, y)

            # update thetas
            self.thetas = [theta - learning_rate * delta_theta
                           for theta, delta_theta in zip(self.thetas, delta_thetas)]

    def _backprop(self, x, y):
        # feed-forward: initialize input value for each nodes in all the layers
        m = x.shape[0]
        A = [self.prepend_ones(x)]

        # compute activation values for rest of the layers
        for i in range(self.n_layers - 1):
            z = np.dot(A[i], self.thetas[i].transpose())
            hx = self.sigmoid(z)

            # last layer i.e o/p layer need not have a bias unit
            if i == self.n_layers - 2:
                a = hx
            else:
                a = self.prepend_ones(hx)

            A.append(a)

        # back propagate to compute errors
        errors = [None] * self.n_layers
        errors[-1] = A[-1] - y    # error for output layer

        # evaluate errors for all but output layer
        for i in reversed(range(self.n_layers - 1)):
            theta_del = np.dot(errors[i+1], self.thetas[i])

            z_prime = A[i] * (1 - A[i])

            errors[i] = (theta_del * z_prime)[:, 1:]

        # evaluate the gradient of thetas
        deltas = [np.dot(errors[i + 1].transpose(), A[i]) / m +
                  self.r_lambda / m * self.prepend_zeros(self.thetas[i][:, 1:])

                  for i in range(self.n_layers - 1)]

        return deltas

    def cost(self, x, y):
        """
        Computes cost for normalized feature-set x against label-set y
        """
        # check dimensions
        self._validate_input_layer(x.shape)
        self._validate_output_layer(y.shape)

        m = x.shape[0]

        # compute costs
        hx = self.predict(x)

        prediction_cost = -1/m * np.sum(y * np.log(hx) + (1-y) * np.log(1-hx))
        regularization_cost = self.r_lambda / (2 * m) * sum(np.sum(np.power(theta[:, 1:], 2)) for theta in self.thetas)

        return prediction_cost + regularization_cost

    def predict(self, x):
        """
        Predicts the output of a normalized feature-set x
        """
        self._validate_input_layer(x.shape)
        a = x
        for theta in self.thetas:
            a = np.dot(self.sigmoid(self.prepend_ones(a)), theta.transpose())

        return self.sigmoid(a)

    def _validate_input_layer(self, shape):
        if shape[1] != self.layers[0]:    # considering the extra bias feature 1
            raise ValueError('Feature set dimension mismatch; expecting feature set with'
                             ' cardinality: {0}, got: {0}'.format(self.layers[0], shape[1]))

    def _validate_output_layer(self, shape):
        if shape[1] != self.layers[-1]:
            raise ValueError('Output dimension mismatch; expecting outputs with cardinality: '
                             '{0}, got: {0}'.format(self.layers[-1], shape[1]))

    @staticmethod
    def sigmoid(a):
        return 1.0 / (1.0 + np.exp(-a))

    @staticmethod
    def scale_features(a):
        """
        Normalizes the features matrix a to zero-mean
        """
        return (a - np.mean(a, axis=0)) / np.std(a, axis=0)

    @staticmethod
    def prepend_ones(a):
        """
        Prepend ones in feature set then apply feature scaling
        """
        return np.insert(a, 0, values=1, axis=1)

    @staticmethod
    def prepend_zeros(a):
        """
        Prepend ones in feature set then apply feature scaling
        """
        return np.insert(a, 0, values=0, axis=1)
