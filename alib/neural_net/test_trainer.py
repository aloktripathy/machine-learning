__author__ = 'Alok'

import math

import numpy as np
from matplotlib import pyplot as plt

from alib.neural_net import NeuralNet


def get_sines(n):
    while True:
        x = 2 * math.pi * np.random.rand(n, 1)
        y = np.sin(x) / 2 + 0.5
        yield x, y


def sine_training():
    n = NeuralNet([1, 5, 1])
    sines = get_sines(10)
    testing_data = next(sines)

    for i in range(1000):
        x, y = next(sines)
        n.train(x, y, 10, 1, testing_data=testing_data)

    # validation
    x, y = next(sines)
    hx = n.predict(x)

    errors = []
    print('doing validation---------------------------')
    for x, y, y_predicted in zip(x, y, hx):
        errors.append(abs(y[0] - y_predicted[0]) / y[0])
        print(x[0], y[0], y_predicted[0])

    print(sum(errors) / len(errors))

    x = np.reshape([2 * math.pi * i/1000 for i in range(1000)], (1000, 1))
    y = [(i[0] - 0.5)*2 for i in n.predict(x)]
    y_actual = [i[0] for i in np.sin(x)]

    plt.plot(x, y_actual)
    plt.hold(True)
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    sine_training()
