__author__ = 'Alok'

import math
import numpy as np
from alib.neural_net import NeuralNet
from matplotlib import pyplot as plt


def get_sines(n):
    while True:
        x = 2 * math.pi * np.random.rand(n, 1)
        y = np.sin(x) / 2 + 0.5
        yield x, y


def cost_tracker(costs):
    print(costs[-1])

if __name__ == "__main__":
    n = NeuralNet([1, 5, 1])
    x, y = next(get_sines(10000))
    data = [(x[i:i+100], y[i:i+100]) for i in range(100)]
    n.mini_batch_train(data, NeuralNet.gradient_descent, learning_rate=0.01, iterations=20, cost_tracker=cost_tracker)

    x = np.reshape([2 * math.pi * i/1000 for i in range(1000)], (1000, 1))
    y = [(i[0] - 0.5)*2 for i in n.predict(x)]
    y_actual = [i[0] for i in np.sin(x)]

    plt.plot(x, y_actual)
    plt.hold(True)
    plt.plot(x, y)
    plt.show()
