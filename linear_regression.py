__author__ = 'Alok'


def gradient_descent(training_data, alpha, theta, iterations=1500):
    """
    :param training_data: tuple containing the training data
    :param alpha:
    :return: tuple containing theta_1 and theta_2
    """
    x = tuple((1, i[0]) for i in training_data)
    y = tuple(i[1] for i in training_data)
    m = len(y)
    theta_len = len(theta)

    for i in range(iterations):
        # print(cost_function(x, y, theta))
        theta = tuple(theta[j] - alpha/m*sum((hx(theta, x[i]) - y[i])*x[i][j]
                                               for i in range(m))
                      for j in range(theta_len))

    return theta


def hx(theta, x):
    return x[0]*theta[0] + x[1]*theta[1]


def cost_function(x, y, theta):
    m = len(y)
    return 1/(2*m) * sum([(hx(theta, x[i]) - y[i])**2 for i in range(m)])


from data.ex1data1 import data
import time


def benchmark(iterations, previous_duration=None):
    t = -time.time()
    theta = gradient_descent(data, 0.01, (0, 0), iterations)
    duration = (t+time.time())*1000
    time_factor = None
    if previous_duration:
        time_factor = round(duration / previous_duration, 2)
    print("Time taken for {0} iterations: {1}ms with time factor: {2}".format(iterations, duration, time_factor))
    return duration


test_count = 10
t = None
for i in range(test_count):
    t = benchmark(1000 * (2 ** i), t)