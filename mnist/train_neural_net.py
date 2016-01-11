import time
import numpy as np
import matplotlib.pyplot as plt

from alib.neural_net import NeuralNet
from mnist.mnist_processor import MNISTReader
from mnist.stats import write_dump, visualize_data


def validate(reader, weights):
    data_stream = reader.get(30, mode='validation')

    n = NeuralNet(metadata['layers'])
    n.weights = weights

    counter = 0
    matches = 0

    for x, y in data_stream:
        counter += x.shape[0]
        predicted = n.predict(x)
        p_digits = np.argmax(predicted, axis=1)
        y_digits = np.argmax(y, axis=1)
        accuracy = sum([1 if i == j else 0 for i, j in zip(p_digits, y_digits)]) / len(p_digits)

        print('predicted-\n', p_digits)
        print('actual-\n', y_digits)
        print('accuracy-{0}%'.format(round(accuracy * 100, 2)))
        return None     # Let's just run only one loop for the time being

if __name__ == "__main__":
    t_main = -time.time()
    metadata = {
        'batch_size': 20,
        'layers': [784, 33, 10],
        'mode': 'training',
        'data_file': 'mnist.pkl.gz',
        'learning_rate': 2,
        'lambda': 0.0001,
        'iterations': 1,
        'epochs': 10,
        'program': 'MNIST-HWD-NN-1'     # MNIST handwritten digits neural-net version 1
    }
    costs = []
    accuracies = []
    t = -time.time()

    network = NeuralNet(layers=metadata['layers'], lmda=metadata['lambda'])
    reader = MNISTReader(metadata['data_file'])

    reader.load()
    testing_data = next(reader.get(10000, mode='test'))

    # visualize a few handwritten digits
    # visualize_data(testing_data[0][:120, :])

    t += time.time()
    print('time taken to initialize neural net and load data: {0} seconds'.format(round(t, 2)))

    t = -time.time()
    # This is where prediction is happening
    for i in range(metadata['epochs']):
        '''
        MemoryError raised multiple times on the following line. Looks like Shuffling a
        50000 x 461 matrix, is beyond my laptop's capacity. Need an extra 4GB stick installed.
        '''
        # reader.shuffle()
        for x, y in reader.get(metadata['batch_size'], 'training'):
            network.train(x, y, iterations=metadata['iterations'],
                          learning_rate=metadata['learning_rate'])

        cost, accuracy = network.cost(testing_data[0], testing_data[1], include_accuracy=True)
        costs.append(cost)
        accuracies.append(accuracy)
        print('{0}. cost: {1}, accuracy: {2}%'.format(i + 1, cost, round(accuracy * 100, 2)))

    t += time.time()
    print('time taken for training: {0} seconds'.format(round(t, 2)))

    t = -time.time()
    validate(reader, network.weights)
    t += time.time()
    print('time taken for validation: {0} seconds'.format(round(t, 2)))

    t_main += time.time()
    print('total time taken: {0} seconds'.format(round(t_main, 2)))

    # plot costs
    plt.plot(list(range(len(costs))), costs)
    plt.show()

    save_to_disk = input('Save training stats to disk? <Y/N> :')

    if save_to_disk.upper() == 'Y':
        data = {'thetas': network.weights, 'time_taken_in_seconds': round(t_main),
                'metadata': metadata, 'costs': costs, 'accuracy': accuracies[-1]}
        path = write_dump(data)
        print('Saved stats from training to file: {}'.format(path))
    else:
        print('Closing script WITHOUT saving stats to disk...')
