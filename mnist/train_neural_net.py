import copy
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

        print 'predicted-\n', p_digits
        print 'actual-\n', y_digits
        print('accuracy-{0}%'.format(round(accuracy * 100, 2)))
        return None     # Let's just run only one loop for the time being


class NeuralNetLearningDiagnostics:
    def __init__(self, reader, metadata):
        self.reader = reader
        self.nn_metadata = metadata

    def run_training(self, metadata, predict_test_data=False):
        network = NeuralNet(layers=metadata['layers'],
                            lmda=metadata['lambda'])

        generator = self.reader.get(metadata['batch_size'], 'training')
        for i in range(metadata['epochs']):
            try:
                x, y = next(generator)
            except StopIteration:
                return network

            network.train(x, y, iterations=metadata['iterations'],
                          learning_rate=metadata['learning_rate'])

        return network

    def diagnose_batch_size(self, batch_sizes):
        costs = {'test': [], 'cv': []}
        accuracies = {'test': [], 'cv': []}

        test_data = next(self.reader.get(10000, mode='test'))
        cv_data = next(self.reader.get(10000, mode='validation'))

        for batch_size in batch_sizes:
            print 'testing for batch_size', batch_size
            metadata = copy.deepcopy(self.nn_metadata)
            metadata['batch_size'] = batch_size
            metadata['epochs'] = 300

            network = self.run_training(metadata)

            test_cost, test_accuracy = network.cost(test_data[0], test_data[1],
                                                    include_accuracy=True)
            cv_cost, cv_accuracy = network.cost(cv_data[0], cv_data[1],
                                                include_accuracy=True)
            costs['test'].append(test_cost)
            costs['cv'].append(cv_cost)
            accuracies['test'].append(test_accuracy)
            accuracies['cv'].append(cv_accuracy)

        print(costs, accuracies)

    def diagnose_layers(self):
        pass

    def diagnose_learning_rate(self):
        pass

    def diagnose_epochs(self):
        pass

    def diagnose_lambda(self):
        pass

    # def save_to_disk_dialog(self):
    #     save_to_disk = input('Save training stats to disk? <Y/N> :')
    #
    #     if save_to_disk.upper() == 'Y':
    #         data = {'thetas': network.weights, 'time_taken_in_seconds': round(t_main),
    #                 'metadata': metadata, 'costs': costs, 'accuracy': accuracies[-1]}
    #         path = write_dump(data)
    #         print('Saved stats from training to file: {}'.format(path))
    #     else:
    #         print('Closing script WITHOUT saving stats to disk...')

if __name__ == "__main__":
    initial_metadata = {
        'batch_size': 10,
        'layers': [784, 30, 10],
        'mode': 'training',
        'data_file': 'mnist.pkl.gz',
        'learning_rate': 2,
        'lambda': 0.0001,
        'iterations': 1,
        'epochs': 10,
        'program': 'MNIST-HWD-NN-2'     # MNIST handwritten digits neural-net version 1
    }

    reader = MNISTReader(initial_metadata['data_file'])
    reader.load()

    diagnostics = NeuralNetLearningDiagnostics(reader, initial_metadata)
    diagnostics.diagnose_batch_size([10, 30, 100])
