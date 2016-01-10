import zlib
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

import time

DUMP_DIRECTORY = 'dumps'


def write_dump(data):
    dump_file_name = data['metadata']['program'] + '-' + str(time.time()).replace('.', '-') + '.ml'
    dump_file_path = os.path.join(DUMP_DIRECTORY, dump_file_name)

    dump_data = zlib.compress(pickle.dumps(data))

    with open(dump_file_path, mode='wb') as f:
        f.write(dump_data)


def read_dump(filepath):
    with open(filepath, 'rb') as f:
        return pickle.loads(zlib.decompress(f.read()))

if __name__ == "__main__":
    filename = 'MNIST-handwritten-digits-1447047318-99682.ml'

    data = read_dump(os.path.join('dumps', filename))

    print(data['metadata'])
    print(data['time_taken_in_seconds'])

    x = np.arange(len(data.get('costs')))
    y = data.get('costs')

    plt.draw()
    plt.plot(x, y)
    plt.show()
