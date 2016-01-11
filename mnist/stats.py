__author__ = 'Alok'

import zlib
import pickle
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time

DUMP_DIRECTORY = 'dumps'


def write_dump(data):
    accuracy = round(data.get('accuracy') * 100, 2)
    t = str(time.time()).replace('.', '-')
    ext = 'aml'

    dump_file_name = '{0}-{1}-{2}.{3}'.format(data['metadata']['program'], accuracy, t, ext)
    dump_file_path = os.path.join(DUMP_DIRECTORY, dump_file_name)

    dump_data = zlib.compress(pickle.dumps(data))

    with open(dump_file_path, mode='wb') as f:
        f.write(dump_data)

    return dump_file_path


def read_dump(filepath):
    with open(filepath, 'rb') as f:
        return pickle.loads(zlib.decompress(f.read()))


def visualize_data(x, width=None):
    """
    Created an image from a matrix
    This piece of code is shamelessly ripped off from Andrew Ng's Octave version of the same function
    :param x:
    :param width:
    :return:
    """
    # Compute rows, cols
    m, n = x.shape

    # Set width automatically if not passed in
    example_width = width or round(math.sqrt(n))

    example_height = (n / example_width)

    # Compute number of items to display
    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)

    # Padding between images
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break

            # Copy patch
            max_val = max(abs(x[curr_ex, :]))
            offset_rows = pad + (j - 1) * (example_height + pad)
            offset_cols = pad + (i - 1) * (example_width + pad)
            patch = np.reshape(x[curr_ex, :], (example_height, example_width)) / max_val
            for p in range(int(example_height)):
                for q in range(int(example_width)):
                    display_array[p + offset_rows][q + offset_cols] = patch[p][q]

            curr_ex += 1
        if curr_ex > m:
            break

    # Display Image
    plt.imshow(display_array, cmap=cm.Greys_r)
    plt.show()

if __name__ == "__main__":
    filename = 'MNIST-HWD-NN-1-95.0-1447204002-728559.aml'

    data = read_dump(os.path.join(DUMP_DIRECTORY, filename))

    print(data['metadata'])
    print(data['time_taken_in_seconds'])

    x = np.arange(len(data.get('costs')))
    y = data.get('costs')

    plt.draw()
    plt.plot(x, y)
    plt.show()

    visualize_data(data['thetas'][0][:, 1:])








