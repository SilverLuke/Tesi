import random
from typing import Optional

import matplotlib
import numpy as np
import tensorflow as tf
import sys
from time import *
import os

from matplotlib import pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
sys.path.insert(0, PROJECT_ROOT)

from IRESNs_tensorflow.initializers import *

PATIENCE = 5
EPOCHS = 200
OUTPUT_UNITS = 20
OUTPUT_ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'

PROJECT_NAME = "character trajectories"
DATA_DIR = os.path.join("datasets", PROJECT_NAME)
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_DIR)


def test_spectral_radius():
    x = []
    y = []
    for _ in range(0, 100):
        sr_real = random.uniform(1, 5)
        size = int(random.uniform(100, 500))
        tensor = generate_matrix((size, size), sr_real, 1)
        sr = get_spectral_radius(tensor)
        x.append(size)
        y.append(sr.numpy() - sr_real)

    print("###########")
    print("average size:", sum(x) / len(x), "\nAverage error:", sum(y) / len(y))
    print("Generate plot:")
    matplotlib.pyplot.scatter(x, y)
    matplotlib.pyplot.show()


def test_compositions_of_matrix():
    x = []
    y = []
    for _ in range(0, 1):
        sr_real = random.uniform(0, 1)
        size = int(random.uniform(100, 500))
        tensor = generate_matrix((size, size), sr_real, connectivity=0.5)
        sr = get_spectral_radius(tensor)
        x.append(size)
        y.append(sr.numpy() - sr_real)
    # print(sr, sr_real)

    print("###########")
    print("average size:", sum(x) / len(x), "\nAverage error:", sum(y) / len(y))


def test_connectivity():
    x = []
    y = []
    for _ in range(0, 1000):
        sr_real = random.uniform(0, 1)
        size = int(random.uniform(100, 500))
        tensor = unipi_generate_matrix((size, size), sr_real)
        zeros = (size * size) - tf.math.count_nonzero(tensor, dtype=float).numpy()
        x.append(size)
        y.append(zeros)

    print("###########")
    print("average size:", sum(x) / len(x), "\nAverage zeros:", sum(y) / len(y))


def test_splits():
    partitions = [random.uniform(0., 1.) for i in range(3)]
    total = sum(partitions)
    t = list(map(lambda _x: _x / total, partitions))
    print(partitions, total)
    print(t, sum(t))


def test_join_matrices():
    mat1 = np.matrix('1 2; 3 4')
    mat2 = np.matrix('5 6; 7 8')
    mat3 = np.matrix('9 10; 11 12')
    print(join_matrices([[mat1, mat2], [mat3, mat1]]))


def unipi_call(inputs, state, kernel, recurrent_kernel):
    input_part = tf.matmul(inputs, kernel)
    state_part = tf.matmul(state, recurrent_kernel)
    output = input_part + state_part
    return output


def tf_call(inputs, state, kernel, recurrent_kernel):
    in_matrix = tf.concat([inputs, state], axis=1)  # Concat horizontally  MAT.Shape [ input.y x input.x+states.x]
    weights_matrix = tf.concat([kernel, recurrent_kernel], axis=0)  # Concat vertically MAT
    output = tf.linalg.matmul(in_matrix, weights_matrix)
    return output


def test_calls():
    dtype = tf.float64
    min_shape = 100
    max_shape = 500

    x = tf.random.uniform((), minval=min_shape, maxval=max_shape, dtype=tf.int32)
    y = tf.random.uniform((), minval=min_shape, maxval=max_shape, dtype=tf.int32)

    minmax = 3
    inputs = tf.random.uniform((x, y), minval=-minmax, maxval=minmax, dtype=dtype)
    kernel = tf.random.uniform((y, x), minval=-minmax, maxval=minmax, dtype=dtype)
    state = tf.random.uniform((x, x), minval=-minmax, maxval=minmax, dtype=dtype)
    recurrent_kernel = tf.random.uniform((x, x), minval=-minmax, maxval=minmax, dtype=dtype)

    w_tf = tf_call(inputs, state, kernel, recurrent_kernel)
    w_unipi = unipi_call(inputs, state, kernel, recurrent_kernel)

    if w_tf.shape != w_unipi.shape:
        print("Le shape sono diverse")

    diff = (w_tf - w_unipi)
    max = 0.
    for i in diff:
        for j in i:
            val = abs(j)
            if (val > max):
                max = val

    n_zeri = tf.math.count_nonzero(diff).numpy()

    print("Valore piu grande:", max.numpy())
    print("Valori a zero:", n_zeri, " Quindi il ", (n_zeri / (diff.shape[0] * diff.shape[1])) * 100, "%")

    if diff != tf.zeros((x, y), dtype=dtype):
        print("Le matrici sono differenti. unipi_call != tf_call")
    else:
        print("Le matrici sono uguali. unipi_call == tf_call")


def benchmark_calls():
    min_v = 900
    max_v = 1000
    time_tf = []
    time_uni = []
    total_time = time()
    for i in range(10000):
        x = tf.random.uniform((), minval=min_v, maxval=max_v, dtype=tf.int32)
        y = tf.random.uniform((), minval=min_v, maxval=max_v, dtype=tf.int32)
        inputs = tf.random.uniform((x, y), minval=-1, maxval=1, dtype=tf.float32)
        kernel = tf.random.uniform((y, x), minval=-1, maxval=1, dtype=tf.float32)
        state = tf.random.uniform((x, x), minval=-1, maxval=1, dtype=tf.float32)
        recurrent_kernel = tf.random.uniform((x, x), minval=-1, maxval=1, dtype=tf.float32)

        start = time()
        _ = unipi_call(inputs, state, kernel, recurrent_kernel)
        time_uni.append(time() - start)

        start = time()
        _ = tf_call(inputs, state, kernel, recurrent_kernel)
        time_tf.append(time() - start)

    print("Totat run time:", time() - total_time)
    print("TF time: ", np.mean(time_tf), "??", np.std(time_tf))
    print("UNI time: ", np.mean(time_uni), "??", np.std(time_uni))
    if np.mean(time_tf) < np.mean(time_uni):
        print("Vince TF_call")
    else:
        print("Vince UNIPI_call")


def benchmark_bias():
    time_if = []
    time_plus = []
    w2 = None
    zero_init = tf.keras.initializers.Zeros()
    for _ in range(30):
        x = tf.random.uniform((), minval=10, maxval=100, dtype=tf.int32)
        y = tf.random.uniform((), minval=10, maxval=100, dtype=tf.int32)
        inputs = tf.random.uniform((x, y), minval=-1, maxval=1)
        bias = tf.random.uniform((x, y), minval=-1, maxval=1)
        start = time()
        if not isinstance(bias, zero_init):
            w2 = inputs + bias
        time_if.append(time() - start)
        start = time()
        w3 = inputs + bias
        time_plus.append(time() - start)
    print(w2)
    print(w3)
    print("With if   :", np.mean(time_if), "+_", np.std(time_if))
    print("Without if:", np.mean(time_plus), "+_", np.std(time_plus))


def benchmark_inits():
    time_uniform = []
    time_glorot = []

    x = 1000
    uniform_init = Kernel(initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1))
    glorot_init = Kernel()
    for _ in range(1000):
        start = time()
        w = uniform_init(shape=(x, x), dtype=tf.float32)
        time_uniform.append(time() - start)

        start = time()
        e = glorot_init(shape=(x, x), dtype=tf.float32)
        time_glorot.append(time() - start)
    print("Uniform initializer      : {:.5f} ?? {:.4f}s".format(np.mean(time_uniform), np.std(time_uniform)))
    print("GlorotUniform initializer: {:.5f} ?? {:.4f}s".format(np.mean(time_glorot), np.std(time_glorot)))


def normalize(mat, sr):
    scaling = tf.math.divide_no_nan(sr, get_spectral_radius(mat))
    return tf.multiply(mat, scaling)


def plot_matrix(title, matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix)
    ax.set_title(title)
    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = tf.cast(matrix[i, j], tf.float32)
            text = ax.text(j, i, "{:.5f}".format(val),
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()


def test_generation_matrix():
    size_x = 10
    size_y = 10
    srx = 6.
    sry = 4.
    gsr = 2.

    init = tf.keras.initializers.GlorotUniform()

    x_mat = init((size_x, size_x))
    y_mat = init((size_y, size_y))
    xy_zero = tf.keras.initializers.Zeros()((size_x, size_y))

    x_norm = normalize(x_mat, srx)
    y_norm = normalize(y_mat, sry)
    xy_norm = join_matrices([[x_norm, xy_zero], [tf.transpose(xy_zero), y_norm]])
    xy_norm2 = normalize(xy_norm, gsr)

    xy_mat = join_matrices([[x_mat, xy_zero], [tf.transpose(xy_zero), y_mat]])
    mat_norm = normalize(xy_mat, gsr)

    diff = xy_norm2 - mat_norm

    print("SR xy_norm", get_spectral_radius(xy_norm).numpy())
    print("SR xy_norm2", get_spectral_radius(xy_norm2).numpy())
    print("")
    print("SR xy_mat", get_spectral_radius(xy_mat).numpy())
    print("SR mat_norm", get_spectral_radius(mat_norm).numpy())
    print("")
    print("SR diff", get_spectral_radius(diff).numpy())

    print("Are equals: ", (tf.math.count_nonzero(diff) == 0).numpy())

    plot_matrix("xy_norm", xy_norm)
    plot_matrix("xy_norm2", xy_norm2)

    plot_matrix("mat_norm", mat_norm)

    plot_matrix("DIFF", xy_norm2 - mat_norm)


if __name__ == '__main__':
    tf.random.set_seed(42)
    # benchmark_calls()
    reservoirs = 3
    test = [[(i, j)
             for i in range(reservoirs)]
            for j in range(reservoirs)]
    for i in test:
        for j in i:
            print(j, end=" ")
        print()

    print(test)