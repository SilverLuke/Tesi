import random
import matplotlib
import numpy as np
import tensorflow as tf
import sys
from time import *
import os

PROJECT_ROOT = os.path.abspath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
sys.path.insert(0, PROJECT_ROOT)

from lib.initializers import *

PATIENCE = 5
EPOCHS = 200
OUTPUT_UNITS = 20
OUTPUT_ACTIVATION = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'

PROJECT_NAME = "character trajectories"
DATA_DIR = os.path.join("data", PROJECT_NAME)
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_DIR)


def test_spectral_radius():
    x = []
    y = []
    for _ in range(0, 100):
        sr_real = random.uniform(1, 5)
        size = int(random.uniform(100, 500))
        tensor = generate_matrix((size, size), sr_real, connectivity=1)
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


def test_equalize_units():
    split = equalize_units(6, 3)
    print(split)


def test_splits():
    units = 302
    parts = [1. / 3., 1. / 3., 1. / 3.]
    splits = split_units(units, parts)
    print(splits)


def test_join_matrices():
    mat1 = np.matrix('1 2; 3 4')
    mat2 = np.matrix('5 6; 7 8')
    mat3 = np.matrix('9 10; 11 12')
    print(join_matrices([[mat1, mat2], [mat3, mat1]]))


def test_activations():
    def tf_val(inputs, state, kernel, recurrent_kernel):
        in_matrix = tf.concat([inputs, state], axis=1)  # Concat horizontally  MAT.Shape [ input.y x input.x+states.x]
        weights_matrix = tf.concat([kernel, recurrent_kernel], axis=0)  # Concat vertically MAT
        output = tf.linalg.matmul(in_matrix, weights_matrix)
        return output

    def unipi_val(inputs, state, kernel, recurrent_kernel):
        input_part = tf.matmul(inputs, kernel)
        state_part = tf.matmul(state, recurrent_kernel)
        output = input_part + state_part
        return output

    time_tf = []
    time_uni = []
    w1 = w2 = None
    for _ in range(30):
        x = tf.random.uniform((), minval=10, maxval=10000, dtype=tf.int32)
        y = tf.random.uniform((), minval=10, maxval=10000, dtype=tf.int32)
        inputs = tf.random.uniform((x, y), minval=-1, maxval=1)
        kernel = tf.random.uniform((y, x), minval=-1, maxval=1)
        state = tf.random.uniform((x, x), minval=-1, maxval=1)
        recurrent_kernel = tf.random.uniform((x, x), minval=-1, maxval=1)

        start = time()
        w1 = tf_val(inputs, state, kernel, recurrent_kernel)
        time_tf.append(time() - start)

        start = time()
        w2 = unipi_val(inputs, state, kernel, recurrent_kernel)
        time_uni.append(time() - start)
    print(w1, w2)
    print(np.mean(time_tf), "+_", np.std(time_tf))
    print(np.mean(time_uni), "+_", np.std(time_uni))


def benchmark_bias():
    time_uni = []
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
        time_uni.append(time() - start)
    print(w2)
    print(np.mean(time_uni), "+_", np.std(time_uni))


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
    print("Uniform initializer      : {:.5f} ± {:.4f}s".format(np.mean(time_uniform), np.std(time_uniform)))
    print("GlorotUniform initializer: {:.5f} ± {:.4f}s".format(np.mean(time_glorot), np.std(time_glorot)))


if __name__ == '__main__':
    benchmark_inits()
# test_inits()
# test_activations()
# test_join_matrices()
# test_spectral_radius()
# test_equalize_units()
# test_MESNI()
