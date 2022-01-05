import tensorflow as tf
import numpy as np
from tensorflow.python.types.core import TensorLike


# Split shape in a vector with proportions there are in part
# sum(vec) == shape only if sum(part) == 1 and len(vec) == len(part)
def split_units(shape, part):
    partitions = []
    used = 0
    for div in part:
        units = int(np.floor(shape * div))
        used += units
        partitions.append(units)
    rest = shape - used  # Rest should be less or equal to len(list)
    for i in range(rest):
        partitions[i] += 1
    assert shape == sum(partitions)
    return partitions


# Return a vector of units for each sub-reservoir
def get_spectral_radius(tensor):
    return tf.cast(tf.reduce_max(tf.abs(tf.linalg.eig(tensor)[0])), tf.float32)


def tf_generate_matrix(shape, spectral_radius):
    w = tf.random.uniform(shape=shape)
    sr = get_spectral_radius(w)
    w = (w / sr) * spectral_radius
    return w


# uses circular law to determine the values of the recurrent weight matrix
# rif. paper
# Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
# "Fast spectral radius initialization for recurrent neural networks."
# INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
def unipi_generate_matrix(shape, spectral_radius: float):
    value = ((spectral_radius / np.sqrt(shape[0])) * (6. / np.sqrt(12)))
    w = tf.random.uniform(shape, minval=-value, maxval=value)
    return w


# Spectral radius should be between 0 and 1 to maintain the echo state property
def generate_matrix(shape, initializer, spectral_radius, connectivity, dtype=None):
    if connectivity == 1.:
        if spectral_radius is not None:
            return unipi_generate_matrix(shape, spectral_radius)
        else:
            return tf.random.uniform(shape)
    elif connectivity == 0.:
        return tf.zeros(shape)
    else:
        # https://github.com/tensorflow/addons/blob/e83e71cf07f65773d0f3ba02b6de66ec3b190db7/tensorflow_addons/rnn/esn_cell.py
        matrix = initializer(shape, dtype=dtype)
        connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), connectivity), dtype)
        matrix = tf.math.multiply(matrix, connectivity_mask)
        if spectral_radius is not None:
            scaling = tf.math.divide_no_nan(spectral_radius, get_spectral_radius(matrix))
            matrix = tf.multiply(matrix, scaling)
        return matrix


# matrices is a python matrix of tf.matrix
def join_matrices(matrices):
    ret = tf.concat(matrices[0], axis=1)
    for r_k in matrices[1:]:
        tmp = tf.concat(r_k, axis=1)  # axis=1 horizontal concatenation
        ret = tf.concat([ret, tmp], axis=0)
    return ret


""" Kernel initializers """


class Kernel(tf.keras.initializers.Initializer):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        self.kernel_initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        return self.kernel_initializer(shape, dtype=dtype)


class SplitKernel(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs, partitions=None, initializer=tf.keras.initializers.GlorotUniform()):
        self.sub_reservoirs = sub_reservoirs
        self.initializer = initializer
        if partitions is None:
            self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]
        else:
            self.partitions = partitions  # check_and_vectorize(partitions, float, sub_reservoirs, "Partitions")

    def __call__(self, shape, dtype=None, **kwargs):
        local_units = split_units(shape[1], self.partitions)
        kernels = []
        for units in local_units:
            kernel = self.initializer((1, units), dtype=dtype)
            kernels.append(tf.linalg.LinearOperatorFullMatrix(kernel))

        ker = tf.linalg.LinearOperatorBlockDiag(kernels).to_dense()
        return ker


""" Recurrent kernel initializers """


# ESN with connectivity variable Not Used yet
class RecurrentStandard(tf.keras.initializers.Initializer):
    def __init__(self, connectivity, spectral_radius, initializer=tf.keras.initializers.GlorotUniform()):
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.recurrent_initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        recurrent_weights = self.recurrent_initializer(shape, dtype)

        connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), self.connectivity), dtype)
        recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)

        # Satisfy the necessary condition for the echo state property `max(eig(W)) < 1`
        abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
        scaling_factor = tf.math.divide_no_nan(
            self.spectral_radius, tf.reduce_max(abs_eig_values)
        )

        recurrent_weights = tf.multiply(recurrent_weights, scaling_factor)
        return recurrent_weights


# ESN fully connected
# RecurrentFullConnected equal to RecurrentStandard with connectivity to 1 and initializer to tf.keras.initializers.RandomUniform
class RecurrentFullConnected(tf.keras.initializers.Initializer):
    def __init__(self, spectral_radius):
        self.spectral_radius = spectral_radius

    def __call__(self, shape, dtype=None, **kwargs):
        value = (self.spectral_radius / np.sqrt(shape[0])) * (6. / np.sqrt(12))
        w = tf.random.uniform(shape, minval=-value, maxval=value, dtype=dtype)
        return w


# General implementation for every type of sub-reservoirs Model
# sub_reservoir: In how many sub reservoir split the kernel
# reservoirs_connectivity: square matrix of size sub_reservoir, indicate the connectivity of that part of space
# spectral_radius: vector of len sub_reservoir, is the sub_reservoir spectral radius
# initializer: generator of that part of space
class RecurrentKernel(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs: int,
                 reservoirs_connectivity: TensorLike,
                 spectral_radius: TensorLike,
                 initializer=tf.keras.initializers.GlorotUniform(),
                 partitions=None,
                 global_spectral_radius: bool = False,
                 ):
        self.global_sr = global_spectral_radius
        if self.global_sr:
            if isinstance(spectral_radius, list):
                raise ValueError("If global spectral radius is true, spectral radius must be a float not a list")

        if partitions is None:
            self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]
        else:
            self.partitions = partitions  # check_and_vectorize(partitions, float, sub_reservoirs, "Partitions")

        self.sub_reservoirs = sub_reservoirs
        self.rc = reservoirs_connectivity
        self.spectral_radius = spectral_radius
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                connectivity = self.rc[i][j]

                spectral_radius = None
                if not self.global_sr and i == j:
                    spectral_radius = self.spectral_radius[i]

                w = generate_matrix(size, self.initializer, spectral_radius, connectivity, dtype)
                recurrent_kernels[i][j] = w
        matrix = join_matrices(recurrent_kernels)
        if self.global_sr:  # Normalize the entire matrix
            scaling = tf.math.divide_no_nan(self.spectral_radius, get_spectral_radius(matrix))
            matrix = tf.multiply(matrix, scaling)
        return matrix


def check_spectral_radius(global_sr, spectral_radius, sub_reservoirs):
    reservoirs_sr = None
    if global_sr:
        if isinstance(spectral_radius, list):
            raise ValueError("When global spectral radius is True, spectral radius must a float")
        else:
            reservoirs_sr = tf.cast(spectral_radius, tf.float32)
    else:
        if isinstance(spectral_radius, float) or isinstance(spectral_radius, int):
            reservoirs_sr = [tf.cast(spectral_radius, tf.float32) for _ in range(sub_reservoirs)]
        elif isinstance(spectral_radius, list):
            if len(spectral_radius) == sub_reservoirs:
                reservoirs_sr = [tf.cast(spectral_radius[i], tf.float32) for i in range(sub_reservoirs)]
            else:
                raise ValueError("The list of spectral radius must have the same length of sub_reservoirs")
    return reservoirs_sr


class Type2(RecurrentKernel):
    def __init__(self, sub_reservoirs, connectivity, spectral_radius, global_sr: bool = False,
                 initializer=tf.keras.initializers.GlorotUniform()):

        reservoirs_connectivity = None
        if isinstance(connectivity, float):
            reservoirs_connectivity = [[connectivity if i == j else 0. for i in range(sub_reservoirs)]
                                       for j in range(sub_reservoirs)]
        elif isinstance(connectivity, list):
            reservoirs_connectivity = [[connectivity[i] if i == j else 0. for i in range(sub_reservoirs)]
                                       for j in range(sub_reservoirs)]

        reservoirs_sr = check_spectral_radius(global_sr, spectral_radius, sub_reservoirs)

        super().__init__(sub_reservoirs, reservoirs_connectivity, reservoirs_sr, global_spectral_radius=global_sr,
                         initializer=initializer)


class Type3(RecurrentKernel):
    def __init__(self, sub_reservoirs, reservoirs_connectivity, spectral_radius, global_sr: bool = False,
                 initializer=tf.keras.initializers.GlorotUniform()):
        reservoirs_sr = check_spectral_radius(global_sr, spectral_radius, sub_reservoirs)

        super().__init__(sub_reservoirs, reservoirs_connectivity, reservoirs_sr, global_spectral_radius=global_sr,
                         initializer=initializer)


class Type4(RecurrentKernel):
    def __init__(self, sub_reservoirs, partitions, reservoirs_connectivity, spectral_radius, global_sr: bool = False,
                 initializer=tf.keras.initializers.GlorotUniform()):

        reservoirs_sr = check_spectral_radius(global_sr, spectral_radius, sub_reservoirs)

        super().__init__(sub_reservoirs, reservoirs_connectivity, reservoirs_sr, global_spectral_radius=global_sr,
                         partitions=partitions, initializer=initializer)
