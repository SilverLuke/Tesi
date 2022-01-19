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
        matrix = unipi_generate_matrix(shape, spectral_radius)
    elif connectivity == 0.:
        matrix = tf.zeros(shape)
    else:
        # https://github.com/tensorflow/addons/blob/e83e71cf07f65773d0f3ba02b6de66ec3b190db7/tensorflow_addons/rnn/esn_cell.py
        matrix = initializer(shape, dtype=dtype)
        connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), connectivity), dtype)
        matrix = tf.math.multiply(matrix, connectivity_mask)
        print(type(spectral_radius), spectral_radius)
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


# Join N different matrices
class SplitBias(tf.keras.initializers.Initializer):
    def __init__(self, bias_scaling, sub_reservoirs):
        self.minmax = []
        if isinstance(bias_scaling, list):
            if len(bias_scaling) != sub_reservoirs:
                raise ValueError(
                    "Lenght of Bias scaling must be equal to sub_reservoirs. {} != {}".format(len(bias_scaling),
                                                                                              sub_reservoirs))
            self.minmax = bias_scaling
        elif isinstance(bias_scaling, float) or isinstance(bias_scaling, int):
            self.minmax = [bias_scaling for _ in range(sub_reservoirs)]
        elif bias_scaling is None:
            self.minmax = [0. for _ in range(sub_reservoirs)]
        else:
            raise ValueError("Bias scaling should be a int / float or list of int/float")

    def __call__(self, shape, dtype=None, **kwargs):
        part = [1. / len(self.minmax) for _ in range(len(self.minmax))]
        local_units = split_units(shape[0], part)
        pieces = []
        for i, units in enumerate(local_units):
            init = tf.keras.initializers.RandomUniform(minval=-self.minmax[i], maxval=self.minmax[i])
            piece = init((units,), dtype=dtype)
            pieces.append(piece)

        join = tf.concat(pieces, axis=0)
        return join


""" Kernel initializers """


class Kernel(tf.keras.initializers.Initializer):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        self.kernel_initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        return self.kernel_initializer(shape, dtype=dtype)


class SplitKernel(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs, partitions=None, initializers=None):
        self.sub_reservoirs = sub_reservoirs

        if partitions is None:
            self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]
        else:
            self.partitions = partitions

        if initializers is None:
            self.initializers = [tf.keras.initializers.GlorotUniform() for _ in sub_reservoirs]
        elif isinstance(initializers, list):
            if len(initializers) != sub_reservoirs:
                raise ValueError("Length of input_scaling must be equal to sub_reservoirs. {} != {}"
                                 .format(len(initializers), sub_reservoirs))
            self.initializers = initializers
        elif isinstance(initializers, tf.keras.initializers.Initializer):
            self.initializers = [initializers for _ in sub_reservoirs]
        else:
            raise ValueError("Initializers should be None, a single Initializer or a list of Initializer. Given: {}"
                             .format(type(initializers)))

    def __call__(self, shape, dtype=None, **kwargs):
        local_units = split_units(shape[1], self.partitions)
        kernels = []
        for i, units in enumerate(local_units):
            kernel = self.initializers[i]((1, units), dtype=dtype)
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
                 gsr: float,
                 initializer=tf.keras.initializers.GlorotUniform(),
                 partitions=None,
                 ):
        self.gsr = gsr

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
                spectral_radius = self.spectral_radius[i][j]
                recurrent_kernels[i][j] = generate_matrix(size, self.initializer, spectral_radius, connectivity, dtype)
        matrix = join_matrices(recurrent_kernels)
        if self.gsr is not None:  # Normalize the entire matrix
            scaling = tf.math.divide_no_nan(self.gsr, get_spectral_radius(matrix))
            matrix = tf.multiply(matrix, scaling)
        return matrix


class Type2(RecurrentKernel):
    def __init__(self, sub_reservoirs, connectivity, spectral_radius, gsr=False,
                 initializer=tf.keras.initializers.GlorotUniform()):

        if isinstance(connectivity, float):
            reservoirs_connectivity = [[connectivity if i == j else 0. for i in range(sub_reservoirs)]
                                       for j in range(sub_reservoirs)]
        elif isinstance(connectivity, list):
            if len(connectivity) == sub_reservoirs:
                reservoirs_connectivity = [[connectivity[i] if i == j else 0. for i in range(sub_reservoirs)]
                                           for j in range(sub_reservoirs)]
            else:
                raise ValueError("The inter-connectivity list must have the same length of sub_reservoirs")
        else:
            raise ValueError("Wrong value type for connectivity. Given {}".format(type(connectivity)))

        if isinstance(spectral_radius, float) or isinstance(spectral_radius, int):
            reservoirs_sr = [[tf.cast(spectral_radius, tf.float32) if i == j else 0. for i in range(sub_reservoirs)]
                             for j in range(sub_reservoirs)]
        elif isinstance(spectral_radius, list):
            if len(spectral_radius) != sub_reservoirs:
                raise ValueError("The list of spectral radius must have the same length of sub_reservoirs")
            elif isinstance(spectral_radius[0], float) or isinstance(spectral_radius[0], int):
                reservoirs_sr = [[tf.cast(spectral_radius[i], tf.float32) if i == j else 0.
                                  for i in range(sub_reservoirs)]
                                 for j in range(sub_reservoirs)]
            else:
                raise ValueError("Type error spectral radius for ESN2")
        else:
            raise ValueError("Wrong value type for spectral radius. Given {}".format(type(spectral_radius)))

        super().__init__(sub_reservoirs, reservoirs_connectivity, reservoirs_sr, gsr, initializer=initializer)


class Type3(RecurrentKernel):
    def __init__(self, sub_reservoirs, reservoirs_connectivity, spectral_radius, gsr=False,
                 initializer=tf.keras.initializers.GlorotUniform()):

        super().__init__(sub_reservoirs, reservoirs_connectivity, spectral_radius, gsr,
                         initializer=initializer)


class Type4(RecurrentKernel):
    def __init__(self, sub_reservoirs, partitions, reservoirs_connectivity, spectral_radius, gsr=False,
                 initializer=tf.keras.initializers.GlorotUniform()):

        super().__init__(sub_reservoirs, reservoirs_connectivity, spectral_radius, gsr,
                         partitions=partitions, initializer=initializer)
