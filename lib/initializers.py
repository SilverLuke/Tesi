from typing import Optional

import tensorflow as tf
import numpy as np
from tensorflow.python.types.core import TensorLike


# Split shape in a vector with proportions there are in part
# sum(vec) == shape only if sum(part) == 1 and len(vec) == len(part)
def split_units(shape, absolute_partition):
    partitions = []
    used = 0
    for div in absolute_partition:
        units = int(np.floor(shape * div))
        used += units
        partitions.append(units)
    rest = shape - used  # Rest should be less or equal to len(list)
    assert rest < len(absolute_partition)
    for i in range(rest):
        partitions[i] += 1
    assert shape == sum(partitions)
    return partitions


def get_spectral_radius(tensor, dtype):
    return tf.cast(tf.reduce_max(tf.abs(tf.linalg.eigvals(tensor))), dtype)


def tf_generate_matrix(shape, spectral_radius):
    w = tf.random.uniform(shape=shape)
    sr = get_spectral_radius(w)
    w = (w / sr) * spectral_radius
    return w


def generate_matrix(shape, initializer, spectral_radius, connectivity, dtype=None):
    if connectivity == 1.:
        matrix = FullConnected(spectral_radius)(shape, dtype=dtype)
    elif connectivity == 0.:
        matrix = tf.zeros(shape, dtype=dtype)
    else:
        # https://github.com/tensorflow/addons/blob/e83e71cf07f65773d0f3ba02b6de66ec3b190db7/tensorflow_addons/rnn/esn_cell.py
        matrix = initializer(shape, dtype=dtype)
        connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(shape), connectivity), dtype)
        matrix = tf.math.multiply(matrix, connectivity_mask)
        if spectral_radius is not None:
            scaling = tf.math.divide_no_nan(spectral_radius, get_spectral_radius(matrix, dtype))
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
                raise ValueError("Lenght of Bias scaling must be equal to sub_reservoirs. {} != {}"
                                 .format(len(bias_scaling), sub_reservoirs))
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
    def __init__(self, sub_reservoirs, input_scaling, partitions=None):
        if isinstance(input_scaling, list):
            if len(input_scaling) != sub_reservoirs:
                raise ValueError("Length of input_scaling must be equal to sub_reservoirs. {} != {}".
                                 format(len(input_scaling), sub_reservoirs))
            self.initializers = [tf.keras.initializers.RandomUniform(minval=-input_scaling[i], maxval=input_scaling[i]) for i in
                    range(sub_reservoirs)]
        elif isinstance(input_scaling, float) or isinstance(input_scaling, int):
            self.initializers = [tf.keras.initializers.RandomUniform(minval=-input_scaling, maxval=input_scaling) for _ in
                    range(sub_reservoirs)]
        elif input_scaling is None:
            self.initializers = [tf.keras.initializers.GlorotUniform() for _ in sub_reservoirs]
        else:
            raise ValueError("Input scaling should be a int / float or list of int/float")

        if partitions is None:
            self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]
        else:
            self.partitions = partitions

    def __call__(self, shape, dtype=None, **kwargs):
        local_units = split_units(shape[1], self.partitions)
        kernels = []
        for i, units in enumerate(local_units):
            kernel = self.initializers[i]((1, units), dtype=dtype)
            kernels.append(tf.linalg.LinearOperatorFullMatrix(kernel))

        ker = tf.linalg.LinearOperatorBlockDiag(kernels).to_dense()
        return ker


""" Recurrent kernel initializers """


# ESN with variable connectivity
class RecurrentKernel(tf.keras.initializers.Initializer):
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
# uses circular law to determine the values of the recurrent weight matrix
# rif. paper
# Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli.
# "Fast spectral radius initialization for recurrent neural networks."
# INNS Big Data and Deep Learning conference. Springer, Cham, 2019.
class FullConnected(tf.keras.initializers.Initializer):
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
class RecurrentKernel_(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs: int,
                 reservoirs_connectivity: TensorLike,
                 spectral_radius: TensorLike,
                 gsr: Optional[float] = None,
                 off_diagonal=None,
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
        self.diagonal_init = initializer
        self.off_diagonal = off_diagonal

    def __call__(self, shape, dtype=None, **kwargs):
        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                connectivity = self.rc[i][j]
                if i == j:  # if the matrix is on the diagonal
                    spectral_radius = self.spectral_radius[i]
                    recurrent_kernels[i][j] = generate_matrix(size, self.diagonal_init, spectral_radius, connectivity, dtype)
                else:
                    if self.off_diagonal is not None:  # the matrix is off-diagonal and it is a non zero values
                        mimmax = self.off_diagonal[i][j]
                        init = tf.keras.initializers.RandomUniform(minval=-abs(mimmax), maxval=abs(mimmax))
                        recurrent_kernels[i][j] = generate_matrix(size, init, None, connectivity, dtype)
                    else:
                        recurrent_kernels[i][j] = tf.zeros(size, dtype=dtype)
        matrix = join_matrices(recurrent_kernels)
        if self.gsr is not None:  # Normalize the entire matrix
            scaling = tf.math.divide_no_nan(self.gsr, get_spectral_radius(matrix, dtype))
            matrix = tf.multiply(matrix, scaling)
        return matrix


class Type2(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs, connectivity, spectral_radius, gsr: Optional[float] = None,
                 initializer=tf.keras.initializers.GlorotUniform()):

        if isinstance(connectivity, float):
            self.connectivity = [connectivity for _ in range(sub_reservoirs)]
        elif isinstance(connectivity, list):
            if len(connectivity) == sub_reservoirs:
                self.connectivity = [connectivity[i] for i in range(sub_reservoirs)]
            else:
                raise ValueError("The inter-connectivity list must have the same length of sub_reservoirs")
        else:
            raise ValueError("Wrong value type for connectivity. Given {}".format(type(connectivity)))

        if isinstance(spectral_radius, float) or isinstance(spectral_radius, int):
            self.spectral_radius = [spectral_radius for _ in range(sub_reservoirs)]
        elif isinstance(spectral_radius, list):
            if len(spectral_radius) == sub_reservoirs:
                self.spectral_radius = [spectral_radius[i] for i in range(sub_reservoirs)]
            else:
                raise ValueError("The list of spectral radius must have the same length of sub_reservoirs")
        else:
            raise ValueError("Wrong value type for spectral radius. Given {}".format(type(spectral_radius)))
        self.sub_reservoirs = sub_reservoirs
        self.gsr = gsr
        self.initializer = initializer
        self.partitions = [1./sub_reservoirs for _ in range(sub_reservoirs)]

    def __call__(self, shape, dtype=None, **kwargs):
        self.connectivity = [tf.cast(connectivity, dtype) for connectivity in self.connectivity]
        self.spectral_radius = [tf.cast(spectral_radius, dtype) for spectral_radius in self.spectral_radius]

        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                if i == j:  # if the matrix is on the diagonal
                    connectivity = self.connectivity[i]
                    spectral_radius = self.spectral_radius[i]
                    recurrent_kernels[i][j] = generate_matrix(size, self.initializer, spectral_radius, connectivity, dtype)
                else:
                    recurrent_kernels[i][j] = tf.zeros(size, dtype=dtype)
        matrix = join_matrices(recurrent_kernels)
        if self.gsr is not None:  # Normalize the entire matrix
            scaling = tf.math.divide_no_nan(tf.cast(self.gsr, dtype), get_spectral_radius(matrix, dtype))
            matrix = tf.multiply(matrix, scaling)
        return matrix


class Type3(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs, reservoirs_connectivity, spectral_radius, off_diagonal, gsr: Optional[float] = None,
                 initializer=tf.keras.initializers.GlorotUniform()):
        self.sub_reservoirs = sub_reservoirs
        self.rc = reservoirs_connectivity
        self.spectral_radius = spectral_radius
        self.off_diagonal = off_diagonal
        self.gsr = gsr
        self.initializer = initializer
        self.partitions = [1./sub_reservoirs for _ in range(sub_reservoirs)]

    def __call__(self, shape, dtype=None, **kwargs):
        self.rc = [[tf.cast(connectivity, dtype) for connectivity in row] for row in self.rc]
        self.spectral_radius = [tf.cast(spectral_radius, dtype) for spectral_radius in self.spectral_radius]

        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                connectivity = self.rc[i][j]
                if i == j:  # if the matrix is on the diagonal
                    spectral_radius = self.spectral_radius[i]
                    recurrent_kernels[i][j] = generate_matrix(size, self.initializer, spectral_radius, connectivity, dtype)
                else:
                    minmax = self.off_diagonal[i][j]
                    init = tf.keras.initializers.RandomUniform(minval=-abs(minmax), maxval=abs(minmax))
                    matrix = init(size, dtype=dtype)
                    connectivity_mask = tf.cast(tf.math.less_equal(tf.random.uniform(size), connectivity), dtype)
                    recurrent_kernels[i][j] = tf.math.multiply(matrix, connectivity_mask)
        matrix = join_matrices(recurrent_kernels)
        if self.gsr is not None:  # Normalize the entire matrix
            scaling = tf.math.divide_no_nan(self.gsr, get_spectral_radius(matrix, dtype))
            matrix = tf.multiply(matrix, scaling)
        return matrix


class Type4(Type3):
    def __init__(self, sub_reservoirs, partitions, reservoirs_connectivity, spectral_radius, off_diagonal,
                 gsr: Optional[float] = None, initializer=tf.keras.initializers.GlorotUniform()):
        super().__init__(sub_reservoirs, reservoirs_connectivity, spectral_radius, off_diagonal, gsr, initializer)
        self.partitions = partitions


