from lib.utility import *
from tensorflow.python.types.core import TensorLike

""" Kernel initializers """


class Kernel(tf.keras.initializers.Initializer):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform()):
        self.kernel_initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        return self.kernel_initializer(shape, dtype=dtype)


class SplitKernel(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs, initializer):
        self.sub_reservoirs = sub_reservoirs
        self.kernel_initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        local_units = equalize_units(shape, self.sub_reservoirs)

        kernels = []
        for units in local_units:
            kernel = self.kernel_initializer((units,), dtype=dtype)
            kernels.append(tf.linalg.LinearOperatorFullMatrix(kernel))

        return tf.linalg.LinearOperatorBlockDiag(kernels).to_dense()


""" Recurrent kernel initializers """


# ESN with connectivity variable
class RecurrentStandard(tf.keras.initializers.Initializer):
    def __init__(self, connectivity, spectral_radius, initializer):
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


# ESN fully connected Unipi
# INITIALIZER equal to RecurrentStandard with connectivity to 1 and recurrent_initializer to tf.keras.initializers.RandomUniform
class RecurrentFullConnected(tf.keras.initializers.Initializer):
    def __init__(self, spectral_radius):
        self.spectral_radius = spectral_radius

    def __call__(self, shape, dtype=None, **kwargs):
        value = (self.spectral_radius / np.sqrt(shape[0])) * (6. / np.sqrt(12))
        w = tf.random.uniform(shape, minval=-value, maxval=value, dtype=dtype)
        return w


# General implementation for every type of sub-reservoirs Model
class RecurrentKernel(tf.keras.initializers.Initializer):
    def __init__(self, sub_reservoirs: int,
                 reservoirs_connectivity: TensorLike,
                 spectral_radius: TensorLike,
                 recurrent_initializer,
                 partitions=None,
                 global_spectral_radius: bool = False,
                 ):

        self.sub_reservoirs = sub_reservoirs
        self.partitions = partitions
        self.spectral_radius = spectral_radius
        self.global_sr = global_spectral_radius
        self.rc = reservoirs_connectivity
        self.recurrent_initializer = recurrent_initializer

        if partitions is None:
            self.partitions = [1. / sub_reservoirs for _ in range(sub_reservoirs)]
        else:
            self.partitions = check_and_vectorize(partitions, float, sub_reservoirs, "Partitions")

    def __call__(self, shape, dtype=None, **kwargs):
        units = split_units(shape[0], self.partitions)
        recurrent_kernels = [[_ for _ in range(self.sub_reservoirs)] for _ in range(self.sub_reservoirs)]
        # Used to build the reservoir
        for i in range(self.sub_reservoirs):
            for j in range(self.sub_reservoirs):
                size = (units[i], units[j])
                connectivity = self.rc[i][j]
                spectral_radius = None
                if i == j and not self.global_sr:
                    spectral_radius = self.spectral_radius[i]
                w = generate_matrix(size, self.recurrent_initializer, spectral_radius, connectivity, dtype)
                recurrent_kernels[i][j] = w

        return join_matrices(recurrent_kernels)


class Type1(RecurrentKernel):
    def __init__(self, connectivity, spectral_radius, recurrent_initializer=tf.keras.initializers.GlorotUniform):
        reservoirs_connectivity = [[connectivity]]
        spectral_radius = [spectral_radius]
        super().__init__(1, reservoirs_connectivity, spectral_radius, recurrent_initializer)


class Type2(RecurrentKernel):
    def __init__(self, sub_reservoirs, connectivity, spectral_radius,
                 recurrent_initializer=tf.keras.initializers.GlorotUniform):
        reservoirs_connectivity = [[connectivity if i == j else 0 for i in range(sub_reservoirs)]
                                   for j in range(sub_reservoirs)]
        spectral_radius = [spectral_radius for _ in range(sub_reservoirs)]
        super().__init__(sub_reservoirs, reservoirs_connectivity, spectral_radius, recurrent_initializer)


class Type3(RecurrentKernel):
    def __init__(self, sub_reservoirs, reservoirs_connectivity, spectral_radius,
                 recurrent_initializer=tf.keras.initializers.GlorotUniform):
        spectral_radius = [spectral_radius for _ in range(sub_reservoirs)]
        super().__init__(sub_reservoirs, reservoirs_connectivity, spectral_radius, recurrent_initializer)
