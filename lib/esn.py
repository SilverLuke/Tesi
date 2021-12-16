import tensorflow as tf
from tensorflow.python.types.core import TensorLike
"""
Terminology:
    sub_reservoirs:  Positive integer, In how many sub-reservoir must split the reservoir (int)
    spectral_radius:  Positive float or list of float, Spectral radius of sub-reservoir if float every sub-reservoir 
        have the same spectral radius if a list of float must have the same length of sub_reservoirs and give the 
        spectral radius for every sub-reservoir.
    partitions:  None or List of float, sum must be 1 and the length must be equal to sub_reservoirs. Splits the units 
        for every sub_reservoir. If None equal subdivision.
    reservoirs_connectivity: List or Matrix of connectivity if list the len must be equal to equal 
        sub_reservoir x sub_reservoirs.                       
"""


class ESN(tf.keras.layers.RNN):
    def __init__(
            self,
            units: TensorLike,
            leaky: TensorLike,
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            recurrent_initializer=tf.keras.initializers.GlorotUniform,
            bias_initializer=tf.keras.initializers.Zeros,
            **kwargs,
    ):
        cell = Reservoir(
            units=units,
            leaky=leaky,
            activation=activation,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
        )
        super().__init__(cell, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
            constants=None,
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def leaky(self):
        return self.cell.leaky

    @property
    def spectral_radius(self):
        return self.cell.initializer.spectral_radius

    @property
    def activation(self):
        return self.cell.activation


class Reservoir(tf.keras.layers.AbstractRNNCell):
    def __init__(
            self,
            units: int,
            leaky: float = 1,
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            recurrent_initializer=tf.keras.initializers.GlorotUniform,
            bias_initializer=tf.keras.initializers.Zeros,  # Change this to use Bias
            **kwargs,
    ):
        super().__init__(trainable=False, name="reservoir", **kwargs)

        self.units = units
        self.leaky = leaky
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self._state_size = units
        self._output_size = units

        self.use_bias = self.bias_initializer is not tf.keras.initializers.Zeros
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        self.built = False

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self.units],
            initializer=self.kernel_initializer,
            trainable=False,
            dtype=self.dtype,
            synchronization=tf.VariableSynchronization.NONE
        )

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.units, self.units],
            initializer=self.recurrent_initializer,
            trainable=False,
            dtype=self.dtype,
            synchronization=tf.VariableSynchronization.NONE
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                trainable=False,
                dtype=self.dtype,
                synchronization=tf.VariableSynchronization.NONE
            )

        self.built = True

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, states):
        in_matrix = tf.concat([inputs, states[0]], axis=1)
        weights_matrix = tf.concat([self.kernel, self.recurrent_kernel], axis=0)

        output = tf.linalg.matmul(in_matrix, weights_matrix)
        if self.use_bias:
            output = output + self.bias
        output = self.activation(output)
        output = (1 - self.leaky) * states[0] + self.leaky * output

        return output, output
