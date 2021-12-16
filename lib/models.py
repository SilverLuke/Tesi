from tensorflow import keras
import lib.esn
from lib.initializers import *


class ESNInterface(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reservoir = None
        self.readout = None

    def compile(self, **kwargs):
        self.readout.compile(**kwargs)

    def call(self, inputs):
        reservoir_states = self.reservoir(inputs)
        output = self.readout(reservoir_states)
        return output

    def fit(self, x, y, **kwargs):
        # applies the reservoirs to all the input sequences in the training set
        x_train_out = self.reservoir(x)

        # does the same for the validation set
        x_val, y_val = kwargs['validation_data']
        x_val_out = self.reservoir(x_val)
        kwargs['validation_data'] = (x_val_out, y_val)

        # trains the readout with the reservoir states just computed
        return self.readout.fit(x_train_out, y, **kwargs)

    def evaluate(self, x, y, **kwargs):
        x_train_out = self.reservoir(x)
        return self.readout.evaluate(x_train_out, y, **kwargs)

    @property
    def units(self):
        return self.reservoir.layers[1].units


class ESN1(ESNInterface):
    def __init__(self,
                 units: int,
                 output_units: int,
                 output_activation,
                 input_scaling: float,
                 bias_scaling: float,
                 spectral_radius=0.9,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = Kernel(initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = RecurrentFullConnected(spectral_radius)
        bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, kernel_initializer=kernel_init, recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])


class ESN2(ESNInterface):
    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 output_units: int,
                 output_activation,
                 input_scaling,
                 bias_scaling,
                 connectivity=1.,
                 spectral_radius=0.9,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = SplitKernel(sub_reservoirs, initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = Type2(sub_reservoirs, connectivity, spectral_radius)
        bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, kernel_initializer=kernel_init, recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])


class ESN3(ESNInterface):
    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 output_units: int,
                 output_activation,
                 input_scaling,
                 bias_scaling,
                 connectivity=1.,
                 spectral_radius=0.9,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = SplitKernel(sub_reservoirs, initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = Type3(sub_reservoirs, connectivity, spectral_radius)
        bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, kernel_initializer=kernel_init, recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])


class ESN4(ESNInterface):
    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 partitions: TensorLike,
                 output_units: int,
                 output_activation,
                 input_scaling,
                 bias_scaling,
                 connectivity=1.,
                 spectral_radius=0.9,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = SplitKernel(sub_reservoirs, partitions=partitions, initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = Type4(sub_reservoirs, partitions, connectivity, spectral_radius)
        bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, kernel_initializer=kernel_init, recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])
