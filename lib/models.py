from typing import Optional

from matplotlib.ticker import MultipleLocator
from tensorflow import keras
import lib.esn
from lib.initializers import *

# For plotting the model
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
from lib.benchmarks import lower_and_replace


class ESNInterface(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = False
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

    def plot(self, names, title=None, bias=False, path=None, x_size=10, show=True):
        dataset_name, class_name, experiment_name, model_name = names
        if not self.build:
            raise Exception("Train the model first")

        kernel_m = self.reservoir.layers[1].cell.kernel
        rec_kernel_m = self.reservoir.layers[1].cell.recurrent_kernel
        readout_m = self.readout.layers[0].weights[0]

        height_ratios = [
            kernel_m.shape[0],
            rec_kernel_m.shape[0],
            rec_kernel_m.shape[0] / 20
        ]
        width_ratios = [
            rec_kernel_m.shape[1],
            readout_m.shape[1],
        ]
        bias_m = None
        if self.use_bias and bias:
            bias_m = self.reservoir.layers[1].cell.bias
            width_ratios.append(1)

        units_per_inch = sum(width_ratios) / x_size
        x = x_size
        y = np.floor(sum(height_ratios) / units_per_inch) + 2

        max_val = max(
            tf.reduce_max(tf.abs(rec_kernel_m)).numpy(),
            tf.reduce_max(tf.abs(kernel_m)).numpy(),
            tf.reduce_max(tf.abs(readout_m)).numpy()
        )

        fig = plt.figure(figsize=(x, y), dpi=500)

        if title is None:
            fig.suptitle(model_name + ' ' + experiment_name, fontsize=20, fontweight='bold')
        else:
            fig.suptitle(title, fontsize=20, fontweight='bold')
        gs = gridspec.GridSpec(nrows=len(height_ratios), ncols=len(width_ratios), figure=fig,
                               height_ratios=height_ratios, width_ratios=width_ratios,
                               left=0.05, right=0.95, bottom=0.05, top=0.92)

        rec_kernel = fig.add_subplot(gs[1, 0])
        kernel = fig.add_subplot(gs[0, 0], sharex=rec_kernel)
        readout = fig.add_subplot(gs[1, 1], sharey=rec_kernel)
        bar = fig.add_subplot(gs[2, 0])

        cmap = mpl.cm.get_cmap("RdBu").copy()
        norm = mpl.colors.SymLogNorm(0.001, vmin=-max_val, vmax=max_val)
        rec_kernel.imshow(rec_kernel_m, cmap=cmap, norm=norm, aspect=1, resample=False, interpolation=None)
        kernel.imshow(kernel_m, cmap=cmap, norm=norm, aspect=1, resample=False, interpolation=None)
        readout.imshow(readout_m, cmap=cmap, norm=norm, aspect=1, resample=False, interpolation=None)

        if self.use_bias and bias:
            bias = fig.add_subplot(gs[1, 2], sharey=rec_kernel)
            bias.imshow(np.asmatrix(bias_m).transpose(), cmap=cmap, norm=norm, aspect=1, resample=False,
                        interpolation=None)
            bias.set_title("Bias")
            bias.axis('tight')
            bias.get_yaxis().set_visible(False)
            bias.yaxis.set_minor_locator(MultipleLocator(10))
            bias.set_xticks([int(0), int(1)])

        rec_kernel.set_title("Recurrent kernel")
        kernel.set_title("Kernel")
        readout.set_title("Readout")

        rec_kernel.axis('tight')
        kernel.axis('tight')
        readout.axis('tight')

        kernel.get_xaxis().set_visible(False)
        kernel.set_yticks([int(0), int(kernel_m.shape[0] - 1)])
        kernel.yaxis.set_minor_locator(MultipleLocator(10))
        kernel.xaxis.set_minor_locator(MultipleLocator(10))

        rec_kernel.set_ylabel('Units', rotation=90)
        ticks = np.append(rec_kernel.get_xticks()[1:-1], rec_kernel_m.shape[1] - 1)
        rec_kernel.set_xticks(ticks)
        rec_kernel.set_yticks(ticks)
        rec_kernel.xaxis.set_minor_locator(MultipleLocator(10))
        rec_kernel.yaxis.set_minor_locator(MultipleLocator(10))

        readout.get_yaxis().set_visible(False)
        readout.set_xticks([int(0), int(readout_m.shape[1] - 1)])
        readout.xaxis.set_minor_locator(MultipleLocator(10))
        readout.yaxis.set_minor_locator(MultipleLocator(10))

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation='horizontal', label='Weights')

        if path is not None:
            path = os.path.join(path, dataset_name, lower_and_replace(class_name))
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path, lower_and_replace(experiment_name + ' ' + model_name) + ".svg"),
                        format='SVG', interpolation='none')
        if show:
            fig.show()
        plt.close(fig)


class ESN1(ESNInterface):
    def __init__(self,
                 units: int,
                 output_units: int,
                 output_activation,
                 activation=tf.nn.tanh,
                 spectral_radius: float = 0.9,
                 connectivity: float = 1.,
                 input_scaling: float = 1.,
                 bias_scaling=None,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = Kernel(initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )

        if connectivity == 1.0:
            recurrent_kernel_init = RecurrentFullConnected(spectral_radius)
        else:
            recurrent_kernel_init = RecurrentStandard(connectivity, spectral_radius)

        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)
        else:
            bias_init = None

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, activation=activation,
                        kernel_initializer=kernel_init,
                        recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])


def generate_split_kernel_inits(input_scaling, sub_reservoirs):
    if isinstance(input_scaling, list):
        if len(input_scaling) != sub_reservoirs:
            raise ValueError("Length of input_scaling must be equal to sub_reservoirs. {} != {}".
                             format(len(input_scaling), sub_reservoirs))
        init = [tf.keras.initializers.RandomUniform(minval=-input_scaling[i], maxval=input_scaling[i]) for i in
                range(sub_reservoirs)]
    elif isinstance(input_scaling, float) or isinstance(input_scaling, int):
        init = [tf.keras.initializers.RandomUniform(minval=-input_scaling, maxval=input_scaling) for _ in
                range(sub_reservoirs)]
    elif input_scaling is None:
        init = None
    else:
        raise ValueError("Input scaling should be a int / float or list of int/float")
    return init


class ESN2(ESNInterface):
    def __init__(self,
                 units: int,
                 sub_reservoirs: int,
                 output_units: int,
                 output_activation,
                 input_scaling,  # This can be a vec or int / float
                 bias_scaling=None,  # This can be a vec or int / float or None
                 gsr: Optional[float] = None,
                 spectral_radius=0.9,
                 connectivity=1.,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        init = generate_split_kernel_inits(input_scaling, sub_reservoirs)
        kernel_init = SplitKernel(sub_reservoirs, initializers=init)
        recurrent_kernel_init = Type2(sub_reservoirs, connectivity, spectral_radius, gsr=gsr)
        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = SplitBias(bias_scaling, sub_reservoirs)
        else:
            bias_init = None

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
                 bias_scaling=None,
                 connectivity=1.,
                 spectral_radius=0.9,
                 gsr: Optional[float] = None,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        init = generate_split_kernel_inits(input_scaling, sub_reservoirs)
        kernel_init = SplitKernel(sub_reservoirs, initializers=init)
        recurrent_kernel_init = Type3(sub_reservoirs, connectivity, spectral_radius, gsr=gsr)
        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = SplitBias(bias_scaling, sub_reservoirs)
        else:
            bias_init = None

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
                 bias_scaling=None,
                 connectivity=1.,
                 spectral_radius=0.9,
                 gsr: Optional[float] = None,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        init = generate_split_kernel_inits(input_scaling, sub_reservoirs)
        kernel_init = SplitKernel(sub_reservoirs, partitions=partitions, initializers=init)
        recurrent_kernel_init = Type4(sub_reservoirs, partitions, connectivity, spectral_radius, gsr=gsr)

        self.use_bias = bias_scaling is not None
        if self.use_bias:
            bias_init = SplitBias(bias_scaling, sub_reservoirs)
        else:
            bias_init = None

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, kernel_initializer=kernel_init, recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])
