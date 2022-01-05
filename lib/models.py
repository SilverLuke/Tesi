from matplotlib.ticker import MultipleLocator
from tensorflow import keras
import lib.esn
from lib.initializers import *

# For plotting the model
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl


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

    def plot(self, name, experiment, title=None, path=None, x_size=10, show=True):
        plt.interactive(False)
        if not self.build:
            raise Exception("Train the model first")

        kernel_m = self.reservoir.layers[1].cell.kernel
        rec_kernel_m = self.reservoir.layers[1].cell.recurrent_kernel
        readout_m = self.readout.layers[0].weights[0]

        width_ratios = [
            rec_kernel_m.shape[1],
            readout_m.shape[1],
        ]
        height_ratios = [
            kernel_m.shape[0],
            rec_kernel_m.shape[0],
            rec_kernel_m.shape[0] / 20
        ]

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
            fig.suptitle(name + ' ' + experiment, fontsize=20, fontweight='bold')
        else:
            fig.suptitle(title, fontsize=20, fontweight='bold')
        gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig, height_ratios=height_ratios, width_ratios=width_ratios,
                               left=0.05, right=0.95, bottom=0.05, top=0.92)

        rec_kernel = fig.add_subplot(gs[2])
        kernel = fig.add_subplot(gs[0], sharex=rec_kernel)
        readout = fig.add_subplot(gs[3], sharey=rec_kernel)
        bar = fig.add_subplot(gs[4])

        cmap = mpl.cm.get_cmap("RdBu", 31).copy()

        #norm = mpl.colors.Normalize(vmin=-max_val, vmax=max_val)
        norm = mpl.colors.PowerNorm(gamma=0.05)
        rec_kernel.imshow(rec_kernel_m, interpolation=None, aspect=1, resample=False, cmap=cmap, norm=norm)#, norm=mpl.colors.SymLogNorm(0.01))
        kernel.imshow(kernel_m, interpolation=None, aspect=1, resample=False, cmap=cmap, norm=norm)
        readout.imshow(readout_m, interpolation=None, aspect=1, resample=False, cmap=cmap, norm=norm)

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
            fig.savefig(os.path.join(path, name + ' ' + experiment + ".svg"), format='SVG', interpolation='none')
        if show:
            fig.show()
        plt.close(fig)

class ESN1(ESNInterface):
    def __init__(self,
                 units: int,
                 output_units: int,
                 output_activation,
                 input_scaling: float,
                 bias_scaling: float,
                 connectivity: float = 1.,
                 spectral_radius=0.9,
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
                 global_sr: bool = False,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = SplitKernel(sub_reservoirs, initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = Type2(sub_reservoirs, connectivity, spectral_radius, global_sr=global_sr)
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
                 global_sr: bool = False,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = SplitKernel(sub_reservoirs, initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = Type3(sub_reservoirs, connectivity, spectral_radius, global_sr=global_sr)
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
                 global_sr: bool = False,
                 leaky=0.1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        kernel_init = SplitKernel(sub_reservoirs, partitions=partitions, initializer=tf.keras.initializers.RandomUniform(
            minval=-input_scaling, maxval=input_scaling)
        )
        recurrent_kernel_init = Type4(sub_reservoirs, partitions, connectivity, spectral_radius, global_sr=global_sr)

        bias_init = tf.keras.initializers.RandomUniform(minval=-bias_scaling, maxval=bias_scaling)

        self.reservoir = keras.Sequential([
            keras.layers.Masking(),
            lib.esn.ESN(units, leaky, kernel_initializer=kernel_init, recurrent_initializer=recurrent_kernel_init,
                        bias_initializer=bias_init),
        ])
        self.readout = keras.Sequential([
            keras.layers.Dense(output_units, activation=output_activation, name="readout")
        ])
