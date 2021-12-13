import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras_tuner.tuners import Hyperband

from matplotlib import gridspec
import matplotlib.pyplot as plt

import os
from time import *
import notify2


class Benchmark:
    def __init__(self, name: str, trained_model, hp, accuracy_stat, time_stat):
        self.name = name
        self.model = trained_model
        self.hyperparams = hp
        self.acc_mean = accuracy_stat[0]
        self.acc_std = accuracy_stat[1]
        self.time_mean = time_stat[0]
        self.time_std = time_stat[1]

    def __str__(self):
        ret = "Model name: {}\n".format(self.name)
        ret += "Accuracy  : {:.2f}±{:.2f}%\n".format(self.acc_mean * 100., self.acc_std * 100)
        ret += "Build time: {:.2f}±{:.2f}s\n".format(self.time_mean, self.time_std)
        ret += "Hyperparameters:\n"
        for key, val in self.hyperparams.values.items():
            ret += "\t{}: {}\n".format(key, val)
        return ret


def tune_and_test(name, build_model_fn,
                  train_set, val_set, test_set,
                  max_epochs, patience, guesses,
                  benchmarks,
                  tuner_path=None, tensorboard_path=None,
                  benchmarks_verbose=0):
    x_train, y_train = train_set
    x_val, y_val = val_set
    x_test, y_test = test_set

    tuner = Hyperband(
        build_model_fn,
        objective='val_accuracy',
        max_epochs=max_epochs,
        directory=tuner_path,
        project_name=name + ' hyperband',
        hyperband_iterations=1,
        seed=42)

    tuner.search(x_train, y_train,
                 epochs=max_epochs,
                 validation_data=(x_val, y_val),
                 callbacks=[
                     keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
                 ])

    # choose the best hyperparameters  # tf.keras.callbacks.CallbackList([])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    if tensorboard_path is not None:
        tensorboard_dir = tensorboard_path + name
        callbacks.append(keras.callbacks.TensorBoard(tensorboard_dir, profile_batch='500,500'))

    print("Start {} benchmarks:".format(guesses))
    best_model_hp = tuner.get_best_hyperparameters()[0]
    test_model = None

    metrics_ts = []
    loss_ts = []
    required_time = []

    tf.random.set_seed(42)
    for i in range(guesses):
        initial_time = time()
        test_model = tuner.hypermodel.build(best_model_hp)
        test_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs,
                       callbacks=callbacks, verbose=benchmarks_verbose)
        loss, metrics = test_model.evaluate(x_test, y_test)
        required_time.append(time() - initial_time)
        metrics_ts.append(metrics)
        loss_ts.append(loss)

    acc_stat = (np.mean(metrics_ts), np.std(metrics_ts))
    loss_stat = (np.mean(loss_ts), np.std(loss_ts))
    time_stat = (np.mean(required_time), np.std(required_time))

    benchmarks[name] = Benchmark(name, test_model, best_model_hp, acc_stat, time_stat)
    print('--C--')
    print('Results:')
    print('\tAccuracy: {:.2f}±{:.2f}%'.format(acc_stat[0] * 100., acc_stat[1] * 100))
    print('\tLoss: {:.2f}±{:.2f}'.format(loss_stat[0], loss_stat[1]))
    print('\tTime: {:.2f}±{:.2f}s'.format(time_stat[0], time_stat[1]))
    send_notification(name, "Accuratezza: {:.2f}±{:.2f}%".format(acc_stat[0] * 100., acc_stat[1] * 100))

    return test_model


def send_notification(title, message):
    notice = notify2.Notification(title, message)
    notice.show()


# Return a vector of units for each sub-reservoir
def equalize_units(units, features):
    local_units = [units // features for _ in range(features)]
    if units % features != 0:
        used_units = (units // features) * features
        rest = units - used_units
        for i in range(rest):
            local_units[i] += 1
    return local_units


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
    return partitions


def get_spectral_radius(tensor):
    return tf.reduce_max(tf.abs(tf.linalg.eig(tensor)[0]))


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


def check_and_vectorize(value, dtype, length, name, _min=0.0, _max=1.0):
    def out_of_bound(val):
        if not _min <= val <= _max:
            raise ValueError(
                "{} must be inside the [{};{}] range so value {} is wrong.".format(name, _min, _max, value))

    if isinstance(value, dtype):
        out_of_bound(value)
        return [value for _ in range(length)]
    elif isinstance(value, list):
        if len(value) != length:
            raise ValueError("{} length is wrong should be {} instead of {}".format(name, length, len(value)))
        for elem in value:
            if isinstance(elem, dtype):
                out_of_bound(elem)


# Spectral radius should be between 0 and 1 to maintain the echo state property
def generate_matrix(shape, initializer, spectral_radius, connectivity, dtype=None):
    if connectivity == 1:
        if spectral_radius is not None:
            return unipi_generate_matrix(shape, spectral_radius)
        else:
            return tf.random.uniform(shape)
    if connectivity == 0:
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
def join_matrices(matricex):
    ret = tf.concat(matricex[0], axis=1)
    for r_k in matricex[1:]:
        tmp = tf.concat(r_k, axis=1)  # axis=1 horizontal concatenation
        ret = tf.concat([ret, tmp], axis=0)
    return ret


def plot_matrices(matrices, name, path=None, units_per_inch=50,
                  titles=["kernel", "reservoir", "readout"],
                  y_visible=[True, False, False]):
    if len(matrices) != 3:
        raise ValueError("Matrices len should be 3 -> (kernel, reservoir, readout)")

    width_ratios = [
        matrices[0].shape[1],
        matrices[1].shape[1],
        matrices[2].shape[1],
    ]

    x = (sum(width_ratios) / units_per_inch) + 2
    y = (max(width_ratios) / units_per_inch) + 2

    fig = plt.figure(figsize=(x, y))

    gs = gridspec.GridSpec(nrows=1, ncols=len(matrices), figure=fig, height_ratios=[1], width_ratios=width_ratios,
                           wspace=0.08, top=1 - 0.20, right=1 - 0.15, bottom=0.075, left=0.075)
    fig.suptitle(name, y=0.975)

    axes = [fig.add_subplot(gs[0])]
    axes.append(fig.add_subplot(gs[1], sharey=axes[0]))
    axes.append(fig.add_subplot(gs[2], sharey=axes[0]))

    c = None
    for axe, matrix, title, visible in zip(axes, matrices, titles, y_visible):
        c = axe.matshow(matrix, aspect='auto')
        y_size = matrix.shape[1]
        if y_size < 20:
            axe.set_xticks([int(0), int(y_size)])
        axe.get_yaxis().set_visible(visible)
        axe.set_title(title)

    # Add the lateral colorbar with the same size of other bars
    pos = axes[2].get_position()
    cax = fig.add_axes(rect=[pos.x0 + pos.width + 0.02,
                             pos.y0,
                             0.03,
                             pos.height])
    fig.colorbar(mappable=c, cax=cax)

    if path is not None:
        fig.savefig(path + os.sep + name + ".svg", format='SVG', dpi=units_per_inch)
    fig.show()
