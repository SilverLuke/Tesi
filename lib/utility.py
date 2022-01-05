import tensorflow as tf
from matplotlib.ticker import MultipleLocator
from tensorflow import keras
import keras_tuner
from keras_tuner.tuners import Hyperband
import numpy as np

import matplotlib.pyplot as plt

import os
import re
from time import *
import notify2

notify2_init = False


def send_notification(title, message):
    def notify2_init_fun():
        global notify2_init
        if not notify2_init:
            notify2.init("Tesi")

    notify2_init_fun()

    notice = notify2.Notification(title, message)
    notice.show()


class Benchmark:
    def __init__(self, model: str, experiment, hp, accuracy, loss, build_time, timestamp=None):
        self.model = model
        self.experiment = experiment
        self.hyperparameters = hp
        self.accuracy = accuracy
        self.loss = loss
        self.time = build_time
        if timestamp is None:
            self.timestamp = int(time())
        else:
            self.timestamp = timestamp

    def get_accuracy_mean(self):
        return np.mean(self.accuracy)

    def get_accuracy_std(self):
        return np.std(self.accuracy)

    def get_accuracy_str(self):
        return "{:.2f}±{:.2f} %".format(self.get_accuracy_mean() * 100., self.get_accuracy_std() * 100)

    def get_loss_mean(self):
        return np.mean(self.loss)

    def get_loss_std(self):
        return np.std(self.loss)

    def get_time_mean(self):
        return np.mean(self.time)

    def get_time_std(self):
        return np.std(self.time)

    def __str__(self):
        ret = "     Model : {}\n".format(self.model)
        ret += "Experiment : {}\n".format(self.experiment)
        ret += "  Accuracy : {}\n".format(self.get_accuracy_str())
        ret += "      Loss : {:.2f}±{:.2f}\n".format(self.get_loss_mean(), self.get_loss_std())
        ret += "Build time : {:.2f}±{:.2f}s\n".format(self.get_time_mean(), self.get_time_std())
        #  ret += "Hyperparameters:\n"
        #  for key, val in self.hyperparameters.values.items():
        #      ret += "\t{}: {}\n".format(key, val)
        return ret

    def toJson(self):
        config = {
            'model': self.model,
            'experiment': self.experiment,
            'hp': self.hyperparameters.get_config(),
            'accuracy': self.accuracy,
            'loss': self.loss,
            'build time': self.time,
            'timestamp': self.timestamp,
        }
        return config

    @classmethod
    def fromJson(cls, values):
        return cls(values['model'], values['experiment'],
                   keras_tuner.engine.hyperparameters.HyperParameters.from_config(values['hp']),
                   values['accuracy'], values['loss'], values['build time'],
                   timestamp=values['timestamp'])


def is_benchmarked(benchmarks, model_name, experiment):
    for b in benchmarks:
        if b.model == model_name and b.experiment == experiment:
            return True
    return False


def tune_and_test(model_name, build_model_fn, experiment_name,
                  train_set, val_set, test_set,
                  max_epochs, patience, guesses,
                  benchmarks,
                  tuner_path=None, tensorboard_path=None,
                  benchmarks_verbose=0, notify=False):
    x_train, y_train = train_set
    x_val, y_val = val_set
    x_test, y_test = test_set

    tuner = Hyperband(
        build_model_fn,
        objective='val_accuracy',
        max_epochs=max_epochs,
        directory=tuner_path,
        project_name=model_name + ' ' + experiment_name + ' hyperband',
        hyperband_iterations=1,
        seed=42)

    tuner.search(x_train, y_train,
                 epochs=max_epochs,
                 validation_data=(x_val, y_val),
                 callbacks=[
                     keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
                 ])  # , workers=4, use_multiprocessing=True)

    # choose the best hyperparameters  # tf.keras.callbacks.CallbackList([])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
    if tensorboard_path is not None:
        tensorboard_dir = tensorboard_path + model_name
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
                       callbacks=callbacks, verbose=benchmarks_verbose)  # , workers=4, use_multiprocessing=True)
        loss, metrics = test_model.evaluate(x_test, y_test)

        required_time.append(time() - initial_time)
        metrics_ts.append(metrics)
        loss_ts.append(loss)

    acc_stat = (np.mean(metrics_ts), np.std(metrics_ts))

    summary = Benchmark(model_name, experiment_name, best_model_hp, metrics_ts, loss_ts, required_time)
    present = next(
        ((i, x) for i, x in enumerate(benchmarks) if x.model == summary.model and x.experiment == summary.experiment),
        None)
    if present is None:
        print("Not already present in benchmarks. Adding...")
        benchmarks.append(summary)
    else:
        print("Already present. Overwriting..")
        benchmarks[present[0]] = present[1]
    print(summary)

    if notify:
        send_notification(model_name, "Accuratezza: {:.2f}±{:.2f}%".format(acc_stat[0] * 100., acc_stat[1] * 100))

    return test_model


def natural_keys(text):
    def atoi(_text):
        return int(_text) if _text.isdigit() else _text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def group_by_experiment(benchmarks, skip_model=[]):
    """
    :param benchmarks:
    :return: a dictionary with key the experiment name and as value a list of benchmarks with that experiment
    """
    tmp = {}
    for b in benchmarks:
        if b.experiment in skip_model:
            continue
        exp = b.experiment
        try:
            tmp[exp].append(b)
        except KeyError:
            tmp[exp] = [b]
    return tmp


def group_by_model(benchmarks, skip_exp=[]):
    tmp = {}
    for b in benchmarks:
        if b.experiment in skip_exp:
            continue
        model = b.model
        try:
            tmp[model].append(b)
        except KeyError:
            tmp[model] = [b]
    return tmp


def get_min_accuracy(benchmarks):
    min_acc = 2.
    for b in benchmarks:
        acc_sum = b.get_accuracy_mean() - b.get_accuracy_std()
        min_acc = min(min_acc, acc_sum)
    return min_acc


def get_experiments_label(texts):
    experiments_labels = []
    for label in texts:
        if "Connectivity 1" in label:
            label = label[:-len(" Connectivity 1")]
            experiments_labels.append(label)
        else:
            experiments_labels.append(label)
    return experiments_labels


def list_experiments(benchmarks):
    tmp = set()
    for model, experiments in benchmarks.items():
        for exp in experiments:
            tmp.add(exp.experiment)
    return tmp


def plot_by_experiment(all_benchmarks, path=None, show=True):
    group = group_by_experiment(all_benchmarks)
    for exp_name, benchmarks in group.items():
        min_acc = min(get_min_accuracy(benchmarks), 0.5)
        labels = [b.model for b in benchmarks]
        x_pos = np.arange(len(benchmarks))

        width = 0.3
        padding = 0.02

        acc_mean = [b.get_accuracy_mean() for b in benchmarks]
        acc_std = [b.get_accuracy_std() for b in benchmarks]

        fig, ax = plt.subplots()
        ax.bar(x_pos, acc_mean,
               width=width,
               yerr=acc_std,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)

        ax.set_ylabel('Accuracy')
        ax.set_ylim((min_acc, 1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_title(exp_name)
        ax.yaxis.grid(True)

        if path is not None:
            fig.savefig(os.path.join(path, exp_name + ".svg"), format='SVG', dpi=100)
        if show:
            fig.show()
        plt.close(fig)


def plot_by_model(all_benchmarks, sorted=True, path=None, show=True, skip_exp=[]):
    group = group_by_model(all_benchmarks, skip_exp=["Best"])

    for model_name, benchmarks in group.items():
        min_acc = min(get_min_accuracy(benchmarks), 0.5)

        if sorted:
            benchmarks.sort(key=lambda b: b.get_accuracy_mean())  # , reverse=True)
        labels = get_experiments_label([b.experiment for b in benchmarks])

        x_pos = np.arange(len(benchmarks))

        acc_mean = [b.get_accuracy_mean() for b in benchmarks]
        acc_std = [b.get_accuracy_std() for b in benchmarks]
        time_mean = [b.get_time_mean() for b in benchmarks]
        time_std = [b.get_time_std() for b in benchmarks]

        fig, ax = plt.subplots()
        ax.bar(x_pos, acc_mean,
               yerr=acc_std,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)

        ax.set_ylabel('Accuracy')
        ax.set_ylim((min_acc, 1.0))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()

        if path is not None:
            fig.savefig(os.path.join(path, model_name + ".svg"), format='SVG', dpi=100)
        if show:
            fig.show()
        plt.close(fig)


def plot_hp_table(all_benchmarks, path=None, show=True):
    group = group_by_model(all_benchmarks)

    for exp_name, benchmarks in group.items():
        row_lables = get_experiments_label([b.experiment for b in benchmarks])
        hp_labels = set()
        hp_labels.add("units")
        """for b in benchmarks:
            for key in b.hyperparameters.values.keys():
                if not ('->' in key or '/' in key):
                    hp_labels.add(key)"""

        hp_labels = list(hp_labels)
        hp_labels.sort(reverse=True)
        body = []
        for b in benchmarks:
            row = []
            for name in hp_labels:
                try:
                    value = b.hyperparameters.values.get(name)
                    if isinstance(value, float):
                        row.append(float("{:0.3f}".format(value)))
                    elif isinstance(value, int):
                        row.append(value)
                    elif value is None:
                        row.append("-")
                    else:
                        print(type(value))
                except Exception as ex:
                    print(b.model + " " + name + " " + type(ex).__name__)
                    row.append("-")
            body.append(row)
        hp_labels.insert(0, 'Accuracy')
        for i, b in enumerate(benchmarks):
            body[i].insert(0, b.get_accuracy_str())

        width = len(hp_labels) * 3
        height = len(row_lables)
        fig, ax = plt.subplots(figsize=(width, height))
        col_width = 0.25
        hoffset = 0.3718  # find this number from trial and error
        voffset = 0.673  # find this number from trial and error
        line_fac = 0.98  # controls the length of the dividing line

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.set_title('Accuracy by model and experiment')
        the_table = ax.table(cellText=body, rowLabels=row_lables, colWidths=[col_width] * width,
                             loc='center')  # remove colLabels
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1, 1.6)

        count = 0
        for string in hp_labels:
            ax.annotate('  ' + string, xy=(hoffset + count * col_width, voffset),
                        xycoords='axes fraction', ha='left', va='bottom',
                        rotation=45, size=10)
            ax.annotate('', xy=(hoffset + (count + 0.5) * col_width, voffset),
                        xytext=(hoffset + (count + 0.5) * col_width + line_fac / width, voffset + line_fac / height),
                        xycoords='axes fraction', arrowprops={'arrowstyle': '-'})

            count += 1

        if path is not None:
            fig.savefig(os.path.join(path, exp_name + " table.svg"), format='SVG', dpi=100)
        if show:
            fig.show()
        plt.close(fig)


def plot_summary_table(all_benchmarks, path=None, show=True):
    group = group_by_model(all_benchmarks)

    model_labels = []
    experiments = set()
    # build x and y lables
    for model_name, benchmarks in group.items():
        model_labels.append(model_name)
        for b in benchmarks:
            experiments.add(b.experiment)

    experiments = list(experiments)

    experiments.sort(key=natural_keys)

    body = []
    for model_name, benchmarks in group.items():
        row = ['-' for _ in experiments]
        for b in benchmarks:
            index = experiments.index(b.experiment)
            row[index] = b.get_accuracy_str()
        body.append(row)

    width = (len(experiments) + 1) * 2
    height = len(model_labels)
    fig, ax = plt.subplots(figsize=(width, height))
    col_width = 0.11

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title('Accuracy by model and experiment')
    table = ax.table(cellText=body, rowLabels=model_labels, colWidths=[col_width] * width,
                     loc='center')  # remove colLabels
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # custom heading titles - new portion
    hoffset = 0.168  # find this number from trial and error
    voffset = 0.67  # find this number from trial and error
    line_fac = 0.98  # controls the length of the dividing line

    count = 0
    for string in get_experiments_label(experiments):
        ax.annotate('  ' + string, xy=(hoffset + count * col_width, voffset),
                    xycoords='axes fraction', ha='left', va='bottom',
                    rotation=45, size=10)

        # add a dividing line
        ax.annotate('', xy=(hoffset + (count + 0.5) * col_width, voffset),
                    xytext=(hoffset + (count + 0.5) * col_width + line_fac / width, voffset + line_fac / height),
                    xycoords='axes fraction', arrowprops={'arrowstyle': '-'})

        count += 1

    if path is not None:
        fig.savefig(os.path.join(path, "summary table.svg"), format='SVG', dpi=100)
    if show:
        fig.show()
    plt.close(fig)


def plot_histograms(all_benchmarks, dataset, path=None, show=True, skip_exp=[]):
    benchmarks = group_by_model(all_benchmarks, skip_exp)
    x_labels = get_experiments_label(list_experiments(benchmarks))
    datafetch = {}
    for model, bench in benchmarks.items():
        val = ([], [])
        for b in bench:
            val[0].append(b.get_accuracy_mean())
            val[1].append(b.get_accuracy_std())
        datafetch[model] = val

    width = 0.15
    margin = 0.02
    total = width + margin

    test = list(datafetch.values())
    models = list(datafetch.keys())
    x = np.arange(6)

    fig, ax = plt.subplots()
    for i, elem in enumerate(test):
        acc, err = elem
        print(acc, err)
        ax.bar(x + ((i * total) - (total - total / 2)), acc, width, yerr=err, label=models[i],
               align = 'center',
               capsize = 3)

    ax.set_xticks(ticks=x, labels=x_labels, rotation=45)
    ax.set_ylabel('Accuracy')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_title(dataset + str(" dataset"))
    ax.legend(loc='lower right')

    if path is not None:
        plt.savefig(os.path.join(path, "summary histogram.svg"), format='SVG', dpi=100)
    if show:
        plt.show()
    plt.close()


def plot_lines(all_benchmarks, dataset, path=None, show=True, skip_exp=[]):
    benchmarks = group_by_model(all_benchmarks, skip_exp)

    datafetch = {}
    for model, bench in benchmarks.items():
        val = ([], [], [])
        for b in bench:
            val[0].append(b.hyperparameters.values.get('units'))
            val[1].append(b.get_accuracy_mean())
            val[2].append(b.get_accuracy_std())
        datafetch[model] = val

    for model, val in datafetch.items():
        sort_list = list(zip(*sorted(zip(*val))))
        (_, caps, _) = plt.errorbar(sort_list[0], sort_list[1], yerr=sort_list[2], marker='o', label=model,
                                    markersize=3, capsize=10)
        for cap in caps:
            cap.set_markeredgewidth(1)
    plt.title(dataset + " dataset")
    plt.legend(loc='lower right')

    if path is not None:
        plt.savefig(os.path.join(path, "summary plot.svg"), format='SVG', dpi=100)
    if show:
        plt.show()
    plt.close()
