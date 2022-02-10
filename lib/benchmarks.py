import json
from itertools import groupby

import keras_tuner
import numpy as np
import os
import re

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from time import time


class Statistic:
    def __init__(self, hp, accuracy, loss, build_time, timestamp=None):
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

    def get_accuracy_min(self):
        return np.mean(self.accuracy) - np.min(self.accuracy)

    def get_accuracy_max(self):
        return np.max(self.accuracy) - np.mean(self.accuracy)

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
        ret = "  Accuracy : {}\n".format(self.get_accuracy_str())
        ret += "      Loss : {:.2f}±{:.2f}\n".format(self.get_loss_mean(), self.get_loss_std())
        ret += "Build time : {:.2f}±{:.2f}s\n".format(self.get_time_mean(), self.get_time_std())
        #  ret += "Hyperparameters:\n"
        #  for key, val in self.hyperparameters.values.items():
        #      ret += "\t{}: {}\n".format(key, val)
        return ret

    def toJson(self):
        config = {
            'hyperparameters': self.hyperparameters.get_config(),
            'accuracy': self.accuracy,
            'loss': self.loss,
            'build time': self.time,
            'timestamp': self.timestamp,
        }
        return config

    @classmethod
    def fromJson(cls, values):
        return cls(keras_tuner.engine.hyperparameters.HyperParameters.from_config(values['hyperparameters']),
                   values['accuracy'], values['loss'], values['build time'],
                   timestamp=values['timestamp'])


class StatisticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Statistic):
            return obj.toJson()
        return json.JSONEncoder.default(self, obj)


class BenchmarksDB:
    def __init__(self, load_path=None, plot_path=None):
        self.head = dict()
        self.file_path = load_path
        self.plot_path = plot_path
        if self.file_path is not None:
            self.load(self.file_path)
        self.min_acc = None

    def __iter__(self):
        for dataset, classes in self.head.items():
            for class_name, experiments in classes.items():
                for experiment, models in experiments.items():
                    for model, stat in models.items():
                        yield dataset, class_name, experiment, model, stat

    def save(self, path=None):
        save_path = get_path(path, self.file_path)
        with open(save_path, "w") as out_file:
            json.dump(self.head, out_file, cls=StatisticEncoder, indent=4)

    def load(self, path=None):
        load_path = get_path(path, self.file_path)

        if os.path.exists(load_path):
            with open(load_path, 'r') as jsonfile:
                datasets = json.load(jsonfile)
                for dataset, class_names in datasets.items():
                    for class_name, experiments in class_names.items():
                        for experiment, models in experiments.items():
                            for model, stats in models.items():
                                stat = Statistic.fromJson(stats)
                                self.add(dataset, class_name, experiment, model, stat)
        else:
            with open(load_path, 'w') as jsonfile:
                json.dump({}, jsonfile)

    def add(self, dataset_name: str, class_name: str, experiment_name: str, model: str, stats: Statistic):
        try:
            # This is the fast way
            self.head[dataset_name][class_name][experiment_name][model] = stats
        except KeyError as _:
            dataset_dict = self.head.get(dataset_name)
            if dataset_dict is None:  # Dataset not already present add all to it
                exp_class = {class_name: {experiment_name: {model: stats}}}
                self.head[dataset_name] = exp_class
            else:
                class_dict = dataset_dict.get(class_name)
                if class_dict is None:  # Class experiment not present add the rest to it
                    exp = {experiment_name: {model: stats}}
                    dataset_dict[class_name] = exp
                else:
                    benchmarks = class_dict.get(experiment_name)
                    if benchmarks is None:  # Experiment not present add the list to it
                        class_dict[experiment_name] = {model: stats}
                    else:
                        model_stat = benchmarks.get(model)
                        if model_stat is not None:
                            print("Already present. Default behavior is overwrite.")
                        benchmarks[model] = stats
        self.min_acc = None

    def get(self, dataset_name: str, class_name: str = None, experiment_name: str = None, model: str = None):
        def get_rec(tree, labels):
            if len(labels) == 0:
                return tree
            label = labels.pop()
            new_tree = tree.get(label)
            if new_tree is None:
                print("No labels with this name: ", label)
                return None
            else:
                return get_rec(new_tree, labels)

        return get_rec(self.head, list(filter(None, [dataset_name, class_name, experiment_name, model])))

    def list_datasets(self):
        tmp = list(self.head.keys())
        tmp.sort(key=natural_sort)
        return tmp

    def list_classes(self, op_type='join'):
        if op_type == 'join':
            tmp = set()
            for classes_dicts in self.head.values():  # head.values return a list of dicts
                for class_name in classes_dicts.keys():  # add each keys to the set
                    tmp.add(class_name)
            tmp = list(tmp)
            tmp.sort(key=natural_sort)
            return tmp

    def list_experiments(self, op_type='join'):
        if op_type == 'join':
            tmp = set()
            for classes_dicts in self.head.values():  # head.values return a list of dicts
                for experiments_dict in classes_dicts.values():  # add each keys to the set
                    for exp in experiments_dict.keys():
                        tmp.add(exp)
            tmp = list(tmp)
            tmp.sort(key=natural_sort)
            return tmp

    def list_models(self, op_type='join'):
        if op_type == 'join':
            tmp = set()
            for classes_dicts in self.head.values():  # head.values return a list of dicts
                for experiments_dict in classes_dicts.values():  # add each keys to the set
                    for models in experiments_dict.values():
                        for model in models.keys():
                            tmp.add(model)
            tmp = list(tmp)
            tmp.sort(key=natural_sort)
            return tmp

    def is_benchmarked(self, dataset_name, class_name, exp_name, model_name):
        try:
            _ = self.head[dataset_name][class_name][exp_name][model_name]
            return True
        except KeyError as ex:
            return False

    def get_min_acc(self):
        if self.min_acc is None:
            self.min_acc = 10.0
            for dataset, class_name, experiment, model, stat in self:
                acc = stat.get_accuracy_mean() - stat.get_accuracy_std()
                self.min_acc = min(acc, self.min_acc)
            return self.min_acc
        else:
            return self.min_acc

    def _save_and_show(self, fig, path, dataset_name, class_name, plot_name, show):
        path = get_path(path, self.plot_path)
        if path is not None:
            if dataset_name is not None:
                path = os.path.join(path, dataset_name, lower_and_replace(class_name))
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, lower_and_replace(plot_name + ".svg"))
            print(path)
            fig.savefig(path, format='SVG', dpi=100, bbox_inches='tight')
        if show:
            fig.show()
        plt.close(fig)

    def plot_datasets_summary(self, class_name: str = 'Best Models', experiment: str = 'Best',
                              path: str = None, show: bool = True):
        models = self.list_models()
        datasets = get_sorted_keys(self.head)

        values = {name: ([], [], []) for name in models}
        for i, dataset in enumerate(datasets):
            try:
                tested_models = self.head[dataset][class_name][experiment].keys()
            except Exception as _:
                return
            for j, model in enumerate(models):
                if model in tested_models:
                    stat = self.head[dataset][class_name][experiment][model]
                    values[model][0].append(stat.get_accuracy_mean())
                    values[model][1].append(stat.get_accuracy_min())
                    values[model][2].append(stat.get_accuracy_max())
                else:
                    values[model][0].append(0)
                    values[model][1].append(0)
                    values[model][2].append(0)

        fig = self._plot_histograms('Exp. Class: "' + class_name + '"\nExperiment: "' + experiment + '"', values, datasets)
        self._save_and_show(fig, path, None, class_name, "datasets summary " + class_name + " " + experiment, show)

    def plot_lines_by_key(self, dataset_name: str, class_name: str, key: str = 'units',
                          path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]
        experiments = get_sorted_keys(exp_tree)
        models = get_models_names(exp_tree)

        values = {name: ([], [], [], []) for name in models}
        for i, experiment in enumerate(experiments):
            tested_models = exp_tree[experiment].keys()
            for j, model in enumerate(models):
                if model in tested_models:
                    stat = exp_tree[experiment][model]
                    values[model][0].append(stat.hyperparameters.values.get(key))
                    values[model][1].append(stat.get_accuracy_mean())
                    values[model][2].append(stat.get_accuracy_min())
                    values[model][3].append(stat.get_accuracy_max())

        fig, ax = plt.subplots()
        for model, val in values.items():
            sort_list = list(zip(*sorted(zip(*val))))
            (_, caps, _) = ax.errorbar(sort_list[0], sort_list[1], yerr=[sort_list[2], sort_list[3]], marker='o',
                                       label=model, markersize=3, capsize=5)
            for cap in caps:
                cap.set_markeredgewidth(1)

        ax.set_ylim(top=1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel(key.capitalize())
        ax.set_title('Dataset: "' + dataset_name + '"\nExp. Class: "' + class_name + '"\nhp: ' + key)

        ax.legend(loc='lower right')

        self._save_and_show(fig, path, dataset_name, class_name, "summary plots", show)

    def plot_summary_histogram(self, dataset_name: str, class_name: str,
                               path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]
        experiments = get_sorted_keys(exp_tree)
        models = get_models_names(exp_tree)

        values = {name: ([], [], []) for name in models}
        for i, experiment in enumerate(experiments):
            tested_models = exp_tree[experiment].keys()
            for j, model in enumerate(models):
                if model in tested_models:
                    stat = exp_tree[experiment][model]
                    values[model][0].append(stat.get_accuracy_mean())
                    values[model][1].append(stat.get_accuracy_min())
                    values[model][2].append(stat.get_accuracy_max())
                else:
                    values[model][0].append(0)
                    values[model][1].append(0)
                    values[model][2].append(0)

        fig = self._plot_histograms('Dataset: "' + dataset_name + '" Exp. Class: "' + class_name + '"', values, experiments)
        self._save_and_show(fig, path, dataset_name, class_name, "summary histograms", show)

    def plot_by_experiment(self, dataset_name: str, class_name: str,
                           path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]
        labels = get_models_names(exp_tree)
        for exp_name, benchmarks in exp_tree.items():
            acc_mean = [benchmarks[label].get_accuracy_mean() if benchmarks.get(label) is not None else 0.
                        for label in labels]
            acc_std = [benchmarks[label].get_accuracy_std() if benchmarks.get(label) is not None else 0.
                       for label in labels]

            fig = self._plot_histogram(exp_name, acc_mean, acc_std, labels)
            self._save_and_show(fig, path, dataset_name, class_name, exp_name, show)

    # Not have much sense
    def plot_by_model(self, dataset_name: str, class_name: str,
                      path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]
        models = get_models_names(exp_tree)

        for model in models:
            experiments = get_sorted_keys(exp_tree)
            acc_mean = [exp_tree[exp][model].get_accuracy_mean() if exp_tree[exp].get(model) is not None else None
                        for exp in experiments]
            acc_min = [exp_tree[exp][model].get_accuracy_min() if exp_tree[exp].get(model) is not None else None
                       for exp in experiments]
            acc_max = [exp_tree[exp][model].get_accuracy_max() if exp_tree[exp].get(model) is not None else None
                       for exp in experiments]

            fig = self._plot_histogram(model, acc_mean, [acc_min, acc_max], experiments)
            self._save_and_show(fig, path, dataset_name, class_name, model, show)

    def plot_summary_table(self, dataset_name: str, class_name: str,
                           path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]

        experiments = get_sorted_keys(exp_tree)
        models = get_models_names(exp_tree)

        body = []
        for model in models:
            row = []
            for experiment in experiments:
                try:
                    stat = exp_tree[experiment][model]
                    row.append(stat.get_accuracy_str())
                except KeyError as _:
                    row.append('-')
            body.append(row)

        fig = plot_table('Dataset: "' + dataset_name + '"\nExp. Class: "' + class_name + '"', experiments, models, body)
        self._save_and_show(fig, path, dataset_name, class_name, "summary table", show)

    # X is hp Y is experiments
    def plot_hp_table_by_model(self, dataset_name: str, class_name: str, model: str, keys: [],
                               path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]
        experiments = get_sorted_keys(exp_tree)

        hp = set()
        for experiment in experiments:
            hp.update(exp_tree[experiment][model].hyperparameters.values.keys())
        hp_keys = get_matching_keys(keys, hp)
        try:
            hp_keys.remove("use G.S.R.")
        except Exception as _:
            pass
        finally:
            hp_keys = get_sorted_keys(hp_keys)

        body = []
        for experiment in experiments:
            row = []
            for key in hp_keys:
                value = exp_tree[experiment][model].hyperparameters.values.get(key)
                if isinstance(value, float):
                    row.append(float("{:0.3f}".format(value)))
                elif isinstance(value, int):
                    row.append(value)
                elif value is None:
                    row.append("-")
                else:
                    print(type(value))
            body.append(row)

        fig = plot_table('Dataset: "' + dataset_name + '"\n' +
                         'Exp. Class: "' + class_name + '"\n' +
                         'Model: "' + model + '"', hp_keys, experiments, body)
        self._save_and_show(fig, path, dataset_name, class_name, "hp_table_by_"+model, show)

    # X is hp Y is model
    def plot_hp_table_by_experiment(self, dataset_name: str, class_name: str, experiment: str, keys: [],
                                    path: str = None, show: bool = True):
        try:
            models_tree = self.head[dataset_name][class_name][experiment]
        except KeyError as _:
            # print("Error: ", class_name, experiment, "Not found")
            return

        models = get_sorted_keys(models_tree)

        hp = set()
        for model in models:
            hp.update(models_tree[model].hyperparameters.values.keys())
        hp_keys = get_matching_keys(keys, hp)
        try:
            hp_keys.remove("use G.S.R.")
        except Exception as _:
            pass
        finally:
            hp_keys = get_sorted_keys(hp_keys)

        body = []
        for model in models:
            row = []
            for key in hp_keys:
                value = models_tree[model].hyperparameters.values.get(key)
                if isinstance(value, float):
                    row.append(float("{:0.3f}".format(value)))
                elif isinstance(value, int):
                    row.append(value)
                elif value is None:
                    row.append("-")
                else:
                    print(type(value))
            body.append(row)

        fig = plot_table('Dataset: "' + dataset_name + '"\n' +
                         'Exp. Class: "' + class_name + '"\n' +
                         'Experiment: "' + experiment + '"', hp_keys, models, body)
        self._save_and_show(fig, path, dataset_name, class_name, "hp_table_by_"+experiment, show)

    def _plot_histograms(self, title, data, x_labels):
        fig, ax = plt.subplots()
        width = 0.15
        margin = 0.02
        total = width + margin
        x = np.arange(len(x_labels))
        for i, (model, val) in enumerate(data.items()):
            ax.bar(x - 1.5 * total + (i * total), val[0], yerr=[val[1], val[2]],
                   label=model,
                   width=width,
                   align='center',
                   capsize=3)

        x_new = []
        for label in x_labels:
            new = "" + label[0]
            for c in label[1:]:
                if c.isupper():
                    new += '\n'
                new += c
            x_new.append(new)

        ax.set_xticks(ticks=x, labels=x_new)
        ax.set_ylabel('Accuracy')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_title(title)
        ax.legend(loc='lower right')
        return fig

    def _plot_histogram(self, title, acc_mean, acc_std, x_labels):
        acc_mean = np.array(acc_mean)
        acc_std = np.array(acc_std)

        fig, ax = plt.subplots()
        x_pos = np.arange(len(acc_mean))
        ax.bar(x_pos, acc_mean,
               width=0.3,
               yerr=acc_std,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)

        ax.set_ylabel('Accuracy')
        ax.set_ylim([self.get_min_acc(), 1.0])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(title)
        ax.yaxis.grid(True)

        plt.tight_layout()
        return fig


def get_path(op_path, obj_path):
    path = op_path
    if path is None:
        if obj_path is None:
            raise ValueError("No path is given")
        else:
            path = obj_path
    return path


def natural_sort(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def get_matching_keys(exps, keys):
    ret = []
    for hp in keys:
        if any(map(hp.__contains__, exps)):
            ret.append(hp)
    return ret


def plot_table(title, x_labels, y_labels, body):
    width = (len(x_labels) + 1)
    height = len(y_labels) / 2
    fig, ax = plt.subplots(figsize=(width, height))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title)
    ax.table(cellText=body, rowLabels=y_labels, colLabels=x_labels,
             loc='center', cellLoc='center', edges='horizontal')

    return fig


def get_min_accuracy(tree):
    min_acc = 10.  # accuracy is a value form 0. to 1. so anything is below 10.
    for stat in tree.values():
        acc_sum = stat.get_accuracy_mean() - stat.get_accuracy_std()
        min_acc = min(min_acc, acc_sum)
    return min_acc


def get_models_names(tree):
    tmp = set()
    for models in tree.values():
        for model in models.keys():
            tmp.add(model)
    tmp = list(tmp)
    tmp.sort(key=natural_sort)
    return tmp


def get_sorted_keys(tree):
    tmp = tree
    if isinstance(tree, dict):
        tmp = list(tree.keys())
    tmp.sort(key=natural_sort)
    return tmp


def lower_and_replace(string: str):
    return string.lower().replace(" ", "_")