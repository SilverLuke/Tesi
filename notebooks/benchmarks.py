import json
import re

import keras_tuner
import numpy as np
import os


class Statistic:
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

    def __init__(self, hp, train_acc, validation_acc, test_acc, build_time, score=0):
        self.hyperparameters = hp
        self.accuracies = (train_acc, validation_acc, test_acc)
        self.time = build_time
        self.score = score

    def add_score(self, score):
        self.score = score

    def get_accuracy_mean(self, acc_type=TEST):
        return np.mean(self.accuracies[acc_type])

    def get_accuracy_std(self, acc_type=TEST):
        return np.std(self.accuracies[acc_type])

    def get_accuracy_min(self, acc_type=TEST):
        return np.min(self.accuracies[acc_type])

    def get_accuracy_max(self, acc_type=TEST):
        return np.max(self.accuracies[acc_type])

    def get_accuracy_error_min(self, acc_type=TEST):
        return np.mean(self.accuracies[acc_type]) - np.min(self.accuracies[acc_type])

    def get_accuracy_error_max(self, acc_type=TEST):
        return np.max(self.accuracies[acc_type]) - np.mean(self.accuracies[acc_type])

    def get_accuracy_str(self, acc_type=TEST):
        return "{:.2f}±{:.2f} %".format(self.get_accuracy_mean(acc_type=acc_type) * 100.,
                                        self.get_accuracy_std(acc_type=acc_type) * 100)

    def get_accuracy_latex(self, acc_type=TEST):
        return "${:.2f} \pm {:.2f} \% $".format(self.get_accuracy_mean(acc_type=acc_type) * 100.,
                                        self.get_accuracy_std(acc_type=acc_type) * 100)

    def get_time_mean(self):
        return np.mean(self.time)

    def get_time_std(self):
        return np.std(self.time)

    def __str__(self):
        ret = "  Accuracies\n".format(self.get_accuracy_str())
        ret += "      TRAIN : {}\n".format(self.get_accuracy_str(acc_type=Statistic.TRAIN))
        ret += " VALIDATION : {}\n".format(self.get_accuracy_str(acc_type=Statistic.VALIDATION))
        ret += "      SCORE : {}\n".format(self.score)
        ret += "       TEST : {}\n".format(self.get_accuracy_str(acc_type=Statistic.TEST))
        ret += " Build time : {:.2f}±{:.2f}s\n".format(self.get_time_mean(), self.get_time_std())
        #  ret += "Hyperparameters:\n"
        #  for key, val in self.hyperparameters.values.items():
        #      ret += "\t{}: {}\n".format(key, val)
        return ret

    def toJson(self):
        config = {
            'hyperparameters': self.hyperparameters.get_config(),
            'accuracies'     : self.accuracies,
            'score'          : self.score,
            'build time'     : self.time,
        }
        return config

    @classmethod
    def fromJson(cls, values):
        return cls(keras_tuner.engine.hyperparameters.HyperParameters.from_config(values['hyperparameters']),
                   values['accuracies'][Statistic.TRAIN],
                   values['accuracies'][Statistic.VALIDATION],
                   values['accuracies'][Statistic.TEST], values['build time'], score=values.get('score'))


class StatisticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Statistic):
            return obj.toJson()
        return json.JSONEncoder.default(self, obj)


def get_min_accuracy(tree):
    min_acc = 10.  # accuracy is a value form 0. to 1. so anything is below 10.
    for stat in tree.values():
        acc_sum = stat.get_accuracy_mean() - stat.get_accuracy_std()
        min_acc = min(min_acc, acc_sum)
    return min_acc


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


def get_models_names(tree):
    tmp = set()
    for models in tree.values():
        for model in models.keys():
            tmp.add(model)
    tmp = list(tmp)
    tmp.sort(key=models_sort)
    return tmp


def models_sort(s):
    order = ["ESN", "IRESN", "IIRESN", "IIRESNvsr"]
    pos = []
    for element in s:
        try:
            i = order.index(element)
        except ValueError:
            i = len(s)
        pos.append(i)
    return pos


def get_sorted_keys(tree, sort=natural_sort):
    tmp = tree
    if isinstance(tree, dict):
        tmp = list(tree.keys())
    tmp.sort(key=sort)
    return tmp


class BenchmarksDB:
    def __init__(self, load_path=None):
        self.head = dict()
        self.file_path = load_path
        self.min_y_axis = None

        if self.file_path is not None:
            self.load(self.file_path)

    def __iter__(self):
        for dataset, classes in self.head.items():
            for class_name, experiments in classes.items():
                for experiment, models in experiments.items():
                    for model, stat in models.items():
                        yield dataset, class_name, experiment, model, stat

    def __getitem__(self, key):
        return self.head[key]

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

    def update_y_min(self, acc):
        if self.min_y_axis is None:
            self.min_y_axis = acc
        else:
            self.min_y_axis = min(self.min_y_axis, acc)

    def add(self, dataset_name: str, class_name: str, experiment_name: str, model_name: str, stat: Statistic):
        try:
            # This is the fast way
            self.head[dataset_name][class_name][experiment_name][model_name] = stat
        except KeyError as _:
            dataset_dict = self.head.get(dataset_name)
            if dataset_dict is None:  # Dataset not already present add all to it
                exp_class = {class_name: {experiment_name: {model_name: stat}}}
                self.head[dataset_name] = exp_class
            else:
                class_dict = dataset_dict.get(class_name)
                if class_dict is None:  # Class experiment_name not present add the rest to it
                    exp = {experiment_name: {model_name: stat}}
                    dataset_dict[class_name] = exp
                else:
                    benchmarks = class_dict.get(experiment_name)
                    if benchmarks is None:  # Experiment not present add the list to it
                        class_dict[experiment_name] = {model_name: stat}
                    else:  # This should be useless because if there is the experiment the code in try will work
                        model_stat = benchmarks.get(model_name)
                        if model_stat is not None:
                            print("Already present. Default behavior is overwrite.")
                        benchmarks[model_name] = stat
        self.update_y_min(stat.get_accuracy_min())

    def get(self, dataset_name: str, class_name: str, experiment_name: str, model_name: str):
        stat = None
        try:
            stat = self.head[dataset_name][class_name][experiment_name][model_name]
        except KeyError as _:
            pass
        return stat

    def copy_stats(self, dataset, equals_classes, experiment, model_name, stat):
        for class_name in equals_classes:
            self.add(dataset, class_name, experiment, model_name, stat)

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
            tmp.sort(key=models_sort)
            return tmp

    def is_benchmarked(self, dataset_name, class_name, exp_name, model_name):
        try:
            _ = self.head[dataset_name][class_name][exp_name][model_name]
            return True
        except KeyError as _:
            return False

    def get_min_acc(self):
        if self.min_y_axis is None:
            self.min_y_axis = 10.0
            for dataset, class_name, experiment, model, stat in self:
                acc = stat.get_accuracy_min()
                self.min_y_axis = min(acc, self.min_y_axis)
            return self.min_y_axis
        else:
            return self.min_y_axis
