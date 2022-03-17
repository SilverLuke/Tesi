import json
import math
import keras_tuner
import numpy as np
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec


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

    def get_accuracy_error_min(self, acc_type=TEST):
        return np.mean(self.accuracies[acc_type]) - np.min(self.accuracies[acc_type])

    def get_accuracy_error_max(self, acc_type=TEST):
        return np.max(self.accuracies[acc_type]) - np.mean(self.accuracies[acc_type])

    def get_accuracy_str(self, acc_type=TEST):
        return "{:.2f}±{:.2f} %".format(self.get_accuracy_mean(acc_type=acc_type) * 100.,
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


class BenchmarksDB:
    def __init__(self, load_path=None, plot_path=None):
        self.head = dict()
        self.file_path = load_path
        self.plot_path = plot_path
        self.min_y_axis = None

        if self.file_path is not None:
            self.load(self.file_path)

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

    def _save_and_show(self, fig, path, dataset_name, class_name, plot_name, show):
        path = get_path(path, self.plot_path)
        if path is not None:
            if dataset_name is not None and class_name is not None:
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
                return False
            for j, model in enumerate(models):
                if model in tested_models:
                    stat = self.head[dataset][class_name][experiment][model]
                    values[model][0].append(stat.get_accuracy_mean())
                    values[model][1].append(stat.get_accuracy_error_min())
                    values[model][2].append(stat.get_accuracy_error_max())
                else:
                    values[model][0].append(0)
                    values[model][1].append(0)
                    values[model][2].append(0)

        fig = self._plot_histograms('Exp. Class: "' + class_name + '"\nExperiment: "' + experiment + '"', values,
                                    datasets)
        self._save_and_show(fig, path, None, class_name, "datasets summary " + class_name + " " + experiment, show)
        return True

    def plot_lines_by_key(self, dataset_name: str, class_name: str, key: str = 'units',
                          path: str = None, show: bool = True):
        try:
            exp_tree = self.head[dataset_name][class_name]
        except KeyError:
            print("Benchmarks not found. %s %s" % (dataset_name, class_name))
            return False
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
                    values[model][2].append(stat.get_accuracy_error_min())
                    values[model][3].append(stat.get_accuracy_error_max())

        fig, ax = plt.subplots()
        for model, val in values.items():
            sort_list = list(zip(*sorted(zip(*val))))
            (_, caps, _) = ax.errorbar(sort_list[0], sort_list[1], yerr=[sort_list[2], sort_list[3]], marker='o',
                                       label=model, markersize=3, capsize=5)
            for cap in caps:
                cap.set_markeredgewidth(1)

        y_min = round_down(
            self.get_min_acc())  # round_down(np.min([np.subtract(x[1], x[2]) for _, x in values.items()]))
        y_max = 1.00  # round_up(np.max([np.add(x[1], x[3]) for _, x in values.items()]))

        ax.set_ylabel('Accuracy')
        ax.set_ylim([y_min, y_max])
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        locator = (y_max - y_min) / 5.
        ax.yaxis.set_major_locator(MultipleLocator(locator))
        ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
        ax.set_xlabel(key.capitalize())
        ax.set_title('Dataset: "' + dataset_name + '"\nExp. Class: "' + class_name + '"\nhp: ' + key)

        ax.legend(loc='lower right')

        self._save_and_show(fig, path, dataset_name, class_name, "summary plots", show)
        return True

    def plot_summary_histogram(self, dataset_name: str, class_name: str,
                               path: str = None, show: bool = True):
        try:
            exp_tree = self.head[dataset_name][class_name]
        except KeyError:
            print("Benchmarks not found. %s %s" % (dataset_name, class_name))
            return False
        experiments = get_sorted_keys(exp_tree)
        models = get_models_names(exp_tree)

        values = {name: ([], [], []) for name in models}
        for i, experiment in enumerate(experiments):
            tested_models = exp_tree[experiment].keys()
            for j, model in enumerate(models):
                if model in tested_models:
                    stat = exp_tree[experiment][model]
                    values[model][0].append(stat.get_accuracy_mean())
                    values[model][1].append(stat.get_accuracy_error_min())
                    values[model][2].append(stat.get_accuracy_error_max())
                else:
                    values[model][0].append(0)
                    values[model][1].append(0)
                    values[model][2].append(0)

        fig = self._plot_histograms('Dataset: "' + dataset_name + '" Exp. Class: "' + class_name + '"', values,
                                    experiments)
        self._save_and_show(fig, path, dataset_name, class_name, "summary histograms", show)
        return True

    def plot_by_experiment(self, dataset_name: str, class_name: str,
                           path: str = None, show: bool = True):
        try:
            exp_tree = self.head[dataset_name][class_name]
        except KeyError:
            print("Benchmarks not found. %s %s" % (dataset_name, class_name))
            return False

        labels = get_models_names(exp_tree)
        for exp_name, benchmarks in exp_tree.items():
            acc_mean = [benchmarks[label].get_accuracy_mean() if benchmarks.get(label) is not None else 0.
                        for label in labels]
            acc_std = [benchmarks[label].get_accuracy_std() if benchmarks.get(label) is not None else 0.
                       for label in labels]

            fig = self._plot_histogram(exp_name, acc_mean, acc_std, labels)
            self._save_and_show(fig, path, dataset_name, class_name, exp_name, show)
        return True

    # Not have much sense
    def plot_by_model(self, dataset_name: str, class_name: str,
                      path: str = None, show: bool = True):
        try:
            exp_tree = self.head[dataset_name][class_name]
        except KeyError:
            print("Benchmarks not found. %s %s" % (dataset_name, class_name))
            return False
        models = get_models_names(exp_tree)

        for model in models:
            experiments = get_sorted_keys(exp_tree)
            acc_mean = [exp_tree[exp][model].get_accuracy_mean() if exp_tree[exp].get(model) is not None else None
                        for exp in experiments]
            acc_min = [exp_tree[exp][model].get_accuracy_error_min() if exp_tree[exp].get(model) is not None else None
                       for exp in experiments]
            acc_max = [exp_tree[exp][model].get_accuracy_error_max() if exp_tree[exp].get(model) is not None else None
                       for exp in experiments]

            fig = self._plot_histogram(model, acc_mean, [acc_min, acc_max], experiments)
            self._save_and_show(fig, path, dataset_name, class_name, model, show)
        return True

    def plot_summary_table(self, dataset_name: str, class_name: str,
                           path: str = None, show: bool = True):
        try:
            exp_tree = self.head[dataset_name][class_name]
        except KeyError:
            print("Benchmarks not found. %s %s" % (dataset_name, class_name))
            return False

        experiments = get_sorted_keys(exp_tree)
        models = get_models_names(exp_tree)

        body = []
        color = []
        for model in models:
            row = []
            color_row = []
            for experiment in experiments:
                try:
                    stat = exp_tree[experiment][model]
                    row.append("{:.2f}% / {}".format(stat.score * 100., stat.get_accuracy_str(Statistic.TEST))
                               # stat.get_accuracy_str(Statistic.TRAIN) + " / " + stat.get_accuracy_str(Statistic.VALIDATION)
                               )
                    color_row.append(stat.score)  # stat.get_accuracy_mean(Statistic.TEST))
                except KeyError as _:
                    row.append('-')
                    color_row.append(None)
            body.append(row)
            color.append(color_row)

        fig = plot_table('Dataset: "' + dataset_name + '"\nExp. Class: "' + class_name + '"', experiments, models, body,
                         best_worse=color)
        self._save_and_show(fig, path, dataset_name, class_name, "summary table", show)
        return True

    def plot_global_summary_table(self, path: str = None, show: bool = True):
        datasets = len(self.head)
        ncols = 2
        fig = plt.figure(figsize=(10, 10), dpi=500)
        fig.patch.set_visible(False)
        gs = gridspec.GridSpec(nrows=int(datasets / ncols), ncols=ncols, figure=fig, hspace=0.2)

        models = self.list_models()
        plot_x = ncols
        for grid, (ds_name, classes) in zip(list(gs), self.head.items()):
            ax = plt.Subplot(fig, grid)
            ax.set_title(ds_name, fontsize=12, y=1.075, )
            ax.axis('off')
            ax.axis('tight')
            inner = gridspec.GridSpecFromSubplotSpec(nrows=len(classes), ncols=1, subplot_spec=grid, hspace=0.2)
            #print(ds_name)
            for sub_grid, (class_name, experiments) in zip(inner, classes.items()):
                #print("\t", class_name, ":")
                sub_ax = plt.Subplot(fig, sub_grid)
                sub_ax.set_title(class_name, fontsize=10, color='grey', loc='left', style='italic')
                sub_ax.axis('off')
                sub_ax.axis('tight')

                best = get_best_models(experiments, models)

                body = [["{:.2f}%".format(best[m][0] * 100), best[m][1].get_accuracy_str()]
                        if best[m][1] is not None else
                        ["-", "-"]
                        for m in models]
                x_labels = ["Validation Acc", "Test acc"]

                the_table = sub_ax.table(cellText=body, rowLabels=models, colLabels=x_labels,
                                             loc='center', cellLoc='center', edges='horizontal', fontsize=7.)
                the_table.auto_set_font_size(False)
                the_table.auto_set_column_width(col=list(range(len(x_labels))))
                cells = the_table.get_celld()
                for cell in cells.values():
                    cell.set_height(0.2)
                for c in range(len(x_labels)):
                    cells[0, c].visible_edges = 'B'

                try:
                    best_row = np.argmax([best[m][0] for m in models], axis=0)
                    for i in range(4):
                        cells[best_row + 1, i].get_text().set_color('#008000')
                except Exception as _:
                    pass

                fig.add_subplot(sub_ax)
            fig.add_subplot(ax)
        self._save_and_show(fig, path, None, None, "summary table", show)
        return True

    # X is hp Y is experiments
    def plot_hp_table_by_model(self, dataset_name: str, class_name: str, model: str, keys: [],
                               path: str = None, show: bool = True):
        exp_tree = self.head[dataset_name][class_name]
        experiments = get_sorted_keys(exp_tree)

        hp = set()
        for experiment in experiments:
            try:
                stat = exp_tree[experiment][model]
                hp.update(stat.hyperparameters.values.keys())
            except Exception as _:
                pass
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
                value = None
                try:
                    stat = exp_tree[experiment][model]
                    value = stat.hyperparameters.values.get(key)
                except Exception as _:
                    pass
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
        self._save_and_show(fig, path, dataset_name, class_name, "hp_table_by_" + model, show)

    # X is hp Y is model_name
    def plot_hp_table_by_experiment(self, dataset_name: str, class_name: str, experiment: str, keys: [],
                                    path: str = None, show: bool = True):
        try:
            models_tree = self.head[dataset_name][class_name][experiment]
        except KeyError as _:
            # print("Error: ", class_name, experiment_name, "Not found")
            return

        models = get_sorted_keys(models_tree, sort=models_sort)

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
        self._save_and_show(fig, path, dataset_name, class_name, "hp_table_by_" + experiment, show)

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

        # Spit dataset_name names
        x_new = []
        for label in x_labels:
            new = "" + label[0]
            for c in label[1:]:
                if c.isupper():
                    new += '\n'
                new += c
            x_new.append(new)

        y_min = round_down(self.get_min_acc())  # round_down(np.min([np.subtract(x[0], x[1]) for _, x in data.items()]))
        y_max = 1.00  # round_up(np.max([np.add(x[0], x[2]) for _, x in data.items()]))

        ax.set_xticks(ticks=x, labels=x_new)
        ax.set_ylabel('Accuracy')

        ax.set_ylim([y_min, y_max])
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        locator = (y_max - y_min) / 5.
        ax.yaxis.set_major_locator(MultipleLocator(locator))
        ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
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
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.set_ylim([round_down(self.get_min_acc()), 1.0])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(title)
        ax.yaxis.grid(True)

        plt.tight_layout()
        return fig


def get_best_models(experiments, list_models):
    best = {name: (0., None, None) for name in list_models}
    for experiment_name, models in experiments.items():
        for model_name, stat in models.items():
            if best[model_name][0] < stat.score:
                best[model_name] = (stat.score, stat, experiment_name)
    return best


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


def get_matching_keys(exps, keys):
    ret = []
    for hp in keys:
        if any(map(hp.__contains__, exps)):
            ret.append(hp)
    return ret


def plot_model(model, names, title=None, plot_bias=False, path=None, x_size=10, show=True):
    import tensorflow as tf
    dataset_name, class_name, experiment_name, model_name = names
    if not model.build:
        raise Exception("Train the model_name first")

    kernel_m = model.reservoir.layers[1].cell.kernel
    rec_kernel_m = model.reservoir.layers[1].cell.recurrent_kernel
    readout_m = model.readout.layers[0].weights[0]

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
    if model.use_bias and plot_bias:
        bias_m = model.reservoir.layers[1].cell.bias
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

    if model.use_bias and plot_bias:
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


def plot_table(title, x_labels, y_labels, body, best_worse=None, show_x=True):
    width = len(x_labels) + 1
    height = np.floor(len(y_labels) / 2) + 0.5
    if height < 1.5:
        height = 2
    fig, ax = plt.subplots(figsize=(width, height))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title)
    the_table = ax.table(cellText=body, rowLabels=y_labels, colLabels=x_labels,
                         loc='center', cellLoc='center', edges='horizontal', fontsize=10.)
    the_table.auto_set_font_size(False)
    the_table.auto_set_column_width(col=list(range(len(x_labels))))
    cells = the_table.get_celld()
    for cell in cells.values():
        cell.set_height(0.2)

    for c in range(len(x_labels)):
        cells[0, c].visible_edges = 'B'

    if best_worse is not None:
        min_color = '#db222a'
        max_color = '#008000'
        # argmin = np.argmin(best_worse, axis=1)
        try:
            argmax = np.argmax(best_worse, axis=1)
            for i in range(len(argmax)):
                # cells[(argmin[i]+1, i)].get_text().set_color(min_color)
                cells[(i + 1, argmax[i])].get_text().set_color(max_color)
        except Exception as _:
            pass
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
    tmp.sort(key=models_sort)
    return tmp


def get_sorted_keys(tree, sort=natural_sort):
    tmp = tree
    if isinstance(tree, dict):
        tmp = list(tree.keys())
    tmp.sort(key=sort)
    return tmp


def lower_and_replace(string: str):
    return string.lower().replace(" ", "_")


def round_up(number: float, decimals: int = 1):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def round_down(number: float, decimals: int = 1):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor
