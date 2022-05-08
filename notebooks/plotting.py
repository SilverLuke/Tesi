import math
import re
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from benchmarks import Statistic, get_sorted_keys, models_sort, get_models_names, get_path, natural_sort


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


def add_stat(values, name, stat):
    """
    :param values: the dict
    :param name: model name
    :param stat: stat of the model
    :return: a tuple (mean, min, max, stat obj)
    """
    if stat is not None:
        values[name][0].append(stat.get_accuracy_mean() * 100)
        values[name][1].append(stat.get_accuracy_error_min() * 100)
        values[name][2].append(stat.get_accuracy_error_max() * 100)
        values[name][3].append(stat)
    else:
        values[name][0].append(np.nan)
        values[name][1].append(np.nan)
        values[name][2].append(np.nan)
        values[name][3].append(np.nan)


def add_stat_box(values, name, stat):
    """
    :param values: the dict
    :param name: model name
    :param stat: stat of the model
    :return: a tuple (train accuracies, stat obj)
    """
    if stat is not None:
        values[name][0].append(np.array(stat.get_accuracy()) * 100)
        values[name][1].append(stat)
    else:
        values[name][0].append(np.nan)
        values[name][1].append(np.nan)


def plot_datasets_summary(ax, dataset, classes, models, plot_x=True, plot_y=True, legend_pos=None):
    width = 0.30
    margin = 0.02
    zoom = 3
    total = width + margin
    x_labels = ["Reference", "Multiple S.R.", "Single S.R."]
    x = np.arange(len(x_labels), dtype=float)
    x[1] = 0.84

    values = {name: ([], [], [], []) for name in models}
    add_stat(values, 'ESN', classes['Reference']['Units 100']['ESN'])
    add_stat(values, 'IRESN', classes['Single S.R.']['Units 100']['IRESN'])
    add_stat(values, 'IIRESN', classes['Single S.R.']['Units 100']['IIRESN'])
    add_stat(values, 'IIRESNvsr', classes['Single S.R.']['Units 100']['IIRESNvsr'])
    add_stat(values, 'IRESN', classes['Multiple S.R.']['Units 100']['IRESN'])
    add_stat(values, 'IIRESN', classes['Multiple S.R.']['Units 100']['IIRESN'])
    add_stat(values, 'IIRESNvsr', classes['Multiple S.R.']['Units 100']['IIRESNvsr'])

    ax.bar(x[0], values['ESN'][0],  # yerr=[values['ESN'][1], values['ESN'][2]],
           label='ESN', width=width, align='center', capsize=3)
    ax.bar(x[1:] - total, values['IRESN'][0],  # yerr=[values['IRESN'][1], values['IRESN'][2]],
           label='IRESN', width=width, align='center', capsize=3)
    ax.bar(x[1:], values['IIRESN'][0],  # yerr=[values['IIRESN'][1], values['IIRESN'][2]],
           label='IIRESN', width=width, align='center', capsize=3)
    ax.bar(x[1:] + total, values['IIRESNvsr'][0],  # yerr=[values['IIRESNvsr'][1], values['IIRESNvsr'][2]],
           label='IIRESNvsr', width=width, align='center', capsize=3)

    if plot_x:
        ax.set_xticks(ticks=x, labels=x_labels, rotation=15)
    else:
        ax.get_xaxis().set_visible(False)
    if plot_y:
        ax.set_ylabel('Accuracy')
    ax.get_yaxis().set_visible(False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    min_acc = []
    mean_acc = []
    max_acc = []
    for a, x in values.items():
        for acc in x[3]:
            min_acc.append(acc.get_accuracy_min() * 100)
            mean_acc.append(acc.get_accuracy_mean() * 100)
            max_acc.append(acc.get_accuracy_max() * 100)

    y_min = np.min(mean_acc) - zoom
    y_max = np.max(mean_acc) + zoom
    ax.set_ylim([y_min, y_max])

    locator = (y_max - y_min) / 5.
    ax.yaxis.set_major_locator(MultipleLocator(locator))
    ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
    ax.set_title(dataset)
    if legend_pos is not None:
        ax.legend(loc=legend_pos)
    return ax


class Plotter:
    ESN = "C0"
    IRESN = "C1"
    IIRESN = "C2"
    MEAN = "C3"
    MEDIAN = "C4"

    def __init__(self, db, plot_path):
        self.db = db
        self.path = plot_path

    def _save_and_show(self, fig, path, plot_name, show=False, dataset_name=None, class_name=None):
        path = get_path(path, self.path)
        if path is not None:
            if dataset_name is not None:
                path = os.path.join(path, dataset_name)
                if class_name is not None:
                    path = os.path.join(path, lower_and_replace(class_name))
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, lower_and_replace(plot_name + ".png"))
            print(path)
            fig.savefig(path, format='PNG', dpi=100, bbox_inches='tight')
        if show:
            fig.show()
        plt.close(fig)

    """
    'Old' definition IRESN IIRESN IIRESNvsr
    def histograms_summary(self, path=None, show=False):
        datasets = len(self.db.head)
        models = self.db.list_models()
        ncols = 3
        fig = plt.figure(figsize=(10, 5), dpi=500)
        gs = gridspec.GridSpec(nrows=math.ceil(datasets / ncols), ncols=ncols, figure=fig)
        for i, (cell, (ds_name, classes)) in enumerate(zip(list(gs), self.db.head.items())):
            plot_x = i // ncols == 1
            plot_y = i == 0 or i == 3
            ax = plt.Subplot(fig, cell)
            ax = plot_datasets_summary(ax, ds_name, classes, models, plot_x, plot_y)
            fig.add_subplot(ax)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
        self._save_and_show(fig, path, "summary_plot")

    def latex_table(self, path=None):
        def add_model(model, stat):
            return "%s & %s & $%.2f \%% $ & %s & $%.2f \%%$ & $ %.2f\%% $ \\tabularnewline\n" % \
                   (model,
                    stat.get_accuracy_latex(Statistic.TRAIN),
                    stat.score * 100,
                    stat.get_accuracy_latex(Statistic.TEST),
                    stat.get_accuracy_min(Statistic.TEST) * 100,
                    stat.get_accuracy_max(Statistic.TEST) * 100
                    )

        def sub_table(dataset, columns):
            table = ""
            # table += "\\hline\n"
            # table += "\\rowcolor{lightgray}\n"
            # table += "\\multicolumn{%i}{c}{%s}\\tabularnewline\n" % (columns, "Riferimento")
            # table += "\\hline\n"
            stat = dataset["Reference"]["Units 100"]["ESN"]
            table += add_model("ESN", stat)
            table += "\\hline\n"
            table += "\\rowcolor{lightgray!50}\n"
            table += "\\multicolumn{%i}{c}{%s}\\tabularnewline\n" % (columns, "Single Spectral Radius")
            table += "\\hline\n"
            stat = dataset["Single S.R."]["Units 100"]["IRESN"]
            table += add_model("IRESN", stat)
            stat = dataset["Single S.R."]["Units 100"]["IIRESN"]
            table += add_model("IIRESN", stat)
            stat = dataset["Single S.R."]["Units 100"]["IIRESNvsr"]
            table += add_model("IIRESNvsr", stat)
            table += "\\hline\n"
            table += "\\rowcolor{lightgray!50}\n"
            table += "\\multicolumn{%i}{c}{%s}\\tabularnewline\n" % (columns, "Multiple Spectral Radius")
            table += "\\hline\n"
            stat = dataset["Multiple S.R."]["Units 100"]["IRESN"]
            table += add_model("IRESN", stat)
            stat = dataset["Multiple S.R."]["Units 100"]["IIRESN"]
            table += add_model("IIRESN", stat)
            stat = dataset["Multiple S.R."]["Units 100"]["IIRESNvsr"]
            table += add_model("IIRESNvsr", stat)
            return table

        datasets = self.db.list_datasets()
        legend = ["Modello", "Train set $\pm \sigma$", "Validation score", "Test set $\pm \sigma$", "Test set min",
                  "Test set max"]
        columns = len(legend)
        table = "\\begin{table}[ht]\n \\centering\n" \
                "\\resizebox{\\textwidth}{!}{%%\n" \
                "\\begin{tabular}{%s}\n" % ("c" * columns)

        for i in legend[:-1]:
            table += i + " & "
        table += legend[-1]
        table += "\\tabularnewline\n"

        for dataset in datasets:
            table += "\hline\n"
            table += "\\rowcolor{lightgray}\n"
            table += "\\multicolumn{%i}{c}{\\textbf{%s}}\\tabularnewline\n" % (columns, dataset)
            table += "\\hline\n"
            table += sub_table(self.db[dataset], columns)
        table += "\\end{tabular}}\n\\end{table}"
        path = get_path(path, self.path)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, "table.tex")
        print(path)
        with open(path, "w") as f:
            f.write(table)
    """

    def latex_table(self, dataset_order=None, classes_order=None, path=None):
        def add_model(model, stat):
            if stat is not None:
                return "%s & %s & $%.2f \%% $ & %s & $%.2f \%%$ & $ %.2f\%% $ \\tabularnewline\n" % \
                       (model,
                        stat.get_accuracy_latex(Statistic.TRAIN),
                        stat.get_score() * 100,
                        stat.get_accuracy_latex(Statistic.TEST),
                        stat.get_accuracy_min(Statistic.TEST) * 100,
                        stat.get_accuracy_max(Statistic.TEST) * 100
                        )
            else:
                return "- & - & - & - & - & - \\tabularnewline\n"

        def sub_table(dataset, columns):
            table = ""
            # table += "\\hline\n"
            # table += "\\rowcolor{lightgray}\n"
            # table += "\\multicolumn{%i}{c}{%s}\\tabularnewline\n" % (columns, "Riferimento")
            # table += "\\hline\n"

            stat = dataset["Reference"]["Units 100"]["ESN"]
            table += add_model("ESN", stat)


            if classes_order is None:
                classes = dataset.keys()
            else:
                classes = classes_order

            for exp in classes:
                table += "\\hline\n"
                table += "\\rowcolor{lightgray!50}\n"
                table += "\\multicolumn{%i}{c}{%s}\\tabularnewline\n" % (columns, exp)
                table += "\\hline\n"
                for (model_name, stat) in dataset[exp]["Units 100"].items():
                    table += add_model(model_name, stat)

            return table

        if dataset_order is None:
            datasets = self.db.list_datasets()
        else:
            datasets = dataset_order
        legend = ["Modello", "Train set $\pm \sigma$", "Validation score", "Test set $\pm \sigma$", "Test set min",
                  "Test set max"]
        columns = len(legend)
        table = ""

        for dataset_name in datasets:
            try:
                dataset = self.db[dataset_name]
            except KeyError:
                continue
            table += "\\begin{table}[ht]\n \\centering\n" \
                     "\\resizebox{\\textwidth}{!}{%%\n" \
                     "\\begin{tabular}{%s}\n" % ("c" * columns)

            for i in legend[:-1]:  # :-1 do not put the last element of the legend in the loop to avoid leading & at the end
                table += i + " & "
            table += legend[-1]
            table += "\\tabularnewline\n"
            table += "\hline\n"
            table += "\\rowcolor{lightgray}\n"
            table += "\\multicolumn{%i}{c}{\\textbf{%s}}\\tabularnewline\n" % (columns, dataset_name)
            table += "\\hline\n"
            table += sub_table(dataset, columns)
            table += "\\end{tabular}}\n\\end{table}"

        path = get_path(path, self.path)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, "table.tex")
        print(path)
        with open(path, "w") as f:
            f.write(table)

    def histogram_summary(self, dataset_order=None, classes_order=None, ncol: int = 1, path: str = None, show: bool = False):
        if dataset_order is None:
            datasets = self.db.head.keys()
        else:
            datasets = dataset_order

        fig = plt.figure(figsize=(10, 20), dpi=500)
        plt.axis('off')
        gs = gridspec.GridSpec(nrows=math.ceil(len(datasets) / ncol), ncols=ncol, figure=fig, hspace=0.4)

        for i, (cell, ds_name) in enumerate(zip(list(gs), datasets)):
            plot_x = i // ncol == 1
            plot_y = i == 0 or i == 3
            ax = plt.Subplot(fig, cell)
            ax = self.plot_histogram_by_dataset(ax, ds_name, plot_x=plot_x, plot_y=plot_y)
            fig.add_subplot(ax)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.085))
        self._save_and_show(fig, path, "histograms", show=show)

    def histogram_by_dataset(self, dataset, path: str = None, show: bool = False):
        fig = plt.figure(figsize=(10, 4.2), dpi=500)
        plt.axis('off')
        gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig, hspace=0.4)

        ax = plt.Subplot(fig, gs[0])
        ax = self.plot_histogram_by_dataset(ax, dataset, plot_x=True, plot_y=True)
        fig.add_subplot(ax)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.07))
        self._save_and_show(fig, path, "histogram", dataset_name=dataset, show=show)

    def plot_histogram_by_dataset(self, ax, dataset, classes_order=None, plot_x=False, plot_y=False):
        models = self.db.list_models()
        classes = self.db[dataset]

        width = 0.30
        margin = 0.02
        zoom = 1.5
        total = width + margin
        if classes_order is None:
            x_labels = dataset.keys()
        else:
            x_labels = classes_order
        x = np.arange(len(x_labels), dtype=float)
        #(mean, min, max, stat obj)
        values = {name: ([], [], [], []) for name in models}
        add_stat(values, 'ESN', classes['Reference']['Units 100']['ESN'])
        for exp in x_labels[1:]:
            for model in models[1:]:
                try:
                    add_stat(values, model, classes[exp]['Units 100'][model])
                except KeyError:
                    add_stat(values, model, None)

        ax.bar(x[0], values['ESN'][0], label='ESN', width=width, align='center', color=self.ESN)
        ax.bar(x[1:] - total / 2, values['IRESN'][0], label='IRESN', width=width, align='center', color=self.IRESN)
        ax.bar(x[1:] + total / 2, values['IIRESN'][0], label='IIRESN', width=width, align='center', color=self.IIRESN)
        ax.axhline(values['ESN'][0][0], color='gray', linestyle='dashed', linewidth=0.5, zorder=-1)
        ax.set_xticks(ticks=x, labels=["Reference",
                                       "Single SR\nSingle IS",
                                       "Single SR\nSingle IS\nVSR",
                                       "Multiple SR\nSingle IS",
                                       "Multiple SR\nSingle IS\nVSR",
                                       "Multiple SR\nMultiple IS",
                                       "Multiple SR\nMultiple IS\nVSR"], rotation=0, horizontalalignment='center',
                      fontsize='x-small')
        plt.yticks(fontsize='x-small')

        min_acc = []
        mean_acc = []
        max_acc = []
        for a, x in values.items():
            for acc in x[3]:
                if acc is np.nan:
                    continue
                min_acc.append(acc.get_accuracy_min() * 100)
                mean_acc.append(acc.get_accuracy_mean() * 100)
                max_acc.append(acc.get_accuracy_max() * 100)

        y_min = np.min(mean_acc) - zoom
        y_max = np.max(mean_acc) + zoom
        ax.set_ylim([y_min, y_max])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        locator = (y_max - y_min) / 5.
        ax.yaxis.set_major_locator(MultipleLocator(locator))
        ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
        ax.set_title(dataset)
        ax.set_ylabel('Accuracy')
        return ax

    def box_summary(self, order=None, ncol: int = 1, path: str = None, show: bool = False):
        if order is None:
            datasets = self.db.head.keys()
        else:
            datasets = order

        fig = plt.figure(figsize=(10, 25), dpi=500)
        plt.axis('off')
        gs = gridspec.GridSpec(nrows=math.ceil(len(datasets) / ncol), ncols=ncol, figure=fig, hspace=0.4)

        for i, (cell, ds_name) in enumerate(zip(list(gs), datasets)):
            plot_x = i // ncol == 1
            plot_y = i == 0 or i == 3
            ax = plt.Subplot(fig, cell)
            ax, bp1, bp2, bp3 = self.plot_box_by_dataset(ax, ds_name, plot_x=plot_x, plot_y=plot_y)
            fig.add_subplot(ax)
        ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp1['medians'][0], bp1['means'][0]],
                  ['ESN', 'IRESN', 'IIRESN', 'median', 'mean'],
                  loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.425))
        self._save_and_show(fig, path, "boxs", show=show)

    def box_by_dataset(self, dataset, path: str = None, show: bool = False):
        fig = plt.figure(figsize=(10, 4.2), dpi=500)
        plt.axis('off')
        gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig, hspace=0.4)

        ax = plt.Subplot(fig, gs[0])
        ax, bp1, bp2, bp3 = self.plot_box_by_dataset(ax, dataset, plot_x=True, plot_y=True)
        fig.add_subplot(ax)

        ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp1['medians'][0], bp1['means'][0]],
              ['ESN', 'IRESN', 'IIRESN', 'median', 'mean'],
              loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.07))
        self._save_and_show(fig, path, "box", dataset_name=dataset, show=show)

    def plot_box_by_dataset(self, ax, dataset, plot_x=False, plot_y=False):
        models = self.db.list_models()
        classes = self.db[dataset]

        width = 0.30
        margin = 0.02
        total = margin + width
        x_labels = ["Reference",
                    "Single SR Single IS",
                    "Single SR Single IS VSR",
                    "Multiple SR Single IS",
                    "Multiple SR Single IS VSR",
                    "Multiple SR Multiple IS",
                    "Multiple SR Multiple IS VSR"]
        x = np.arange(len(x_labels), dtype=float)

        values = {name: ([], []) for name in models}
        add_stat_box(values, 'ESN', classes['Reference']['Units 100']['ESN'])
        for exp in x_labels[1:]:
            for model in models[1:]:
                try:
                    add_stat_box(values, model, classes[exp]['Units 100'][model])
                except KeyError:
                    add_stat_box(values, model, None)

        bp1 = ax.boxplot(values['ESN'][0], positions=[0], widths=width,
                         showmeans=True, meanline=True, patch_artist=True, boxprops=dict(facecolor=self.ESN),
                         medianprops=dict(color=self.MEDIAN, linestyle='-'),
                         meanprops=dict(color=self.MEAN, linestyle='-'))
        bp2 = ax.boxplot(values['IRESN'][0],
                         positions=np.arange(len(values['IRESN'][0]), dtype=float) + (1. - total / 2.), widths=width,
                         showmeans=True, meanline=True, patch_artist=True, boxprops=dict(facecolor=self.IRESN),
                         medianprops=dict(color=self.MEDIAN, linestyle='-'),
                         meanprops=dict(color=self.MEAN, linestyle='-'))
        bp3 = ax.boxplot(values['IIRESN'][0],
                         positions=np.arange(len(values['IIRESN'][0]), dtype=float) + (1. + total / 2.), widths=width,
                         showmeans=True, meanline=True, patch_artist=True, boxprops=dict(facecolor=self.IIRESN),
                         medianprops=dict(color=self.MEDIAN, linestyle='-'),
                         meanprops=dict(color=self.MEAN, linestyle='-'))
        ax.axhline(values['ESN'][1][0].get_accuracy_mean() * 100, color='gray', linestyle='dashed', linewidth=0.5,
                   zorder=-1)

        ax.set_xticks(ticks=x, labels=["Reference",
                                       "Single SR\nSingle IS",
                                       "Single SR\nSingle IS\nVSR",
                                       "Multiple SR\nSingle IS",
                                       "Multiple SR\nSingle IS\nVSR",
                                       "Multiple SR\nMultiple IS",
                                       "Multiple SR\nMultiple IS\nVSR"],
                      rotation=0, horizontalalignment='center', fontsize='x-small')
        ax.set_title(dataset)
        ax.set_ylabel('Accuracy')
        return ax, bp1, bp2, bp3

    def hp_table_by_model(self, dataset_name: str, model: str, keys: [],
                          path: str = None, show: bool = True):
        exps_tree = self.db[dataset_name]

        def get_experiments(exp_tree, model):
            tmp = set()
            for exp_name, sub_exps in exp_tree.items():
                if model in sub_exps['Units 100'].keys():
                    tmp.add(exp_name)
            tmp = list(tmp)
            tmp.sort(key=natural_sort)
            return tmp

        experiments = get_experiments(exps_tree, model)

        hps = set()
        for experiment in experiments:
            stat = exps_tree[experiment]['Units 100'][model]
            hps.update(stat.get_hyperparameters())

        hp_keys = get_matching_keys(keys, list(hps))
        hp_keys = get_sorted_keys(hp_keys)

        body = []
        for experiment in experiments:
            row = []
            for key in hp_keys:
                stat = exps_tree[experiment]['Units 100'][model]
                value = stat.get_hyperparameter_value(key)
                if isinstance(value, float):
                    row.append(float("{:0.3f}".format(value)))
                elif isinstance(value, int):
                    row.append(value)
                elif value is None:
                    row.append("-")
                else:
                    print(type(value))
            body.append(row)

        fig = plot_table('Dataset: "' + dataset_name + '"\nModel: "' + model + '"',
                         hp_keys,
                         experiments,
                         body)
        self._save_and_show(fig, path, "hps " + model, dataset_name=dataset_name, show=False)


"""
    def plot_lines_by_key(self, dataset_name: str, class_name: str, key: str = 'units',
                          path: str = None, show: bool = True):
        try:
            exp_tree = self.db.head[dataset_name][class_name]
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
                    values[model][0].append(stat.keras_hps.values.get(key))
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

        y_min = round_down(self.db.get_min_acc())
        # round_down(np.min([np.subtract(x[1], x[2]) for _, x in values.items()]))
        y_max = 1.00  # round_up(np.max([np.add(x[1], x[3]) for _, x in values.items()]))

        ax.set_ylabel('Accuracy')
        ax.set_ylim([y_min, y_max])
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        locator = (y_max - y_min) / 5.
        ax.yaxis.set_major_locator(MultipleLocator(locator))
        ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
        ax.set_xlabel(key.capitalize())
        ax.set_title('Dataset: "' + dataset_name + '"\nExp. Class: "' + class_name + '"\nself: ' + key)

        ax.legend(loc='lower right')

        self._save_and_show(fig, path, dataset_name, class_name, "summary plots", show)
        return True

    def plot_summary_histogram(self, dataset_name: str, class_name: str,
                               path: str = None, show: bool = True):
        try:
            exp_tree = self.db.head[dataset_name][class_name]
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
            exp_tree = self.db.head[dataset_name][class_name]
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
            exp_tree = self.db.head[dataset_name][class_name]
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

    def plot_summary_table_by_class(self, dataset_name: str, class_name: str,
                                    path: str = None, show: bool = True):
        try:
            exp_tree = self.db.head[dataset_name][class_name]
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
                               # stat.get_accuracy_str(Statistic.TRAIN) + " /
                               # " + stat.get_accuracy_str(Statistic.VALIDATION)
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

    def plot_summary_table_by_dataset(self, dataset_name: str, path: str = None, show: bool = True):
        try:
            classes_tree = self.db.head[dataset_name]
        except KeyError:
            print("Benchmarks not found. %s" % dataset_name)
            return False

        x_labels = ["Test accuracy"]
        cell_height = 0.25
        fig = plt.figure(figsize=(3, 5), dpi=500)
        fig.patch.set_visible(False)
        plt.title(dataset_name, fontsize=12, y=1.075, )
        plt.axis('off')
        plt.axis('tight')

        models = self.db.list_models()
        models.remove('ESN')

        ratio = [2, 3, 3]
        total = sum(ratio)
        # Normalize the partition vector now sum(partitions) == 1.
        ratio = list(map(lambda _x: 0 if total == 0 else _x / total, ratio))

        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, height_ratios=ratio)
        ref_ax = plt.Subplot(fig, gs[0])
        ref_ax.set_title("Reference", fontsize=10, color='grey', loc='left', style='italic')
        ref_ax.axis('off')
        ref_ax.axis('tight')

        reference = [
            # classes_tree['Multiple S.R.']['Units 50']['ESN'],
            classes_tree['Multiple S.R.']['Units 100']['ESN'],
            # classes_tree['Multiple S.R.']['Units 250']['ESN'],
        ]

        i = np.argmax([r.score for r in reference])

        body = [[reference[i].get_accuracy_str()]]
        the_table = ref_ax.table(cellText=body, rowLabels=["ESN"], colLabels=x_labels,
                                 loc='center', cellLoc='center', edges='horizontal', fontsize=7.)
        the_table.auto_set_font_size(False)
        the_table.auto_set_column_width(col=list(range(len(x_labels))))
        cells = the_table.get_celld()
        for cell in cells.values():
            cell.set_height(cell_height + 0.05)
        for c in range(len(x_labels)):
            cells[0, c].visible_edges = 'B'

        for sub_grid, (class_name, experiments) in zip([gs[1], gs[2]], classes_tree.items()):
            # print("\t", class_name, ":")
            sub_ax = plt.Subplot(fig, sub_grid)
            sub_ax.set_title(class_name, fontsize=10, color='grey', loc='left', style='italic')
            sub_ax.axis('off')
            sub_ax.axis('tight')

            best = get_best_models(experiments, models)

            body = [[best[m][1].get_accuracy_str()]
                    if best[m][1] is not None else
                    ["-"]
                    for m in models]

            the_table = sub_ax.table(cellText=body, rowLabels=models, colLabels=x_labels,
                                     loc='center', cellLoc='center', edges='horizontal', fontsize=7.)
            the_table.auto_set_font_size(False)
            the_table.auto_set_column_width(col=list(range(len(x_labels))))
            cells = the_table.get_celld()
            for cell in cells.values():
                cell.set_height(cell_height)
            for c in range(len(x_labels)):
                cells[0, c].visible_edges = 'B'

            try:
                best_row = np.argmax([best[m][1].get_accuracy_mean() for m in models], axis=0)
                cells[best_row + 1, 0].get_text().set_color('#008000')
            except IndexError or KeyError:
                import traceback
                traceback.print_exc()

            fig.add_subplot(sub_ax, sharex=ref_ax)
        fig.add_subplot(ref_ax)

        self._save_and_show(fig, path, dataset_name, None, "summary " + dataset_name, show)
        return fig

    def plot_global_summary_table(self, path: str = None, show: bool = True):
        datasets = len(self.db.head)
        ncols = 3
        x_labels = ["Accuratezza"]
        fig = plt.figure(figsize=(10, 10), dpi=500)
        fig.patch.set_visible(False)
        gs = gridspec.GridSpec(nrows=math.ceil(datasets / ncols), ncols=ncols, figure=fig, hspace=0.15)

        models = self.db.list_models()
        models.remove('ESN')
        plot_x = ncols
        for grid, (ds_name, classes) in zip(list(gs), self.db.head.items()):
            ax = plt.Subplot(fig, grid)
            ax.set_title(ds_name, fontsize=12, y=1.075, )
            ax.axis('off')
            ax.axis('tight')
            inner = gridspec.GridSpecFromSubplotSpec(nrows=len(classes) + 1, ncols=1, subplot_spec=grid, hspace=0.15)
            # print(ds_name)
            ref_ax = plt.Subplot(fig, inner[0])
            ref_ax.set_title("Riferimento", fontsize=10, color='grey', loc='left', style='italic')
            ref_ax.axis('off')
            ref_ax.axis('tight')

            reference = [
                # classes['Multiple S.R.']['Units 50']['ESN'],
                classes['Multiple S.R.']['Units 100']['ESN'],
                # classes['Multiple S.R.']['Units 250']['ESN'],
            ]

            i = np.argmax([r.score for r in reference])

            body = [[reference[i].get_accuracy_str()]]
            the_table = ref_ax.table(cellText=body, rowLabels=["ESN"], colLabels=x_labels,
                                     loc='center', cellLoc='center', edges='horizontal', fontsize=7.)
            the_table.auto_set_font_size(False)
            the_table.auto_set_column_width(col=list(range(len(x_labels))))
            cells = the_table.get_celld()
            for cell in cells.values():
                cell.set_height(0.2)
            for c in range(len(x_labels)):
                cells[0, c].visible_edges = 'B'
            fig.add_subplot(ref_ax)
            for sub_grid, (class_name, experiments) in zip([inner[1], inner[2]], classes.items()):
                # print("\t", class_name, ":")
                sub_ax = plt.Subplot(fig, sub_grid)
                sub_ax.set_title(class_name, fontsize=10, color='grey', loc='left', style='italic')
                sub_ax.axis('off')
                sub_ax.axis('tight')

                best = get_best_models(experiments, models)

                body = [[best[m][1].get_accuracy_str()] for m in models]

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
                    best_row = np.argmax([best[m][1].get_accuracy_mean() for m in models], axis=0)
                    cells[best_row + 1, 0].get_text().set_color('#008000')
                except IndexError or KeyError:
                    pass

                fig.add_subplot(sub_ax, sharex=ref_ax)

            fig.add_subplot(ax)

        self._save_and_show(fig, path, None, None, "summary table", show)
        return True

    

        # X is self Y is model_name

    def plot_hp_table_by_experiment(self, dataset_name: str, class_name: str, experiment: str, keys: [],
                                    path: str = None, show: bool = True):
        try:
            models_tree = self.db.head[dataset_name][class_name][experiment]
        except KeyError as _:
            # print("Error: ", class_name, experiment_name, "Not found")
            return

        models = get_sorted_keys(models_tree, sort=models_sort)

        self = set()
        for model in models:
            self.update(models_tree[model].keras_hps.values.keys())
        hp_keys = get_matching_keys(keys, self)
        try:
            hp_keys.remove("use G.S.R.")
        except IndexError or KeyError:
            pass
        finally:
            hp_keys = get_sorted_keys(hp_keys)

        body = []
        for model in models:
            row = []
            for key in hp_keys:
                value = models_tree[model].keras_hps.values.get(key)
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

    def plot_dataset_histogram(self, dataset_name: str, path: str = None, show: bool = True):
        try:
            classes_tree = self.db.head[dataset_name]
        except KeyError:
            print("Benchmarks not found. %s" % (dataset_name))
            return False

        models = self.db.list_models()

        def add_stat(values, name, stat):
            values[name][0].append(stat.get_accuracy_mean())
            values[name][1].append(stat.get_accuracy_error_min())
            values[name][2].append(stat.get_accuracy_error_max())
            values[name][3].append(stat)

        values = {name: ([], [], [], []) for name in models}
        add_stat(values, 'ESN', classes_tree['Multiple S.R.']['Units 100']['ESN'])
        add_stat(values, 'IRESN', classes_tree['Multiple S.R.']['Units 100']['IRESN'])
        add_stat(values, 'IRESN', classes_tree['Single S.R.']['Units 100']['IRESN'])
        add_stat(values, 'IIRESN', classes_tree['Multiple S.R.']['Units 100']['IIRESN'])
        add_stat(values, 'IIRESN', classes_tree['Single S.R.']['Units 100']['IIRESN'])
        add_stat(values, 'IIRESNvsr', classes_tree['Multiple S.R.']['Units 100']['IIRESNvsr'])
        add_stat(values, 'IIRESNvsr', classes_tree['Single S.R.']['Units 100']['IIRESNvsr'])

        x_labels = ["Reference", "Multiple S.R.", "Single S.R."]
        fig, ax = plt.subplots()
        # fig.set_size_inches(6, 4)
        width = 0.15
        margin = 0.02
        total = width + margin
        x = np.arange(len(x_labels))
        ax.set_xticks(ticks=x, labels=x_labels)

        ax.bar(x[0], values['ESN'][0], yerr=[values['ESN'][1], values['ESN'][2]],
               label='ESN', width=width, align='center', capsize=3)
        ax.bar(x[1:] - total, values['IRESN'][0], yerr=[values['IRESN'][1], values['IRESN'][2]],
               label='IRESN', width=width, align='center', capsize=3)
        ax.bar(x[1:], values['IIRESN'][0], yerr=[values['IIRESN'][1], values['IIRESN'][2]],
               label='IIRESN', width=width, align='center', capsize=3)
        ax.bar(x[1:] + total, values['IIRESNvsr'][0], yerr=[values['IIRESNvsr'][1], values['IIRESNvsr'][2]],
               label='IIRESNvsr', width=width, align='center', capsize=3)

        ax.set_ylabel('Accuracy')

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

        min_acc = []
        max_acc = []
        for a, x in values.items():
            for acc in x[3]:
                min_acc.append(acc.get_accuracy_min())
                max_acc.append(acc.get_accuracy_max())

        y_min = round_down(np.min(min_acc))
        y_max = round_up(np.max(max_acc))
        ax.set_ylim([y_min, y_max])

        locator = (y_max - y_min) / 5.
        ax.yaxis.set_major_locator(MultipleLocator(locator))
        ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
        ax.set_title(dataset_name)
        ax.legend(loc='lower right')
        self._save_and_show(fig, path, dataset_name, None, "bars " + dataset_name, show)
        return fig

    def _plot_histograms(self, title, data, x_labels, split=True):
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
        x_new = x_labels
        if split:
            # Spit dataset_name names
            x_new = []
            for label in x_labels:
                new = "" + label[0]
                for c in label[1:]:
                    if c.isupper():
                        new += '\n'
                    new += c
                x_new.append(new)

        # y_min = round_down(self.get_min_acc())  # round_down(np.min([np.subtract(x[0], x[1]) for _, x in data.items()]))
        y_max = 1.00  # round_up(np.max([np.add(x[0], x[2]) for _, x in data.items()]))

        ax.set_xticks(ticks=x, labels=x_new)
        ax.set_ylabel('Accuracy')

        # ax.set_ylim([y_min, y_max])
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        # locator = (y_max - y_min) / 5.
        # ax.yaxis.set_major_locator(MultipleLocator(locator))
        # ax.yaxis.set_minor_locator(MultipleLocator(locator / 2.))
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
        ax.set_ylim([round_down(self.db.get_min_acc()), 1.0])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(title)
        ax.yaxis.grid(True)

        plt.tight_layout()
        return fig
"""


def get_best_models(experiments, list_models):
    best = {name: (0., None, None) for name in list_models}
    for experiment_name, models in experiments.items():
        for model_name in list_models:
            stat = models[model_name]
            if best[model_name][0] < stat.get_score():
                best[model_name] = (stat.get_score(), stat, experiment_name)
    return best


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
        except IndexError or KeyError:
            pass
    return fig
