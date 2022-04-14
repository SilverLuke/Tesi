import math
import re
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

from benchmarks import Statistic


def get_path(op_path, obj_path):
    path = op_path
    if path is None:
        if obj_path is None:
            raise ValueError("No path is given")
        else:
            path = obj_path
    return path


class Plotter:
    def __init__(self, db, plot_path):
        self.db = db
        self.path = plot_path

    def histograms_summary(self, path=None, show=False):
        def plot_datasets_summary(ax, dataset, classes, models, plot_x=True, plot_y=True, legend_pos=None):
            width = 0.30
            margin = 0.02
            zoom = 3
            total = width + margin
            x_labels = ["Reference", "Multiple S.R.", "Single S.R."]
            x = np.arange(len(x_labels), dtype=float)
            x[1] = 0.84
            def add_stat(values, name, stat):
                values[name][0].append(stat.get_accuracy_mean() * 100)
                values[name][1].append(stat.get_accuracy_error_min() * 100)
                values[name][2].append(stat.get_accuracy_error_max() * 100)
                values[name][3].append(stat)

            values = {name: ([], [], [], []) for name in models}
            add_stat(values, 'ESN', classes['Reference']['Units 100']['ESN'])
            add_stat(values, 'IRESN', classes['Single S.R.']['Units 100']['IRESN'])
            add_stat(values, 'IIRESN', classes['Single S.R.']['Units 100']['IIRESN'])
            add_stat(values, 'IIRESNvsr', classes['Single S.R.']['Units 100']['IIRESNvsr'])
            add_stat(values, 'IRESN', classes['Multiple S.R.']['Units 100']['IRESN'])
            add_stat(values, 'IIRESN', classes['Multiple S.R.']['Units 100']['IIRESN'])
            add_stat(values, 'IIRESNvsr', classes['Multiple S.R.']['Units 100']['IIRESNvsr'])

            ax.bar(x[0], values['ESN'][0], #yerr=[values['ESN'][1], values['ESN'][2]],
                   label='ESN', width=width, align='center', capsize=3)
            ax.bar(x[1:] - total, values['IRESN'][0], #yerr=[values['IRESN'][1], values['IRESN'][2]],
                   label='IRESN', width=width, align='center', capsize=3)
            ax.bar(x[1:], values['IIRESN'][0], #yerr=[values['IIRESN'][1], values['IIRESN'][2]],
                   label='IIRESN', width=width, align='center', capsize=3)
            ax.bar(x[1:] + total, values['IIRESNvsr'][0], #yerr=[values['IIRESNvsr'][1], values['IIRESNvsr'][2]],
                   label='IIRESNvsr', width=width, align='center', capsize=3)

            if plot_x:
                ax.set_xticks(ticks=x, labels=x_labels,  rotation=15)
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
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5,-0.05))
        self._save_and_show(fig, path, "summary_plot")

    def print_latex_table(self, path=None):
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
            #table += "\\hline\n"
            #table += "\\rowcolor{lightgray}\n"
            #table += "\\multicolumn{%i}{c}{%s}\\tabularnewline\n" % (columns, "Riferimento")
            #table += "\\hline\n"
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
        legend = ["Modello", "Train set $\pm \sigma$", "Validation score", "Test set $\pm \sigma$", "Test set min", "Test set max"]
        columns = len(legend)
        table = "\\begin{table}[ht]\n \\centering\n" \
                "\\resizebox{\\textwidth}{!}{%%\n" \
                "\\begin{tabular}{%s}\n" % ("c" * columns)

        for i in legend[:-1]:
            table += i +" & "
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
    def plot_datasets_summary(self, class_name: str = 'Best Models', experiment: str = 'Best',
                              path: str = None, show: bool = True):
        models = self.db.list_models()
        datasets = get_sorted_keys(self.db.head)

        values = {name: ([], [], []) for name in models}
        for i, dataset in enumerate(datasets):
            try:
                tested_models = self.db.head[dataset][class_name][experiment].keys()
            except KeyError as _:
                return False
            for j, model in enumerate(models):
                if model in tested_models:
                    stat = self.db.head[dataset][class_name][experiment][model]
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

    def plot_hp_table_by_model(self, dataset_name: str, class_name: str, model: str, keys: [],
                               path: str = None, show: bool = True):
        exp_tree = self.db.head[dataset_name][class_name]
        experiments = get_sorted_keys(exp_tree)

        self = set()
        for experiment in experiments:
            try:
                stat = exp_tree[experiment][model]
                self.update(stat.hyperparameters.values.keys())
            except IndexError or KeyError:
                pass
        hp_keys = get_matching_keys(keys, self)
        try:
            hp_keys.remove("use G.S.R.")
        except IndexError or KeyError:
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
                except IndexError or KeyError:
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
            self.update(models_tree[model].hyperparameters.values.keys())
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
            if best[model_name][0] < stat.score:
                best[model_name] = (stat.score, stat, experiment_name)
    return best


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
        except IndexError or KeyError:
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


def natural_sort(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


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
