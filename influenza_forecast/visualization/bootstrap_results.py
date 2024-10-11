import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import r2_score
import os.path as osp
from typing import List
from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS

from bootstrapping.predict_gates import PredictGatesGenerator
from visualization.colors_config import *


def get_colormap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    References https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib"""
    return plt.cm.get_cmap(name, n)


def get_rand_colors(amount):
    colors_ = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    return colors_(amount)


def plot_95CI_bootstrapped_curves(df_weekly, bootstrapped_curves, city, year, output_dir="./"):
    """Plots 95% bootstrapped curves based on the value of determination coefficient"""
    column_titles = bootstrapped_curves[0].columns
    print(column_titles)
    colormap = get_rand_colors(len(column_titles))
    for i_color, title in enumerate(column_titles):
        m, n = df_weekly[title].index.min(), df_weekly[title].index.max()

        r2_values = [r2_score(df_weekly[title], curve[title][m:n + 1]) for curve in bootstrapped_curves]

        lhs, rhs = np.percentile(r2_values, [5, 95])

        for i, curve in enumerate(bootstrapped_curves):
            if lhs <= r2_values[i] <= rhs:
                plt.plot(curve[title][:n + 15].index, curve[title][:n + 15], color=colormap[i_color], alpha=0.5)

        plt.plot(df_weekly[title].index, df_weekly[title], 'o', color=colormap[i_color], label=f"{title} data")

    plt.xlabel('Time, weeks', fontsize=13)
    plt.ylabel('Incidence, cases', fontsize=13)
    plt.title(f'{city}, {year}$-${int(year) + 1}\n95% CI bootstrapped curves', fontsize=13)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend()

    if output_dir:
        plt.savefig(osp.join(output_dir, "plot_95CI_bootstrapped_curves.png"), dpi=300)

    plt.show()


class EmpiricalDistributionPlot:
    """Class for plotting empirical distributions of the epidemic parameters"""

    def __init__(self, plt, name, values=None):
        self.name = name
        self.plt = plt
        self.values = [] if values is None else values

    def plot_distribution(self, city, year):
        self.plt.hist(self.values)

        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places

        locs, _ = plt.yticks()

        plt.yticks(locs, np.round(locs / len(self.values), 3))

        self.plt.title(f'\n{city}, {year}$-${year + 1}' +
                       '\n95% CI: {}\n'.format(np.round(np.percentile(self.values, [5, 95]), 4)) +
                       f'CV={np.round(np.std(self.values), 4)}, ' +
                       f'mean={np.round(np.mean(self.values), 4)}, ' +
                       f'median={np.round(np.median(self.values), 4)}',
                       fontsize=12)

        self.plt.xlabel(self.name)
        self.plt.ylabel("Density")

    def draw_mean(self, **kwargs):
        self.plt.axvline(np.mean(self.values), **kwargs)

    def draw_median(self, **kwargs):
        self.plt.axvline(np.median(self.values), **kwargs)

    def save_to(self, path, filetype, **kwargs):
        self.plt.savefig(osp.join(path, f"{self.name}.{filetype}"), **kwargs)


def plot_empirical_distributions(df_samples_mapped, city, year, output_dir="./", **keys_titles):
    for name, dataframe in df_samples_mapped.items():
        plot_empirical_distribution(dataframe, name, city, year, output_dir, **keys_titles)


def plot_empirical_distribution(df_sample, name, city, year, output_dir="./", **keys_titles):
    for key in keys_titles.keys():
        dist_plot = EmpiricalDistributionPlot(plt, f"{name}; parameter: {keys_titles[key]}",
                                              np.array([val for val in df_sample[key]]))
        dist_plot.plot_distribution(city, year)
        dist_plot.draw_mean(color="black", linestyle="--")
        dist_plot.draw_median(color="grey", linestyle="--")
        dist_plot.plt.xticks(fontsize=12)
        dist_plot.plt.yticks(fontsize=12)

        if output_dir:
            dist_plot.save_to(output_dir, "png", dpi=300)

        dist_plot.plt.show()


def restructurate_parameters_df(df_samples_raw, columns=None):
    if columns is None:
        columns = ["total"]

    df_mapping = dict()

    def transform_value(obj, index):
        if isinstance(obj, list):
            if len(obj) > index:
                return obj[i]
            return obj[0]
        return obj

    for i, key_column in enumerate(columns):
        df_sample_i = df_samples_raw.copy()
        for column in df_sample_i.columns:
            df_sample_i[column] = df_sample_i[column].map(lambda v: transform_value(v, i))
        df_mapping[key_column] = df_sample_i
        print(df_sample_i)

    return df_mapping


def plot_synth_data(city, year, datasets_list, synth_data_params, incidence_data, model_fit=None, output_path='./'):
    if len(synth_data_params) != 0:
        title = f'{city}, {year}$-${year + 1}\n' + 'Predictive analysis simulated datasets' + '\n' + \
                f'inflation parameter = {round(synth_data_params["inflation_parameter"], 2)}'
        sample_size = synth_data_params['sample_size']
    else:
        title = f'{city}, {year}$-${year + 1}\n' + 'Retrospective analysis simulated datasets'
        sample_size = 0

    columns = datasets_list[0].columns

    for column_i, column in enumerate(columns):
        color_setup = setup_colors[column_i]

        outbreak_begin, outbreak_end = incidence_data[column].index.min(), incidence_data[column].index.max() + 1

        for i, sm in enumerate(datasets_list):
            plt.plot(sm[column][outbreak_begin:outbreak_end], color=color_setup.errorStructureColor)

        if model_fit is not None:
            plt.plot(model_fit[column][outbreak_begin:outbreak_end], color=color_setup.incidenceCurveColor,
                     label='Model Best-Fit')

        if sample_size == 0:
            plt.plot(incidence_data[column], 'o', color=color_setup.dataColor.full,
                     label='Original data')
        else:
            plt.plot(incidence_data[column][:sample_size], 'o', color=color_setup.dataColor.sample,
                     label='Predict sample')
            plt.plot(incidence_data[column][sample_size:], 'o', color=color_setup.dataColor.full,
                     label='Original data')

        plt.xlabel('Time, weeks', fontsize=13)
        plt.ylabel('Incidence, cases', fontsize=13)
        plt.title(title, fontsize=13)

        plt.legend()
        if output_path:
            plt.savefig(osp.join(output_path, f'{column}_{len(datasets_list)}_synth_datasets.png'), dpi=1200)
        plt.show()
        plt.close()


def plot_gates(city, year, predict_gates_generator: PredictGatesGenerator, file_path_gates, *gates,
               simulated_datasets_max_week=None):
    columns = predict_gates_generator.simulated_datasets[0].columns

    for column_i, column in enumerate(columns):
        color_setup = setup_colors[column_i]

        # plotting error structures
        for sm in predict_gates_generator.simulated_datasets:
            plt.plot(sm[column][predict_gates_generator.outbreak_begin + predict_gates_generator.sample_size - 1:
                                (
                                        predict_gates_generator.outbreak_end + 1) if simulated_datasets_max_week is None else simulated_datasets_max_week + 1],
                     color=color_setup.errorStructureColor)

        # plotting predict sample and original data
        plt.plot(predict_gates_generator.original_data[column][:predict_gates_generator.sample_size], 'o',
                 color=color_setup.dataColor.sample,
                 # label=f'"{column}" Predict sample',
                 zorder=5000 + column_i)
        plt.scatter(predict_gates_generator.original_data[column][predict_gates_generator.sample_size:].index,
                    predict_gates_generator.original_data[column][predict_gates_generator.sample_size:],
                    color='white', edgecolor=color_setup.dataColor.full,
                    # label=f'"{column}" Original data',
                    zorder=5000 + column_i)

        # plotting model fit
        plt.plot(predict_gates_generator.model_fit[column][
                 predict_gates_generator.outbreak_begin:predict_gates_generator.outbreak_end],
                 color=color_setup.incidenceCurveColor,
                 label=f'"{column if column != "Все" else "total"}" Calibration',
                 zorder=210 + column_i)

        # plotting gates
        for gate_i, gate in enumerate(gates):
            predict_gate = next(filter(lambda gt: gt.column == column, gate))
            print(predict_gate)
            predict_gates_generator.draw_gate(plt, predict_gate,
                                              show_title=False,
                                              color=color_setup.gatesColors[gate_i], alpha=0.3,
                                              zorder=1000 * (gate_i + 1) + column_i)

    plt.axvline(predict_gates_generator.outbreak_begin + predict_gates_generator.sample_size - 1, color='black',
                linestyle='dashed')

    plt.xlabel('Time, weeks', fontsize=13)
    plt.ylabel('Incidence, cases', fontsize=13)
    plt.title(f'{city}, {year}$-${year + 1}\n' +
              '$n_{sample}=$' + f'{predict_gates_generator.sample_size}', fontsize=13)

    plt.legend()
    if file_path_gates:
        plt.savefig(file_path_gates, dpi=1200)
    plt.show()
    plt.close()


def export_epidemic_indicators(bs_optimizers, full_bs_path):
    with open(osp.join(full_bs_path, f'active_population_list.pickle'), 'wb') as f:
        active_population_list = [optimizer.active_population for optimizer in bs_optimizers]
        pickle.dump(active_population_list, f)

    with open(osp.join(full_bs_path, f'population_immunity_list.pickle'), 'wb') as f:
        population_immunity_list = [optimizer.population_immunity for optimizer in bs_optimizers]
        pickle.dump(population_immunity_list, f)

    with open(osp.join(full_bs_path, f'Rt_list.pickle'), 'wb') as f:
        Rt_list = [optimizer.rt for optimizer in bs_optimizers]
        pickle.dump(Rt_list, f)

    with open(osp.join(full_bs_path, f'R0_list.pickle'), 'wb') as f:
        R0_list = [optimizer.r0 for optimizer in bs_optimizers]
        pickle.dump(R0_list, f)


def plot_rt_95CI(rt_list: List[np.ndarray], r2_list, labels: List, city: str, year: int, output_file: str):
    """Plots Rt 95CI"""
    colors = list(TABLEAU_COLORS.keys()) + list(BASE_COLORS.keys())
    labels = [item.replace('15 и ст.', '15+').replace('_', " ")
              for item in labels]
    labels_plot = [True] * len(labels)

    percentile_5  = [np.percentile(r2_list[i], 5) for i in range(len(labels))]
    percentile_95 = [np.percentile(r2_list[i], 95) for i in range(len(labels))]

    for rt_i, rt in enumerate(rt_list):
        for i, partial_rt in enumerate(rt):
            if not (percentile_5[i] <= r2_list[i][rt_i] <= percentile_95[i]): continue
            for j in range(0, len(partial_rt)):
                idx = i if len(rt) > 1 else j

                if labels_plot[idx]:
                    plt.plot(partial_rt[j][11:500], color=colors[i * len(partial_rt) + j], label=labels[idx], alpha=0.5)
                    labels_plot[idx] = False
                else:
                    plt.plot(partial_rt[j][11:500], color=colors[i * len(partial_rt) + j], alpha=0.5)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.axhline(y=1.0, color='black', linestyle='--')
    plt.title(f"{city}, {year}$-${year + 1}\n95% CI Rt bootstrapped curves", fontsize=13)
    plt.xlabel('Days', fontsize=13)
    plt.ylabel('Rt', fontsize=13)
    plt.legend()
    plt.savefig(output_file, dpi=450, bbox_inches='tight')
    plt.show()


def plot_immune_population_95CI(population_immunity_list: List[np.ndarray], r2_list, labels: List,
                                city: str, year: int, output_file: str):
    """Plots population immunity 95CI"""
    colors = list(TABLEAU_COLORS.keys()) + list(BASE_COLORS.keys())
    labels = [item.replace('15 и ст.', '15+').replace('_', " ")
              for item in labels]
    labels_plot = [True] * len(labels)

    percentile_5  = [np.percentile(r2_list[i], 5) for i in range(len(labels))]
    percentile_95 = [np.percentile(r2_list[i], 95) for i in range(len(labels))]

    for pi_i, pop_imm in enumerate(population_immunity_list):
        for i, immun in enumerate(pop_imm):
            if not (percentile_5[i] <= r2_list[i][pi_i] <= percentile_95[i]): continue
            for j in range(0, len(immun)):
                idx = i if len(pop_imm) > 1 else j

                if labels_plot[idx]:
                    plt.plot(immun[j][:500], color=colors[i * len(immun) + j], label=labels[idx], alpha=0.5)
                    labels_plot[idx] = False
                else:
                    plt.plot(immun[j][:500], color=colors[i * len(immun) + j], alpha=0.5)

    plt.title(f"{city}, {year}$-${year + 1}\n95% CI population immunity bootstrapped curves", fontsize=13)
    plt.xlabel('Year', fontsize=13)
    plt.ylabel('Immune population', fontsize=13)
    plt.legend()
    plt.savefig(output_file, dpi=450, bbox_inches='tight')
    plt.show()
