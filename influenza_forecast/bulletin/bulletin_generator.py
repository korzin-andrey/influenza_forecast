import json
import os
import os.path as osp
import pandas as pd
import io
from io import StringIO
import jsonpickle
import matplotlib.pyplot as plt
import sys
import inspect
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import hex_to_rgb
from pathlib import Path
from sklearn.metrics import r2_score
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from matplotlib import rc

from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS
from typing import List
import base64

from gui import aux_functions
# import aux_functions
from bootstrapping import predict_gates
from models import BR_model_new


# if importing modules doesn't work - apply this command in command line
# export PYTHONPATH="${PYTHONPATH}:/home/andrey/RSCF_Uncertainty/BaroyanAgeMultistrain_v2/"

def compile_tex_file(tex_file_name):
    print("Compiling PDF...")
    os.system("pdflatex " + tex_file_name)
    print("Bulletin successfully generated!")
    return


def make_plot(bulletin_req_json):
    print("Creating beautiful plots...")
    epid_data = pd.read_csv(StringIO(jsonpickle.decode(
        bulletin_req_json["epid_data_pickled"])))
    simul_weekly = pd.read_csv(StringIO(jsonpickle.decode(
        bulletin_req_json["simul_weekly_pickled"])))

    incidence = bulletin_req_json["simulation_parameters"]["incidence"]
    # exposed_values = bulletin_req_json['simulation_parameters']['exposed']
    # lambda_values = bulletin_req_json['simulation_parameters']['lambda']
    # a = bulletin_req_json['simulation_parameters']['a']
    # mu = bulletin_req_json['simulation_parameters']['mu']
    delta = bulletin_req_json['simulation_parameters']['delta']
    city = bulletin_req_json['simulation_parameters']['city']
    year = bulletin_req_json['simulation_parameters']['year']
    bulletin_req_json['simulation_parameters']

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 15
    xticks_vals, xticks_text = aux_functions.generate_xticks(
        epid_data, year, last_simul_ind)

    fig = go.Figure()
    colors = px.colors.qualitative.D3
    groups = bulletin_req_json['groups']
    labels = bulletin_req_json['groups']

    for i, (group, label) in enumerate(zip(simul_weekly.columns, labels)):
        fig.add_trace(go.Scatter(x=epid_data[group].index,
                                 y=epid_data[group],
                                 customdata=epid_data.loc[:, ['Неделя']],
                                 hovertemplate="<br>%{customdata[0]} неделя"
                                 "<br>Количество заболеваний: %{y}"
                                 "<extra></extra>",
                                 mode='markers+lines',
                                 legendgroup='data',
                                 line={'dash': 'dash', 'shape': 'spline',
                                       'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'},
                                 marker={'color': colors[i], 'size': 10, },
                                 name='Данные, ' + label))

        fig.add_trace(go.Scatter(x=simul_weekly[group].index[:last_simul_ind],
                                 y=simul_weekly[group][:last_simul_ind],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                 "<extra></extra>",
                                 mode='lines',
                                 legendgroup='model-fit',
                                 marker={'color': colors[i], 'size': 10},
                                 name='Модель, ' + label))
        fig.update_layout(
            template='plotly_white',
            autosize=False,
            height=800,
            width=1600,
            margin={'l': 60, 'r': 50, 'b': 50, 't': 50, 'pad': 4},
            paper_bgcolor="white",
            title={
                'text': f"{aux_functions.cities[city]}, {year}-{year + 1} гг.",
                'font': {'size': 25},
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font={'size': 25},
            yaxis={'exponentformat': 'power',
                   'showtickprefix': 'all', 'tickfont': {'size': 20}},
            xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
                   'tickfont': {'size': 24}, 'showgrid': False}
        )
        print(xticks_text)
        fig.update_layout(uniformtext_minsize=15, uniformtext_mode='hide')
        fig.update_xaxes(title_text="Недели")
        fig.update_yaxes(
            title_text="Количество случаев заболевания")
    # fig.show()
    if incidence == 'age-group':
        fig.write_image(os.path.join("age-group.png"))
    elif incidence == 'strain':
        fig.write_image(os.path.join("strain.png"))
    elif incidence == 'strain_age-group':
        fig.write_image(os.path.join("strain_age-group.png"))
    elif incidence == 'total':
        fig.write_image(os.path.join("total.png"))


def make_plot_forecast(bulletin_req_json):
    print("Creating beautiful plots...")

    epid_data = pd.read_csv(StringIO(jsonpickle.decode(
        bulletin_req_json["epid_data_pickled"])))
    simul_weekly = pd.read_csv(StringIO(jsonpickle.decode(
        bulletin_req_json["simul_weekly_pickled"])))

    incidence = bulletin_req_json["simulation_parameters"]["incidence"]
    exposed_values = bulletin_req_json['simulation_parameters']['exposed']
    lambda_values = bulletin_req_json['simulation_parameters']['lambda']
    a = bulletin_req_json['simulation_parameters']['a']
    mu = bulletin_req_json['simulation_parameters']['mu']
    delta = bulletin_req_json['simulation_parameters']['delta']
    city = bulletin_req_json['simulation_parameters']['city']
    year = bulletin_req_json['simulation_parameters']['year']
    sample_size = int(
        bulletin_req_json['simulation_parameters']['sample_size'])
    inflation_parameter = bulletin_req_json['simulation_parameters']['inflation_parameter']
    forecast_term = int(
        bulletin_req_json['simulation_parameters']['inflation_parameter'])
    forecast_term = 5

    colors = px.colors.qualitative.D3
    colors_unexistant = ["#383838", "#585858", "#696969", "#909090"]
    error_structures = ["#D3D3D3", "#E5E4E2", "#C0C0C0", "#A9A9A9"]
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = aux_functions.prepare_exposed_list(incidence, exposed_list)

    fig = go.Figure()

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 15

    ds_amount = int(100 / len(simul_weekly.columns))
    groups = bulletin_req_json['groups']
    labels = bulletin_req_json['groups']

    predict_gates_generator = predict_gates.PredictGatesGenerator(epid_data.loc[:, simul_weekly.columns],
                                                                  simul_weekly.dropna(
        axis=1),
        ds_amount, sample_size, inflation_parameter, end=last_simul_ind)
    percentiles = [(5, 95)]
    gates = [
        predict_gates_generator.generate_predict_gate(
            p[0], p[1], length=forecast_term)
        for p in percentiles
    ]

    xticks_vals, xticks_text = aux_functions.generate_xticks(
        epid_data, year, last_simul_ind)
    pos_x = xticks_vals[xticks_vals['year'] == year].index[-1]

    r_squared = r2_score(epid_data[groups], simul_weekly.iloc[delta:epid_data.index[-1] + 1, :],
                         multioutput='raw_values')

    labels = [group if '15 и ст.' not in group
              else group.replace('15 и ст.', '15+') for group in groups]

    for i, (group, label) in enumerate(zip(simul_weekly.columns, labels)):
        print(group)
        fig.add_trace(go.Scatter(x=epid_data[group][sample_size - 1:sample_size + 1].index,
                                 y=epid_data[group][sample_size -
                                                    1:sample_size + 1],
                                 customdata=epid_data.loc[sample_size -
                                                          1:sample_size + 1, ['Неделя']],
                                 mode='lines',
                                 line={'dash': 'dash', 'shape': 'spline',
                                       'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'}, showlegend=False))

        # points of data
        fig.add_trace(go.Scatter(x=epid_data[group][:sample_size].index,
                                 y=epid_data[group][:sample_size],
                                 customdata=epid_data.loc[:, ['Неделя']],
                                 hovertemplate="<br>%{customdata[0]} неделя"
                                               "<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='markers+lines',
                                 legendgroup='data',
                                 line={'dash': 'dash', 'shape': 'spline',
                                       'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'},
                                 marker={'color': colors[i], 'size': 10, },
                                 name='Данные, ' + label))

        # points of prediction
        fig.add_trace(go.Scatter(x=epid_data[group][sample_size:].index,
                                 y=epid_data[group][sample_size:],
                                 customdata=epid_data.loc[sample_size:, [
                                     'Неделя']],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='markers+lines',
                                 line={'dash': 'dash', 'shape': 'spline',
                                       'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'},
                                 legendgroup='data',
                                 marker={
                                     'color': colors[i], 'size': 10, 'opacity': 0.5},
                                 showlegend=False))

        # lines of the model
        fig.add_trace(go.Scatter(x=simul_weekly[group].index[:last_simul_ind],
                                 y=simul_weekly[group][:last_simul_ind],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='lines',
                                 legendgroup='model-fit',
                                 marker={'color': colors[i], 'size': 10},
                                 name='Модель, ' + label))

        for gate_i, gate_list in enumerate(gates):
            predict_gate = next(
                filter(lambda gt: gt.column == group, gate_list))
            x_ = predict_gate.x[sample_size -
                                1:sample_size + predict_gate.length]
            y1_ = predict_gate.y_min[sample_size -
                                     1:sample_size + predict_gate.length]
            y2_ = predict_gate.y_max[sample_size -
                                     1:sample_size + predict_gate.length]

            y1_[0] = y2_[
                0] = simul_weekly[predict_gate.column][predict_gate.week_begin + sample_size - 1]

            # borders of prediction
            fig.add_trace(go.Scatter(x=x_, y=y1_, fill=None, fillcolor=f'rgba{(*hex_to_rgb(colors[i]), 0.3)}',
                                     mode='lines',
                                     marker={'color': "rgba(255,0,0,0.5)", 'size': 10}, showlegend=False))

            fig.add_trace(go.Scatter(x=x_, y=y2_, fill='tonexty', fillcolor=f'rgba{(*hex_to_rgb(colors[i]), 0.3)}',
                                     mode='lines',
                                     name=f"Границы прогноза (PR: {percentiles[gate_i][0]}-{percentiles[gate_i][1]}), {label}",
                                     marker={'color': "rgba(255,0,0,0.5)", 'size': 10}, showlegend=False))

    # for i, r2 in enumerate(r_squared):
    #     fig.add_annotation(text=f'<b>$R^2={str(round(r2, 2))}$</b>',
    #                        showarrow=False,
    #                        xanchor='left',
    #                        xref='paper',
    #                        x=0.03,
    #                        yshift=i * (-25) + 300,
    #                        font={'color': colors[i], 'size': 32})
    fig.update_layout(
        template='plotly_white',
        autosize=False,
        height=1000,
        width=2400,
        margin={'l': 60, 'r': 50, 'b': 50, 't': 50, 'pad': 4},
        paper_bgcolor="white",
        title={
            'text': f"{aux_functions.cities[city]}, {year}-{year + 1} гг.",
            'font': {'size': 40},
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font={'size': 32},
        yaxis={'exponentformat': 'power', 'showtickprefix': 'all'},
        xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
               'tickfont': {'size': 32}, 'showgrid': False},
        legend={'traceorder': 'normal'}
    )
    border_of_data = m + sample_size - 1
    fig.add_vline(x=border_of_data,
                  line_width=3, line_dash="dash", line_color="red",
                  name=f'Граница доступности <br> данных',
                  showlegend=True)
    fig.add_vline(x=pos_x, line_width=2, line_dash="dash", line_color="green",
                  name=f'Граница  <br>{year}-{year + 1} годов',
                  showlegend=True)
    fig.update_xaxes(title_text="Недели")
    fig.update_yaxes(title_text="Количество случаев заболевания")
    if incidence == 'age-group':
        fig.write_image(os.path.join("age-group_forecast.png"))
        # filling up description arguments
        description_args = {'number_of_week': None, 'year': None, 'incidence_0-14': None, 'incidence_15+': None,
                            'prev_incidence_0-14': None, 'prev_incidence_15+': None, 'increase/decrease_0-14': None,
                            'delta_0-14': None, 'delta_0-14_percent': None,
                            'increase/decrease_15+': None, 'delta_15+': None, 'delta_15+_percent': None}
        weeks = epid_data['15 и ст.'][:sample_size].index
        epid_data_0_14 = epid_data['0-14'][:sample_size].to_list()
        epid_data_15 = epid_data['15 и ст.'][:sample_size].to_list()
        print(epid_data_15[-1])

        description_args['number_of_week'] = 1
        description_args['year'] = 2020
        description_args['incidence_0-14'] = int(epid_data_0_14[-1])
        description_args['incidence_15+'] = int(epid_data_15[-1])
        description_args['prev_incidence_0-14'] = int(epid_data_0_14[-2])
        description_args['prev_incidence_15+'] = int(epid_data_15[-2])
        description_args['delta_0-14'] = int(abs(
            epid_data_0_14[-1] - epid_data_0_14[-2]))
        description_args['delta_15+'] = int(abs(
            epid_data_15[-1] - epid_data_15[-2]))
        description_args['delta_0-14_percent'] = round(
            100 * abs(epid_data_0_14[-1] - epid_data_0_14[-2]) / epid_data_0_14[-2], 1)
        print(description_args['delta_0-14_percent'])
        description_args['delta_15+_percent'] = round(
            100 * abs(epid_data_15[-1] - epid_data_15[-2]) / epid_data_15[-2], 1)
        if (epid_data_0_14[-1] - epid_data_0_14[-2] > 0):
            description_args['increase/decrease_0-14'] = 'увеличилось'
        else:
            description_args['increase/decrease_0-14'] = 'уменьшилось'
        if (epid_data_15[-1] - epid_data_15[-2] > 0):
            description_args['increase/decrease_15+'] = 'увеличилось'
        else:
            description_args['increase/decrease_15+'] = 'уменьшилось'

        generate_description_age_group(description_args)
    elif incidence == 'strain':
        fig.write_image(os.path.join("strain_forecast.png"))
    elif incidence == 'strain_age-group':
        fig.write_image(os.path.join("strain_age-group_forecast.png"))
    elif incidence == 'total':
        fig.write_image(os.path.join("total_forecast.png"))

    # generating description
    return fig


def plot_rt(rt: ndarray, labels: List, city: str, year: int, output_file: str):
    colors = list(TABLEAU_COLORS.keys()) + list(BASE_COLORS.keys())
    labels = [item.replace('15 и ст.', '15+').replace('_', " ")
              for item in labels]
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['text.usetex'] = True
    max_rt = 0
    for i, partial_rt in enumerate(rt):
        for j in range(0, len(partial_rt)):
            max_rt = max(max(partial_rt[0]), max_rt)
            idx = i if len(rt) > 1 else j
            plt.plot(partial_rt[j][11:500], color=colors[i *
                     len(partial_rt) + j], label=labels[idx], linewidth=3)

    plt.title(f"{city}, {year}$-${year + 1}", fontsize=20)
    plt.xlabel('Дни', fontsize=18)
    plt.ylabel(r'$R_t$', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, color='b', alpha=0.2, linestyle='--')
    plt.legend(fontsize=14)
    plt.xlim([-5, 500])
    # plt.ylim([0, 1.1*max_rt])

    plt.axhline(y=1.0, color='black', linestyle='--')
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')


def plot_immune_population(population_immunity: ndarray, labels: List,
                           city: str, year: int, output_file: str):
    """Plots population immunity"""
    colors = list(TABLEAU_COLORS.keys()) + list(BASE_COLORS.keys())
    labels = [item.replace('15 и ст.', '15+').replace('_', " ")
              for item in labels]
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    max_immun = 0
    for i, immun in enumerate(population_immunity):
        for j in range(0, len(immun)):
            idx = i if len(population_immunity) > 1 else j
            max_immun = max(max(immun[0]), max_immun)
            ax.plot(immun[j][:500], color=colors[i *
                                                 len(immun) + j], label=labels[idx], linewidth=3)

    plt.title(f"{city}, {year}$-${year + 1}", fontsize=20)
    plt.xlabel('Дни', fontsize=18)
    plt.ylabel('Количество людей с иммунитетом, \n тыс. чел.', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, color='b', alpha=0.2, linestyle='--')
    plt.legend(fontsize=14)
    plt.xlim([0, 500])
    # plt.ylim([0, 1.1*max_immun/1000])
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')


def make_plot_indicators(bulletin_req_json):
    simul_weekly = pd.read_csv(StringIO(jsonpickle.decode(
        bulletin_req_json["simul_weekly_pickled"])))
    incidence = bulletin_req_json["simulation_parameters"]["incidence"]
    year = bulletin_req_json['simulation_parameters']['year']

    model_obj = jsonpickle.decode(bulletin_req_json["model_obj_pickled"])
    y, population_immunity, rho, r0, rt = model_obj.make_simulation()

    plot_rt(rt, list(simul_weekly.columns), 'Санкт-Петербург',
            year, r'{}_rt.png'.format(incidence))
    plot_immune_population(population_immunity, list(
        simul_weekly.columns), 'Санкт-Петербург', year, r'{}_immunne_population.png'.format(incidence))
    return


def generate_description_age_group(description_args):
    # description_args = {'number_of_week': 10000, 'year': 2019, 'incidence_0-14': 14000, 'incidence_15+': 13000,
    #                     'prev_incidence_0-14': 10000, 'prev_incidence_15+': 10000, 'increase/decrease_0-14': "возросло",
    #                     'delta_0-14': 10000, 'increase/decrease_15+': "уменьшилось", 'delta_15+': 10000}
    desctription_file = io.open(
        "description_age_group.txt", mode='r', encoding='utf-8')
    description_string = desctription_file.read()
    # number of week, year, incidence cases 0-14, incidence cases 15+, prev incidence cases 0-14, prev incidence cases 15+,
    # increase/decrease word, delta 0-14, increase/decrease, delta 15+
    description_string = description_string.format(
        description_args['number_of_week'], description_args['year'], description_args['incidence_0-14'],
        description_args['incidence_15+'], description_args['prev_incidence_0-14'], description_args['prev_incidence_15+'],
        description_args['increase/decrease_0-14'], description_args['delta_0-14'], description_args['delta_0-14_percent'],
        description_args['increase/decrease_15+'], description_args['delta_15+'], description_args['delta_15+_percent'])
    print(description_string)
    tex_file = io.open(
        "description_age_group.tex", mode='w', encoding='utf-8')
    tex_file.write(description_string)


def generate_bulletin():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    for file in os.listdir(os.getcwd()):
        # Check whether file is in text format or not
        # if file.endswith("retrospective.json"):
        #     # file_path = f"{os.path}\{file}"
        #     # filename = "total_request.json"
        #     bulletin_req_path = osp.join(file)
        #     with open(bulletin_req_path, 'rb') as f:
        #         bulletin_req_json = json.load(f)
        #     incidence = bulletin_req_json["simulation_parameters"]["incidence"]
        #     # forecasting = bulletin_req_json["simulation_parameters"]["forecasting"]
        #     # print(forecasting)
        #     print(incidence)
        #     make_plot(bulletin_req_json)
            # make_plot_indicators(bulletin_req_json)
        if file.endswith("forecast.json"):
            bulletin_req_path = osp.join(file)
            with open(bulletin_req_path, 'rb') as f:
                bulletin_req_json = json.load(f)
            incidence = bulletin_req_json["simulation_parameters"]["incidence"]
            # forecasting = bulletin_req_json["simulation_parameters"]["forecasting"]
            # print(forecasting)
            print(incidence)
            make_plot_forecast(bulletin_req_json)

    compile_tex_file(tex_file_name="bulletin_template.tex")


if __name__ == "__main__":
    print("Generating bulletin ...")
    generate_bulletin()
