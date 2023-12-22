import base64
import json
import datetime
import pickle
import jsonpickle
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import State, ALL
from dash import Dash, Input, Output, ctx, html, dcc, callback
from dash_extensions.enrich import DashProxy, Output, Input, CycleBreakerTransform, CycleBreakerInput
from plotly.colors import hex_to_rgb
from sklearn.metrics import r2_score
from dash.exceptions import PreventUpdate
import components.age_groups_components as age_groups_comps
import components.strains_age_components as strains_age_comps
import components.strains_components as strains_comps
import components.total_components as total_comps
from aux_functions import prepare_exposed_list, get_data_and_model, transform_days_to_weeks, cities, generate_xticks, \
    exposed_dict_to_inputs, lambda_dict_to_inputs
from bootstrapping.predict_gates import PredictGatesGenerator
from components import multi_strain, multi_age, multi_strain_age, total_c
from layout import layout, get_model_params_components, get_data_components
import bulletin.bulletin_generator as bulletin_generator
import calibration
import time
import os


import diskcache
from utils.experiment_setup import ExperimentalSetup
from aux_functions import get_contact_matrix, prepare_calibration_data, get_config
from optimizers.multiple_model_fit import MultipleModelFit
from dash.long_callback import DiskcacheLongCallbackManager

import dash

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = DashProxy(__name__,
                transforms=[CycleBreakerTransform()],
                external_stylesheets=[dbc.themes.MATERIA],
                prevent_initial_callbacks="initial_duplicate", 
                long_callback_manager=long_callback_manager)

app.layout = layout
PRESET_MODE = False
a_default = None

@app.callback(
    [Output('exposed-accordion-item', 'children', allow_duplicate=True),
     Output('lambda-accordion-item', 'children', allow_duplicate=True)],
    Input('incidence', 'value'),
    prevent_initial_call=True
)
def update_components(incidence):
    """
    Returns respect components according to the provided incidence type
    """
    global PRESET_MODE
    if PRESET_MODE:
        PRESET_MODE = False
        raise PreventUpdate

    if incidence == 'age-group':
        return [multi_age['exposed'], multi_age['lambda']]
    elif incidence == 'strain':
        return [multi_strain['exposed'], multi_strain['lambda']]
    elif incidence == 'strain_age-group':
        return [multi_strain_age['exposed'], multi_strain_age['lambda']]
    elif incidence == 'total':
        return [total_c['exposed'], total_c['lambda']]


@app.callback(
    [Output({'type': 'exposed', 'index': ALL}, 'value'),
     Output({'type': 'lambda', 'index': ALL}, 'value'),
     Output('a', 'value'),
     Output('mu', 'value'),
     Output('delta', 'value'),
     Output('sample', 'value'),
     Output('forecast-term', 'value'),
     Output('inflation-parameter', 'value')],
    [Input({'type': 'exposed_io', 'index': ALL}, 'value'),
     Input({'type': 'lambda_io', 'index': ALL}, 'value'),
     CycleBreakerInput('a_io', 'value'),
     CycleBreakerInput('mu_io', 'value'),
     CycleBreakerInput('delta_io', 'value'),
     CycleBreakerInput('sample_io', 'value'),
     CycleBreakerInput('forecast-term_io', 'value'),
     CycleBreakerInput('inflation-parameter_io', 'value')],
    prevent_initial_call=True
)
def update_sliders(exposed_io, lambda_io, a_io, mu_io, delta_io, sample_io, forecast_term_io, inflation_parameter_io):
    """
    Returns values to the all slider components from the all input components
    (1st callback of the mutually dependent components: sliders and inputs)
    """
    global a_default
    return [exposed_io, lambda_io, a_io if a_io!=None else a_default, mu_io, delta_io, sample_io, forecast_term_io, inflation_parameter_io]


@app.callback(
    [Output({'type': 'exposed_io', 'index': ALL}, 'value'),
     Output({'type': 'lambda_io', 'index': ALL}, 'value'),
     Output('a_io', 'value'),
     Output('mu_io', 'value'),
     Output('delta_io', 'value'),
     Output('sample_io', 'value'),
     Output('forecast-term_io', 'value'),
     Output('inflation-parameter_io', 'value')],
    [Input({'type': 'exposed', 'index': ALL}, 'value'),
     Input({'type': 'lambda', 'index': ALL}, 'value'),
     Input('a', 'value'),
     Input('mu', 'value'),
     Input('delta', 'value'),
     Input('sample', 'value'),
     Input('forecast-term', 'value'),
     Input('inflation-parameter', 'value')],
    prevent_initial_call=True
)
def update_inputs(exposed, lambda_, a, mu, delta, sample, forecast_term, inflation_parameter):
    """
    Returns values to the all input components from the all slider components
    (2nd callback of the mutually dependent components: sliders and inputs)
    """
    global a_default
    return [exposed, lambda_, a if a!=None else a_default, mu, delta, sample, forecast_term, inflation_parameter]


# @app.callback(
#     Output('model-fit', 'figure', allow_duplicate=True),
#     Input('fit-button', 'n_clicks'),
#     State('incidence', 'value'),

#     State({'type': 'exposed_io', 'index': ALL}, 'value'),
#     State({'type': 'lambda_io', 'index': ALL}, 'value'),

#     State('a_io', 'value'),
#     State('mu_io', 'value'),
#     State('delta_io', 'value'),
#     State('sample_io', 'value'),
#     State('city', 'value'),
#     State('year', 'value'),
#     prevent_initial_call=False
# )
def update_graph(_, incidence, exposed_values,
                 lambda_values, a, mu, delta, sample_size, city, year):
    """
    Returns a figure with the depicted original data and model fit using provided parameter values
    """
    colors = px.colors.qualitative.D3
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = prepare_exposed_list(incidence, exposed_list)

    epid_data, model_obj, groups = get_data_and_model(mu, incidence, year)

    if sample_size < len(epid_data.index):
        epid_data = epid_data[:sample_size]

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 15

    xticks_vals, xticks_text = generate_xticks(epid_data, year, last_simul_ind)
    pos_x = xticks_vals[xticks_vals['year'] == year].index[-1]

    r_squared = r2_score(epid_data[groups], simul_weekly.iloc[delta:epid_data.index[-1] + 1, :],
                         multioutput='raw_values')

    fig = go.Figure()
    labels = [group if '15 и ст.' not in group
              else group.replace('15 и ст.', '15+') for group in groups]
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
    for i, r2 in enumerate(r_squared):
        fig.add_annotation(text=f'<b>$R^2={str(round(r2, 2))}$</b>',
                           showarrow=False,
                           xanchor='left',
                           xref='paper',
                           x=0.03,
                           yshift=i * (-25) + 300,
                           font={'color': colors[i], 'size': 18})
    fig.update_layout(
        template='plotly_white',
        autosize=False,
        height=700,
        margin={'l': 60, 'r': 50, 'b': 50, 't': 50, 'pad': 4},
        paper_bgcolor="white",
        title={
            'text': f"{cities[city]}, {year}-{year + 1} гг.",
            'font': {'size': 16},
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font={'size': 13},
        yaxis={'exponentformat': 'power', 'showtickprefix': 'all'},
        xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
               'tickfont': {'size': 14}, 'showgrid': False}
    )
    fig.add_vline(x=pos_x, line_width=2, line_dash="dash", line_color="green",
                  name=f'Граница  <br>{year}-{year + 1} годов',
                  showlegend=True)
    fig.update_xaxes(title_text="Недели")
    fig.update_yaxes(title_text="Количество случаев заболевания")

    return fig


@app.callback(
    Output('model-fit', 'figure'),
    Input('forecast-button', 'n_clicks'),
    State('incidence', 'value'),

    State({'type': 'exposed_io', 'index': ALL}, 'value'),
    State({'type': 'lambda_io', 'index': ALL}, 'value'),

    State('a_io', 'value'),
    State('mu_io', 'value'),
    State('delta_io', 'value'),
    State('sample_io', 'value'),
    State('city', 'value'),
    State('year', 'value'),
    State('forecast-term_io', 'value'),
    State('inflation-parameter_io', 'value'),
    State('plot-error_structures', 'value'),
    prevent_initial_call=True
)
def update_graph_predict(_, incidence, exposed_values,
                         lambda_values, a, mu, delta, sample_size, city, year,
                         forecast_term, inflation_parameter, plot_error_structures):
    """
    Returns a figure with the depicted original data, model fit and prediction
    using provided parameter values
    """
    colors = px.colors.qualitative.D3
    colors_unexistant = ["#383838", "#585858", "#696969", "#909090"]
    error_structures = ["#D3D3D3", "#E5E4E2", "#C0C0C0", "#A9A9A9"]
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = prepare_exposed_list(incidence, exposed_list)

    epid_data, model_obj, groups = get_data_and_model(mu, incidence, year)
    if sample_size > len(epid_data.index):
        return update_graph(_, incidence, exposed_values,
                            lambda_values, a, mu, delta, sample_size, city, year)

    fig = go.Figure()

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 15

    ds_amount = int(100 / len(simul_weekly.columns))

    print(epid_data.loc[:, simul_weekly.columns])
    print(simul_weekly.dropna(axis=1))
    print(ds_amount)
    print(sample_size)
    print(inflation_parameter)
    print(last_simul_ind)

    predict_gates_generator = PredictGatesGenerator(epid_data.loc[:, simul_weekly.columns],
                                                    simul_weekly.dropna(
                                                        axis=1),
                                                    ds_amount, sample_size, inflation_parameter, end=last_simul_ind)
    percentiles = [(5, 95)]
    gates = [
        predict_gates_generator.generate_predict_gate(
            p[0], p[1], length=forecast_term)
        for p in percentiles
    ]

    # if sample_size < len(epid_data.index):
    #     fig.add_vline(x=m + sample_size - 1, line_width=2, line_dash="dash", line_color="grey",
    #                   name=f'Размер выборки для прогноза <br>с {previous_index[0]+1} по {previous_index[0] + sample_size} недели',
    #                   showlegend=True)

    xticks_vals, xticks_text = generate_xticks(epid_data, year, last_simul_ind)
    pos_x = xticks_vals[xticks_vals['year'] == year].index[-1]

    r_squared = r2_score(epid_data[groups], simul_weekly.iloc[delta:epid_data.index[-1] + 1, :],
                         multioutput='raw_values')

    labels = [group if '15 и ст.' not in group
              else group.replace('15 и ст.', '15+') for group in groups]

    for i, (group, label) in enumerate(zip(simul_weekly.columns, labels)):
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

        # points of predict
        # fig.add_trace(go.Scatter(x=epid_data[group][sample_size:].index,
        #                          y=epid_data[group][sample_size:],
        #                          customdata=epid_data.loc[sample_size:, [
        #                              'Неделя']],
        #                          hovertemplate="<br>Количество заболеваний: %{y}"
        #                                        "<extra></extra>",
        #                          mode='markers+lines',
        #                          line={'dash': 'dash', 'shape': 'spline',
        #                                'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'},
        #                          legendgroup='data',
        #                          marker={
        #                              'color': colors[i], 'size': 10, 'opacity': 0.5},
        #                          showlegend=False))

        # lines of the model
        fig.add_trace(go.Scatter(x=simul_weekly[group].index[:last_simul_ind],
                                 y=simul_weekly[group][:last_simul_ind],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='lines',
                                 legendgroup='model-fit',
                                 marker={'color': colors[i], 'size': 10},
                                 name='Модель, ' + label))

        pr_m = m + sample_size - 1
        pr_n = m + sample_size + forecast_term
        if plot_error_structures:
            for simulated_error_ds in predict_gates_generator.simulated_datasets:
                fig.add_trace(go.Scatter(x=simulated_error_ds[group].index[pr_m:pr_n],
                                         y=simulated_error_ds[group][pr_m:pr_n],
                                         hovertemplate="Негативная биномиальная структура ошибок"
                                                       "<extra></extra>",
                                         mode='lines',
                                         line={
                                             'color': f'rgba{(*hex_to_rgb(error_structures[i]), 0.3)}'},
                                         showlegend=False))

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
                                     mode='lines', showlegend=False,
                                     marker={'color': "rgba(255,0,0,0.5)", 'size': 10}))

            fig.add_trace(go.Scatter(x=x_, y=y2_, fill='tonexty', fillcolor=f'rgba{(*hex_to_rgb(colors[i]), 0.3)}',
                                     mode='lines',
                                     name=f"Границы прогноза (PR: {percentiles[gate_i][0]}-{percentiles[gate_i][1]}), {label}",
                                     marker={'color': "rgba(255,0,0,0.5)", 'size': 10}))

    for i, r2 in enumerate(r_squared):
        fig.add_annotation(text=f'<b>$R^2={str(round(r2, 2))}$</b>',
                           showarrow=False,
                           xanchor='left',
                           xref='paper',
                           x=0.03,
                           yshift=i * (-25) + 300,
                           font={'color': colors[i], 'size': 18})
    fig.update_layout(
        template='plotly_white',
        autosize=False,
        height=700,
        margin={'l': 60, 'r': 50, 'b': 50, 't': 50, 'pad': 4},
        paper_bgcolor="white",
        title={
            'text': f"{cities[city]}, {year}-{year + 1} гг.",
            'font': {'size': 16},
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font={'size': 13},
        yaxis={'exponentformat': 'power', 'showtickprefix': 'all'},
        xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
               'tickfont': {'size': 14}, 'showgrid': False}
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

    return fig

'''
@app.callback(Input("ci-button", "n_clicks"),
              Output("download-ci-request-json", "data"),
              State('incidence', 'value'),

              State({'type': 'exposed_io', 'index': ALL}, 'value'),
              State({'type': 'lambda_io', 'index': ALL}, 'value'),

              State('a_io', 'value'),
              State('mu_io', 'value'),
              State('delta_io', 'value'),
              State('sample_io', 'value'),
              State('city', 'value'),
              State('year', 'value'),
              State('forecast-term_io', 'value'),
              State('inflation-parameter_io', 'value'),
              force_no_output=True, prevent_initial_call=True)
def bulletin_client_call(_, incidence, exposed_values,
                         lambda_values, a, mu, delta, sample_size, city, year, forecast_term, inflation_parameter):
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = prepare_exposed_list(incidence, exposed_list)

    epid_data, model_obj, groups = get_data_and_model(mu, incidence, year)

    # if sample_size < len(epid_data.index):
    #     epid_data = epid_data[:sample_size]

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    current_time = str(datetime.datetime.now())

    forecasting = False
    if sample_size <= len(epid_data.index):
        forecasting = True

    bulletin_request = {
        "datetime": current_time,
        "simulation_parameters": {
            "city": city,
            "city_russian": cities[city],
            "year": int(year),
            "incidence": incidence,
            "exposed": exposed_values,
            "lambda": lambda_values,
            "a": a,
            "mu": mu,
            "delta": delta,
            "sample_size": sample_size,
            "forecasting": forecasting,
            "forecast_term": forecast_term,
            "inflation_parameter": inflation_parameter
        },
        "groups": groups,
        "epid_data_pickled": jsonpickle.encode(epid_data.to_csv(index=False)),
        "model_obj_pickled": jsonpickle.encode(model_obj),
        "simul_weekly_pickled": jsonpickle.encode(simul_weekly.to_csv(index=False))
    }

    simulation_type_string = ""  # retrospective or forecast
    if forecasting:
        simulation_type_string = "_forecast"
    else:
        simulation_type_string = "_retrospective"

    # bulletin_generator.generate_bulletin()
    return dict(content=json.dumps(bulletin_request),
                filename=f"request_for_bulletin_{current_time.replace(' ', '_')}{simulation_type_string}.json")
'''

@app.callback([Output('data_components', 'children', allow_duplicate=True),
               Output('city', 'value', allow_duplicate=True),
               Output('year', 'value', allow_duplicate=True),
               Output('params-components', 'children', allow_duplicate=True)],
              Input('upload-preset', 'contents'),
              State('upload-preset', 'filename'),
              State('upload-preset', 'last_modified'),
              )
def process_preset(list_of_contents, list_of_names, list_of_dates):
    incidence_default = "age-group"
    city_default = 'spb'
    year_default = '2010'

    component_bunch = age_groups_comps.get_multi_age_c()
    global a_default
    a_default = 0.01093982936993367
    mu_default = 0.2
    delta_default = 30

    if list_of_contents is not None:
        global PRESET_MODE
        PRESET_MODE = True
        preset = json.loads(base64.b64decode(list_of_contents[29:]))
        print(preset)
        incidence_default = preset['incidence']
        city_default = preset["city"]
        year_default = preset["year"]

        a_default = preset["a"]
        mu_default = preset["mu"]
        delta_default = preset["delta"]

        exposed_def = exposed_dict_to_inputs(
            preset['exposed'], incidence_default)
        lambda_def = lambda_dict_to_inputs(preset['lambda'], incidence_default)

        if incidence_default == 'total':
            component_bunch = total_comps.get_total_c(exposed_def, lambda_def)
        elif incidence_default == 'strain':
            component_bunch = strains_comps.get_multi_strain_c(
                exposed_def, lambda_def)
        elif incidence_default == 'age-group':
            component_bunch = age_groups_comps.get_multi_age_c(
                exposed_def, lambda_def)
        elif incidence_default == 'strain_age-group':
            component_bunch = strains_age_comps.get_multi_strain_age_c(
                exposed_def, lambda_def)
        else:
            raise ValueError(f"can't parse incidence: {incidence_default}")

    return (get_data_components(incidence_default).children,
            city_default, year_default,
            get_model_params_components(
                component_bunch, a_default, mu_default, delta_default)
            .children[0].children)


@app.callback(    
    Input('calibration-button', 'n_clicks'),
    
    #Output('loading', 'children'),
    Output("download-preset", "data"),
    

    State('incidence', 'value'),

    State({'type': 'exposed_io', 'index': ALL}, 'value'),
    State({'type': 'lambda_io', 'index': ALL}, 'value'),

    State('a_io', 'value'),
    State('mu_io', 'value'),
    State('delta_io', 'value'),
    State('sample_io', 'value'),
    State('city', 'value'),
    State('year', 'value'),
    State('forecast-term_io', 'value'),
    State('inflation-parameter_io', 'value'),
    State('plot-error_structures', 'value'),

    running=[
        (Output("calibration-button", "disabled"), True, False),
    ],
    prevent_initial_call=True,
    background=True, 
)
def launch_calibration(_, incidence, exposed_values,
                         lambda_values, a, mu, delta, sample_size, city, year,
                         forecast_term, inflation_parameter, plot_error_structures):
    
    print("Calibration started...")
    ''' {'exposed': [0.30148313273350635], 
    'lambda': [0.09544288428775363], 
    'a': [0.13599275369791994], 
    'delta': 57, 
    'total_recovered': [483795.443701321], 
    'R2': [0.5166538711970509], 
    'r0': [[3.7383068917701014]]}'''
    print(incidence)
    # incidence - уровень детализации из вкладки данные
    # exposed_values - значения "Доля переболевших"
    # lambda_values - значения "Вирулентность"
    # а - значение "Доля населения, теряющего иммунитет за год"
    # mu - значение "Доля населения, потерявшего иммунитет"
    # delta - значение "Сдвиг данных"
    # sample - значение "Доступность данных" 
    current_time = str(datetime.datetime.now())

    calibration_parameters = calibration.calibration(year=year, mu=mu, incidence=incidence)
    # calibration_parameters = {'exposed': [0.30148313273350635], 
    # 'incidence': 'total',
    # 'lambda': [0.09544288428775363], 
    # 'a': [0.13599275369791994], 
    # 'delta': 57, 
    # 'total_recovered': [483795.443701321], 
    # 'R2': [0.5166538711970509], 
    # 'r0': [[3.7383068917701014]],
    # 'mu': 0.1}
    # print("Something")

    # exposed_def = exposed_dict_to_inputs(
    #     calibration_parameters['exposed'], incidence)
    # lambda_def = lambda_dict_to_inputs(calibration_parameters['lambda'], incidence)

    preset_dict = {
        "city": 'spb',
        "year": year,
        "incidence": incidence,
        "exposed": calibration_parameters['exposed'],
        "lambda": calibration_parameters['lambda'],
        "a": calibration_parameters["a"][0],
        "mu": mu,
        "delta": calibration_parameters["delta"],
    }

    return dict(content=json.dumps(preset_dict),
                filename=f"{current_time.replace(' ', '_')}_preset.json")
    # return dict(content="Hello world!", filename="hello.txt")

@app.callback(
    Output('model-fit', 'figure', allow_duplicate=True),
    Input('save_plot-button', 'n_clicks'),
    State('incidence', 'value'),

    State({'type': 'exposed_io', 'index': ALL}, 'value'),
    State({'type': 'lambda_io', 'index': ALL}, 'value'),

    State('a_io', 'value'),
    State('mu_io', 'value'),
    State('delta_io', 'value'),
    State('sample_io', 'value'),
    State('city', 'value'),
    State('year', 'value'),
    State('forecast-term_io', 'value'),
    State('inflation-parameter_io', 'value'),
    State('plot-error_structures', 'value'),
    prevent_initial_call=True
)
def save_plot(_, incidence, exposed_values,
                         lambda_values, a, mu, delta, sample_size, city, year,
                         forecast_term, inflation_parameter, plot_error_structures):
    """
    Saves plot
    """
    colors = px.colors.qualitative.D3
    colors_unexistant = ["#383838", "#585858", "#696969", "#909090"]
    error_structures = ["#D3D3D3", "#E5E4E2", "#C0C0C0", "#A9A9A9"]
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = prepare_exposed_list(incidence, exposed_list)

    epid_data, model_obj, groups = get_data_and_model(mu, incidence, year)
    if sample_size > len(epid_data.index):
        fig = update_graph(_, incidence, exposed_values,
                            lambda_values, a, mu, delta, sample_size, city, year)
        fig.write_image(os.path.join('plots',r"week_{}_forecast.png".format(incidence)))
        return fig
        

    fig = go.Figure()

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 15

    ds_amount = int(100 / len(simul_weekly.columns))

    predict_gates_generator = PredictGatesGenerator(epid_data.loc[:, simul_weekly.columns],
                                                    simul_weekly.dropna(
                                                        axis=1),
                                                    ds_amount, sample_size, inflation_parameter, end=last_simul_ind)
    percentiles = [(5, 95)]
    gates = [
        predict_gates_generator.generate_predict_gate(
            p[0], p[1], length=forecast_term)
        for p in percentiles
    ]
    xticks_vals, xticks_text = generate_xticks(epid_data, year, last_simul_ind)
    pos_x = xticks_vals[xticks_vals['year'] == year].index[-1]

    r_squared = r2_score(epid_data[groups], simul_weekly.iloc[delta:epid_data.index[-1] + 1, :],
                         multioutput='raw_values')

    labels = [group if '15 и ст.' not in group
              else group.replace('15 и ст.', '15+') for group in groups]

    for i, (group, label) in enumerate(zip(simul_weekly.columns, labels)):
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
        # lines of the model
        fig.add_trace(go.Scatter(x=simul_weekly[group].index[:last_simul_ind],
                                 y=simul_weekly[group][:last_simul_ind],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='lines',
                                 legendgroup='model-fit',
                                 marker={'color': colors[i], 'size': 10},
                                 name='Модель, ' + label))

        pr_m = m + sample_size - 1
        pr_n = m + sample_size + forecast_term
        if plot_error_structures:
            for simulated_error_ds in predict_gates_generator.simulated_datasets:
                fig.add_trace(go.Scatter(x=simulated_error_ds[group].index[pr_m:pr_n],
                                         y=simulated_error_ds[group][pr_m:pr_n],
                                         hovertemplate="Негативная биномиальная структура ошибок"
                                                       "<extra></extra>",
                                         mode='lines',
                                         line={
                                             'color': f'rgba{(*hex_to_rgb(error_structures[i]), 0.3)}'},
                                         showlegend=False))

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
                                     mode='lines', showlegend=False,
                                     marker={'color': "rgba(255,0,0,0.5)", 'size': 10}))

            fig.add_trace(go.Scatter(x=x_, y=y2_, fill='tonexty', fillcolor=f'rgba{(*hex_to_rgb(colors[i]), 0.3)}',
                                     mode='lines',
                                     name=f"Границы прогноза (PR: {percentiles[gate_i][0]}-{percentiles[gate_i][1]}), {label}",
                                     marker={'color': "rgba(255,0,0,0.5)", 'size': 10}))

    for i, r2 in enumerate(r_squared):
        fig.add_annotation(text=f'<b>$R^2={str(round(r2, 2))}$</b>',
                           showarrow=False,
                           xanchor='left',
                           xref='paper',
                           x=0.03,
                           yshift=i * (-25) + 300,
                           font={'color': colors[i], 'size': 18})
    fig.update_layout(
        template='plotly_white',
        autosize=False,
        height=700,
        margin={'l': 60, 'r': 50, 'b': 50, 't': 50, 'pad': 4},
        paper_bgcolor="white",
        title={
            'text': f"{cities[city]}, {year}-{year + 1} гг.",
            'font': {'size': 16},
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font={'size': 13},
        yaxis={'exponentformat': 'power', 'showtickprefix': 'all'},
        xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
               'tickfont': {'size': 14}, 'showgrid': False}
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
    print("Saving...")
    fig.write_image(os.path.join('plots', r"week_{}_forecast.png".format(1,incidence)))
    return fig


if __name__ == '__main__':
    app.run(debug=True)
