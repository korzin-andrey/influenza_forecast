from app import app
from dash import Input, Output, State, callback, ALL
import plotly.express as px
from plotly.colors import hex_to_rgb
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from toTable.toExcel import toExcel
from bootstrapping.predict_gates import PredictGatesGenerator
from aux_functions import prepare_exposed_list, get_data_and_model, transform_days_to_weeks, generate_xticks
import pandas as pd

_GENERATE = None


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

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 6

    xticks_vals, xticks_text = generate_xticks(epid_data, year, last_simul_ind)
    pos_x = xticks_vals[xticks_vals['year'] == year].index[-1]

    r_squared = r2_score(epid_data[groups], simul_weekly.iloc[delta:epid_data.index[-1] + 1, :],
                         multioutput='raw_values')

    fig = go.Figure()
    labels = [group if '15 и ст.' not in group
              else group.replace('15 и ст.', '15+') for group in groups]

    y0_rect = 0
    Data = []
    Predict = []
    for i, (group, label) in enumerate(zip(simul_weekly.columns, labels)):

        D = [0]*(delta) + list(epid_data[group])
        P = list(round(simul_weekly[group][:last_simul_ind], 0))
        D += [0]*(len(P) - len(D))
        Data.append(D)
        Predict.append(P)

        y0_rect = max(y0_rect, max(
            simul_weekly[group][:last_simul_ind]), max(epid_data[group]))
        fig.add_trace(go.Scatter(x=epid_data[group].index,
                                 y=epid_data[group],
                                 customdata=epid_data.loc[:, ['Неделя']],
                                 hovertemplate="<br>%{customdata[0]} неделя"
                                               "<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='markers+lines',
                                 legend='legend',
                                 legendgroup='data',
                                 line={'dash': 'dash', 'shape': 'spline',
                                       'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'},
                                 marker={'color': colors[i], 'size': 10, },
                                 line_shape='linear',
                                 name=label))

        fig.add_trace(go.Scatter(x=simul_weekly[group].index[:last_simul_ind],
                                 y=simul_weekly[group][:last_simul_ind],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='lines',
                                 legend='legend2',
                                 legendgroup='model',
                                 marker={'color': colors[i], 'size': 10},
                                 line_shape='spline',
                                 name=label))

    # for i, r2 in enumerate(r_squared):
    #     fig.add_annotation(text=f'<b>$R^2={str(round(r2, 2))}$</b>',
    #                        showarrow=False,
    #                        xanchor='left',
    #                        xref='paper',
    #                        x=0.03,
    #                        yshift=i * (-25) + 405,
    #                        font={'color': colors[i], 'size': 60,
    #                              "family": "Times New Roman"},
    #                        bgcolor="rgb(255, 255, 255)",
    #                        opacity=0.8)

    global _GENERATE
    _GENERATE = toExcel(incidence, exposed_values,
                        lambda_values, labels, Data, Predict)

    model_y = {"total": 0.8, "strain_age-group": 0.13,
               "strain": 0.73, "age-group": 0.785}
    data_y = {"total": 1, "strain_age-group": 1, "strain": 1, "age-group": 1}

    y0_rect = y0_rect + 1400

    # yticks_index = (y0_rect//(len(str(y0_rect))-1)) #4
    fig.update_layout(
        template='none',
        autosize=False,
        height=600,
        margin={'l': 100, 'r': 20, 'b': 65, 't': 60, 'pad': 0},
        title={
            'text': f"Российская Федерация, {year}-{year + 1} гг.",
            'font': {'size': 40},
            "font_family": "Times New Roman",
            'xanchor': 'center',
        },

        yaxis={'tickfont': {'size': 18}},
        yaxis_tickformat='0',
        xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
               'tickfont': {'size': 20}, 'showgrid': True},
        legend={
            "title": " Данные ",
            "y": data_y[incidence],
            "title_font_family": "Times New Roman",
            "font": {
                "family": "Courier",
                "size": 23,
                "color": "black"
            },
            "bgcolor": "rgb(255, 255, 255)",
            "bordercolor": "Black",
            "borderwidth": 2
        },
        legend2={
            "title": " Модель ",
            "y": model_y[incidence],
            "title_font_family": "Times New Roman",
            "font": {
                "family": "Courier",
                "size": 23,
                "color": "black"
            },
            "bgcolor": "rgb(255, 255, 255)",
            "bordercolor": "Black",
            "borderwidth": 2
        }
    )

    fig.add_trace(go.Scatter(x=[pos_x]*int(y0_rect*1.03), y=[i for i in range(int(y0_rect*1.05))],
                             line=dict(color='rgba(0, 128, 0, 0.6)',
                                       width=3, dash='dot'),
                             name='dash', showlegend=False))

    fig.add_shape(type="rect",
                  x0=0, y0=y0_rect*1.03, x1=pos_x, y1=y0_rect*1.101,
                  line=dict(
                      color="rgba(211, 211, 211, 0.6)",
                      width=2,
                  ),
                  fillcolor="rgba(135, 206, 250, 0.6)",
                  label=dict(text=f"{year}", textposition="top center",
                             font=dict(size=25))
                  )

    fig.add_shape(type="rect",
                  x0=pos_x, y0=y0_rect*1.03, y1=y0_rect*1.101, x1=len(xticks_text)-1,
                  line=dict(
                      color="rgba(211, 211, 211, 0.6)",
                      width=2,
                  ),
                  fillcolor="rgba(135, 206, 250, 0.6)",
                  label=dict(text=f"{year+1}",
                             textposition="top center", font=dict(size=25))
                  )

    # ось х
    fig.update_xaxes(title_text="Номер недели в году", title_font_size=25, showline=True, linewidth=2, linecolor='black',
                     mirror=True, zeroline=False, griddash='dash', ticks="outside", tickwidth=1, gridcolor='rgb(202, 222, 255)')

    # ось у
    fig.update_yaxes(title_text="Количество случаев заболевания", title_font_size=25, showline=True, linewidth=2, linecolor='black',
                     mirror=True, zeroline=False, griddash='dash', ticks="outside", tickwidth=1, gridcolor='rgb(202, 222, 255)')

    return fig, False


@app.callback(
    Output('model-fit', 'figure'),
    Output('excel-button', 'disabled'),
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
        print("Retrospective plot")
        return update_graph(_, incidence, exposed_values,
                            lambda_values, a, mu, delta, sample_size, city, year)
    last_week_number = epid_data.loc[epid_data.index[-1], 'Неделя']
    print(last_week_number)
    for i in range(forecast_term):
        last_week_number += 1
        epid_data.loc[len(epid_data)] = 0
        epid_data.loc[epid_data.index[-1], 'Неделя'] = last_week_number
    print(epid_data.head(30))
    print(len(epid_data.index))
    print("Forecast plot")
    fig = go.Figure()

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    epid_data.index = epid_data.reset_index().index + delta
    m, n = epid_data.index[0], epid_data.index[-1]
    last_simul_ind = n + 20

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

    y0_rect = 0
    x0_start = float('inf')
    x1_end = 0
    Data = []
    Predict = []
    for i, (group, label) in enumerate(zip(simul_weekly.columns, labels)):

        D = [0]*(delta) + list(epid_data[group][:sample_size])
        P = list(round(simul_weekly[group][:last_simul_ind], 0))
        D += [0]*(len(P) - len(D))
        Data.append(D)
        Predict.append(P)

        y0_rect = max(y0_rect, max(
            simul_weekly[group][:last_simul_ind]), max(epid_data[group]))
        x0_start = min(x0_start, epid_data[group][:sample_size].index[0])
        x1_end = max(x1_end, epid_data[group][sample_size:].index[-1])
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
                                 legend='legend3',
                                 legendgroup='data',
                                 line={'dash': 'dash', 'shape': 'spline',
                                       'color': f'rgba{(*hex_to_rgb(colors[i])  , 0.5)}'},
                                 marker={'color': colors[i], 'size': 10, },
                                 name="ВСЕ ВОЗРАСТЫ"))

        # # points of predict - NO NEED TO SHOW, actually we do not know them
        # fig.add_trace(go.Scatter(x=epid_data[group][sample_size:].index,
        #                           y=epid_data[group][sample_size:],
        #                           customdata=epid_data.loc[sample_size:, [
        #                               'Неделя']],
        #                           hovertemplate="<br>Количество заболеваний: %{y}"
        #                                         "<extra></extra>",
        #                           mode='markers+lines',
        #                           line={'dash': 'dash', 'shape': 'spline',
        #                                 'color': f'rgba{(*hex_to_rgb(colors[i]), 0.5)}'},
        #                           legend='legend',
        #                           legendgroup='data',
        #                           marker={
        #                               'color': colors[i], 'size': 10, 'opacity': 0.5},
        #                           showlegend=False))

        # lines of the model
        fig.add_trace(go.Scatter(x=simul_weekly[group].index[:last_simul_ind],
                                 y=simul_weekly[group][:last_simul_ind],
                                 hovertemplate="<br>Количество заболеваний: %{y}"
                                               "<extra></extra>",
                                 mode='lines',
                                 showlegend=False,
                                 marker={'color': colors[i], 'size': 10},
                                 name="ВСЕ ВОЗРАСТЫ"))

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
            y0_rect = max(y0_rect, max(predict_gate.y_max))
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
                                     marker={'color': f'rgba{(*hex_to_rgb(colors[i]), 0.3)}', 'size': 10}))

            fig.add_trace(go.Scatter(x=x_, y=y2_, fill='tonexty', fillcolor=f'rgba{(*hex_to_rgb(colors[i]), 0.3)}',
                                     mode='lines',
                                     name=f"ВСЕ ВОЗРАСТЫ",
                                     marker={'color': f'rgba{(*hex_to_rgb(colors[i]), 0.3)}', 'size': 10}))

    global _GENERATE
    _GENERATE = toExcel(incidence, exposed_values,
                        lambda_values, labels, Data, Predict)

    # for i, r2 in enumerate(r_squared):
    #     fig.add_annotation(text=f'<b>$R^2={str(round(r2, 2))}$</b>',
    #                        showarrow=False,
    #                        xanchor='left',
    #                        xref='paper',
    #                        x=0.03,
    #                        yshift=i * (-25) + 200,
    #                        font={'color': colors[i], 'size': 18,
    #                              "family": "Courier New, monospace"},
    #                        bgcolor="rgb(255, 255, 255)",
    #                        opacity=0.8)

    model_y = {"total": 0.8, "strain_age-group": 0.125,
               "strain": 0.73, "age-group": 0.785}
    data_y = {"total": 1, "strain_age-group": 1, "strain": 1, "age-group": 1}

    fig.update_layout(
        template='none',
        autosize=False,
        height=600,
        margin={'l': 100, 'r': 20, 'b': 65, 't': 60, 'pad': 0},
        title={
            'text': f"Российская Федерация, {year}-{year + 1} гг.",
            'font': {'size': 40},
            "font_family": "Times New Roman",
            'xanchor': 'center',
        },
        font={'size': 13},
        yaxis={'tickfont': {'size': 18}},
        yaxis_tickformat='0',
        xaxis={'tickvals': xticks_vals.index, 'ticktext': xticks_text, 'tickangle': 0,
               'tickfont': {'size': 20}, 'showgrid': True},
        legend={
            "title": " Модель ",
            "y": model_y[incidence],
            "title_font_family": "Times New Roman",
            "font": {
                "family": "Times New Roman",
                "size": 23,
                "color": "black"
            },
            "bgcolor": "rgb(255, 255, 255)",
            "bordercolor": "Black",
            "borderwidth": 2
        },
        legend3={
            "title": " Данные ",
            "y": data_y[incidence],
            "title_font_family": "Times New Roman",
            "font": {
                "family": "Times New Roman",
                "size": 23,
                "color": "black"
            },
            "bgcolor": "rgb(255, 255, 255)",
            "bordercolor": "Black",
            "borderwidth": 2
        },
        legend4={
            "title": " Область ",
            "y": 0.9,
            "x": 0.005,
            "title_font_family": "Times New Roman",
            "font": {
                "family": "Times New Roman",
                "size": 23,
                "color": "black"
            },
            "bgcolor": "rgb(255, 255, 255)",
            "bordercolor": "Black",
            "borderwidth": 2,
        }
    )

    border_of_data = m + sample_size - 1

    y0_rect = y0_rect + 1400

    # data separation line
    fig.add_trace(go.Scatter(x=[border_of_data]*(int(y0_rect*1.03)), y=[i for i in range(int(y0_rect*1.03))],
                             line=dict(color='rgba(243, 21, 21, 0.6)',
                                       width=3, dash='dot'),
                             name='dash', showlegend=False))

    # area with known data
    fig.add_shape(type="rect",
                  x0=x0_start, y0=0, x1=border_of_data, y1=y0_rect*1.03,
                  line=dict(
                      width=0,
                  ),
                  fillcolor="rgba(225, 255, 221, 0.35)",
                  )

    fig.add_shape(type="rect",
                  x0=0, y0=0, x1=0, y1=0,
                  line=dict(
                      width=0,
                  ),
                  fillcolor="rgb(199, 255, 194)",
                  showlegend=True,
                  legend='legend4',
                  name='известных данных'
                  )

    # area with unknowm data
    fig.add_shape(type="rect",
                  x0=border_of_data, y0=0, x1=x1_end, y1=y0_rect*1.03,
                  line=dict(
                      width=0,
                  ),
                  fillcolor="rgba(255, 224, 224, 0.3)",
                  )

    fig.add_shape(type="rect",
                  x0=0, y0=0, x1=0, y1=0,
                  line=dict(
                      width=0,
                  ),
                  fillcolor="rgb(255, 197, 197)",
                  showlegend=True,
                  legend='legend4',
                  name='без данных'
                  )

    # years separation line
    fig.add_trace(go.Scatter(x=[pos_x]*int(y0_rect*1.03), y=[i for i in range(int(y0_rect*1.03))],
                             line=dict(color='rgba(0, 128, 0, 0.6)',
                                       width=3, dash='dot'),
                             name='dash', showlegend=False))

    # rectangle with year
    fig.add_shape(type="rect",
                  x0=0, y0=y0_rect*1.03, x1=pos_x, y1=y0_rect*1.101,
                  line=dict(
                      color="rgba(211, 211, 211, 0.6)",
                      width=2,
                  ),
                  fillcolor="rgba(135, 206, 250, 0.6)",
                  label=dict(text=f"{year}", textposition="top center",
                             font=dict(size=25))
                  )

    # rectangle with year+1
    fig.add_shape(type="rect",
                  x0=pos_x, y0=y0_rect*1.03, y1=y0_rect*1.101, x1=len(xticks_text)-1,
                  line=dict(
                      color="rgba(211, 211, 211, 0.6)",
                      width=2,
                  ),
                  fillcolor="rgba(135, 206, 250, 0.6)",
                  label=dict(text=f"{year+1}",
                             textposition="top center", font=dict(size=25))
                  )

    fig.update_xaxes(title_text="Номер недели в году", title_font_size=25, showline=True, linewidth=2, linecolor='black',
                     mirror=True, zeroline=False, griddash='dash', ticks="outside", tickwidth=1, gridcolor='rgb(202, 222, 255)')

    fig.update_yaxes(title_text="Количество случаев заболевания", title_font_size=25, showline=True, linewidth=2, linecolor='black',
                     mirror=True, zeroline=False, griddash='dash', ticks="outside", tickwidth=1, gridcolor='rgb(202, 222, 255)')
    return fig, False
