from app import app
from dash import Input, Output, State, callback, ALL
from aux_functions import prepare_exposed_list, get_data_and_model
import plotly.io as pio
import os

# CHARTING
import plot_callbacks

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
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = prepare_exposed_list(incidence, exposed_list)

    epid_data, model_obj, groups = get_data_and_model(mu, incidence, year)
    if sample_size > len(epid_data.index):
        fig = plot_callbacks.update_graph(_, incidence, exposed_values,
                           lambda_values, a, mu, delta, sample_size, city, year)[0]

    else:
        fig = plot_callbacks.update_graph_predict(_, incidence, exposed_values,
                                   lambda_values, a, mu, delta, sample_size, city, year,
                                   forecast_term, inflation_parameter, plot_error_structures)[0]
    pio.write_image(fig, os.path.join('gui/static/plots',
                    r"week_{}_forecast.png".format(incidence)), 'png', width=1598, height=700)
    return fig
