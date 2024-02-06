from flask import send_from_directory
from dash import State, ALL
from dash import Dash, Input, Output, ctx, html, dcc, callback
from dash_extensions.enrich import Output, Input, CycleBreakerTransform, CycleBreakerInput

from aux_functions import prepare_exposed_list, get_data_and_model, transform_days_to_weeks, cities, generate_xticks, \
    exposed_dict_to_inputs, lambda_dict_to_inputs
from components import multi_strain, multi_age, multi_strain_age, total_c
from layout import get_model_params_components, get_data_components, get_model_advance_params
import bulletin.bulletin_generator as bulletin_generator
import time
import os

from utils.experiment_setup import ExperimentalSetup
from aux_functions import get_contact_matrix, prepare_calibration_data, get_config

import plotly.io as pio



from app import app

_GENERATE = None


# UPDATING COMPONENTS
import update_callbacks



@app.callback(
    Output("offcanvas", "is_open"),
    Input("advance_setting", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

# CHARTING
import plot_callbacks

@app.callback(
    Input('excel-button', 'n_clicks'),

    Output('model-fit', 'figure', allow_duplicate=True),

    prevent_initial_call=True,
)
def excel_create(_):

    global _GENERATE
    _GENERATE.generate()

    time.sleep(3)

    if True:
        raise PreventUpdate
    return 0


# GENERATE CLIENT BULLETIN
import bulletin_callback


# PRESET PROCESS
import preset_callback

# DATA UPLOAD PROCESS
import data_upload_callback


# CALIBRATION START AND STOP
import calibration_callbacks


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
        fig = update_graph(_, incidence, exposed_values,
                           lambda_values, a, mu, delta, sample_size, city, year)[0]

    else:
        fig = update_graph_predict(_, incidence, exposed_values,
                                   lambda_values, a, mu, delta, sample_size, city, year,
                                   forecast_term, inflation_parameter, plot_error_structures)[0]
    pio.write_image(fig, os.path.join('gui/static/plots',
                    r"week_{}_forecast.png".format(incidence)), 'png', width=1598, height=700)
    return fig


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8050)
