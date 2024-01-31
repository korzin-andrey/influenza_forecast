from flask import send_from_directory
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
from gui import app
import diskcache
from utils.experiment_setup import ExperimentalSetup
from aux_functions import get_contact_matrix, prepare_calibration_data, get_config
from optimizers.multiple_model_fit import MultipleModelFit
from dash.long_callback import DiskcacheLongCallbackManager
import dash
import plotly.io as pio
import signal
from optimizers import multiple_model_fit
from dash.exceptions import PreventUpdate
import io
import pandas as pd
from data import excel_preprocessing


def get_calibration_callbacks(app):
    @app.callback(Output('output-data-upload', 'children'),
                  Input('upload-source', 'contents'),
                  State('upload-preset', 'filename'),
                  State('upload-preset', 'last_modified'),
                  prevent_initial_call=True,
                  )
    def process_upload_data(contents, list_of_names, list_of_dates):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded), skiprows=[0])
        excel_preprocessing.preprocess_excel_source(df)
        children = ""
        return children

    @app.callback(
        Input('calibration-button-stop', 'n_clicks'),
        Output('calibration-button', 'n_clicks'),
        prevent_initial_call=True,
    )
    def stop_calibration(_):
        qe = multiple_model_fit.queue
        while not qe.empty():
            os.kill(qe.get(), signal.SIGKILL)
        if True:
            raise PreventUpdate

    @app.callback(
        Input('calibration-button', 'n_clicks'),

        # Output('loading', 'children'),
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
            (Output("calibration-button-stop", "disabled"), False, True)
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

        calibration_parameters = calibration.calibration(
            year=year, mu=mu, incidence=incidence, sample_size=sample_size)

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
