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


PRESET_MODE = False

# UPDATING COMPONENTS


def get_update_callbacks(app):
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
             Output({'type': 'exposed_io', 'index': ALL}, 'value'),
             Output({'type': 'lambda_io', 'index': ALL}, 'value')],

            [Input({'type': 'exposed_io', 'index': ALL}, 'value'),
             Input({'type': 'lambda_io', 'index': ALL}, 'value'),
             Input({'type': 'exposed', 'index': ALL}, 'value'),
             Input({'type': 'lambda', 'index': ALL}, 'value')],
            prevent_initial_call=True
        )
        def update_exposed_and_lambda(exposed_io, lambda_io, exposed, lambda_):
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if "exposed" in trigger_id:
                if "io" in trigger_id:
                    exposed = exposed_io
                else:
                    exposed_io = exposed
            else:
                if "io" in trigger_id:
                    lambda_ = lambda_io
                else:
                    lambda_io = lambda_

            return exposed, lambda_, exposed_io, lambda_io

        @app.callback(
            [Output('a', 'value'),
             Output('mu', 'value'),
             Output('delta', 'value'),
             Output('sample', 'value'),
             Output('forecast-term', 'value'),
             Output('inflation-parameter', 'value')],
            [CycleBreakerInput('a_io', 'value'),
             CycleBreakerInput('mu_io', 'value'),
             CycleBreakerInput('delta_io', 'value'),
             CycleBreakerInput('sample_io', 'value'),
             CycleBreakerInput('forecast-term_io', 'value'),
             CycleBreakerInput('inflation-parameter_io', 'value')],
            prevent_initial_call=True
        )
        def update_sliders(a_io, mu_io, delta_io, sample_io, forecast_term_io, inflation_parameter_io):
            """
            Returns values to the all slider components from the all input components
            (1st callback of the mutually dependent components: sliders and inputs)
            """
            return [a_io, mu_io, delta_io, sample_io, forecast_term_io, inflation_parameter_io]

        @app.callback(
            [Output('a_io', 'value'),
             Output('mu_io', 'value'),
             Output('delta_io', 'value'),
             Output('sample_io', 'value'),
             Output('forecast-term_io', 'value'),
             Output('inflation-parameter_io', 'value')],
            [Input('a', 'value'),
             Input('mu', 'value'),
             Input('delta', 'value'),
             Input('sample', 'value'),
             Input('forecast-term', 'value'),
             Input('inflation-parameter', 'value')],
            prevent_initial_call=True
        )
        def update_inputs(a, mu, delta, sample, forecast_term, inflation_parameter):
            """
            Returns values to the all input components from the all slider components
            (2nd callback of the mutually dependent components: sliders and inputs)
            """
            return [a, mu, delta, sample, forecast_term, inflation_parameter]
