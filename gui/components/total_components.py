import dash_bootstrap_components as dbc
from dash import dcc
from dash_bootstrap_components import Row, Col


def get_exposed_sliders(exposed=(0.6081385649385342,)):
    return dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                      id={'type': 'exposed', 'index': 0},
                      tooltip={"placement": "bottom"}, value=exposed[0])


def get_lambda_sliders(lambda_=(0.16707651314610894,)):
    return dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                      id={'type': 'lambda', 'index': 0},
                      tooltip={"placement": "bottom"}, value=lambda_[0])


def get_exposed_inputs(exposed=(0.6081385649385342,)):
    return dbc.Input(id={'type': 'exposed_io', 'index': 0}, type='number',
                     value=exposed[0], style={'width': '170px'})


def get_lambda_inputs(lambda_=(0.16707651314610894,)):
    return dbc.Input(id={'type': 'lambda_io', 'index': 0}, type='number',
                     value=lambda_[0], style={'width': '170px'})


def get_total_c(exposed_params=(0.6081385649385342,), lambda_params=(0.16707651314610894,)):
    exposed_inputs = get_exposed_inputs(exposed_params)
    lambda_inputs = get_lambda_inputs(lambda_params)
    exposed_sliders = get_exposed_sliders(exposed_params)
    lambda_sliders = get_lambda_sliders(lambda_params)
    return {'exposed': [
        Row([
            Col([exposed_sliders]),
            Col([exposed_inputs], align='center')
        ])
    ],
        'lambda': [
            Row([
                Col([lambda_sliders]),
                Col([lambda_inputs], align='center')
            ])
        ]
    }
