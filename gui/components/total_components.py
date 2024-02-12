import dash_bootstrap_components as dbc
from dash import dcc
from dash_bootstrap_components import Row, Col


def get_exposed_sliders(exposed=(0.3525758329656415,)):
    _min = round(exposed[0]-0.0005, 4)
    _max = round(exposed[0]+0.0005, 4)
    return dcc.Slider(min=_min, max=_max, step=0.0001, marks={_min: f'{_min}', _max: f'{_max}'},
                      id={'type': 'exposed', 'index': 0},
                      tooltip={"placement": "bottom"}, value=exposed[0])


def get_lambda_sliders(lambda_=(0.1109152781016393,)):
    _min = round(lambda_[0]-0.00005, 5)
    _max = round(lambda_[0]+0.00005, 5)
    return dcc.Slider(min=_min, max=_max, step=0.00001, marks={_min: f'{_min}', _max: f'{_max}'},
                      id={'type': 'lambda', 'index': 0},
                      tooltip={"placement": "bottom"}, value=lambda_[0])


def get_exposed_inputs(exposed=(0.3525758329656415,)):
    return dbc.Input(id={'type': 'exposed_io', 'index': 0}, type='number',
                     value=exposed[0], style={'width': '170px'})


def get_lambda_inputs(lambda_=(0.1109152781016393,)):
    return dbc.Input(id={'type': 'lambda_io', 'index': 0}, type='number',
                     value=lambda_[0], style={'width': '170px'})


def get_total_c(exposed_params=(0.3525758329656415,), lambda_params=(0.1109152781016393,)):
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
