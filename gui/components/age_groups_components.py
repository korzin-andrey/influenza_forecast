import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_components import Row, Col


def get_exposed_sliders(exposed=(0.005, 0.7)):
    return {
        '0-14': [html.P('0-14 лет'),
                 dcc.Slider(min=0, max=1, step=10e-7, marks={0: '0', 1: '1'},
                            id={'type': 'exposed', 'index': 0}, value=exposed[0],
                            tooltip={"placement": "bottom"})],

        '15+': [html.P('15+ лет'),
                dcc.Slider(min=0, max=1, step=10e-7, marks={0: '0', 1: '1'},
                           id={'type': 'exposed', 'index': 1}, value=exposed[1],
                           tooltip={"placement": "bottom"})]
    }


def get_lambda_sliders(lambda_=(0.3400433789879575,)):
    return {
        'generic': dcc.Slider(min=0, max=1.5, step=10e-7, marks={0: '0', 1.5: '1.5'},
                              id={'type': 'lambda', 'index': 0}, value=lambda_[0],
                              tooltip={"placement": "bottom"})
    }


def get_exposed_inputs(exposed=(0.005, 0.7)):
    return {
        '0-14': dbc.Input(id={'type': 'exposed_io', 'index': 0},
                          value=exposed[0], type='number', style={'width': '200px'}),
        '15+': dbc.Input(id={'type': 'exposed_io', 'index': 1},
                         value=exposed[1], type='number', style={'width': '200px'})
    }


def get_lambda_inputs(lambda_=(0.3400433789879575,)):
    return {'generic': dbc.Input(id={'type': 'lambda_io', 'index': 0}, type='number',
                                 value=lambda_[0], style={'width': '200px'})}


def get_multi_age_c(exposed_params=(0.005, 0.7), lambda_params=(0.3400433789879575,)):
    exposed_inputs = get_exposed_inputs(exposed_params)
    lambda_inputs = get_lambda_inputs(lambda_params)
    exposed_sliders = get_exposed_sliders(exposed_params)
    lambda_sliders = get_lambda_sliders(lambda_params)
    return {'exposed': [
        Row([
            Col(exposed_sliders['0-14']),
            Col([exposed_inputs['0-14']], align='center')
        ]),
        Row([
            Col(exposed_sliders['15+']),
            Col([exposed_inputs['15+']], align='center')
        ])],
        'lambda': [
            Row([
                Col(lambda_sliders['generic']),
                Col(lambda_inputs['generic'], align='center')
            ])
        ]
    }
