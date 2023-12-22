import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_components import Row, Col


def get_exposed_sliders(exposed=(0.8318945681908346, 0.4532701740867703, 0.11561966774301881)):
    return {
        'A(H1N1)': [html.P('A(H1N1)'),
                    dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                               id={'type': 'exposed', 'index': 0},
                               tooltip={"placement": "bottom"},
                               value=exposed[0])
                    ],
        'A(H3N2)': [html.P('A(H3N2)'),
                    dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                               id={'type': 'exposed', 'index': 1},
                               tooltip={"placement": "bottom"},
                               value=exposed[1])
                    ],
        'B': [html.P('B'),
              dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                         id={'type': 'exposed', 'index': 2},
                         tooltip={"placement": "bottom"},
                         value=exposed[2])
              ]
    }


def get_lambda_sliders(lambda_=(0.13516497447559267, 0.09344715673194291, 0.07693719659160894)):
    return {
        'A(H1N1)': [html.P('A(H1N1)'),
                    dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                               id={'type': 'lambda', 'index': 0},
                               tooltip={"placement": "bottom"},
                               value=lambda_[0])
                    ],
        'A(H3N2)': [html.P('A(H3N2)'),
                    dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                               id={'type': 'lambda', 'index': 1},
                               tooltip={"placement": "bottom"},
                               value=lambda_[1])
                    ],
        'B': [html.P('B'),
              dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                         id={'type': 'lambda', 'index': 2},
                         tooltip={"placement": "bottom"},
                         value=lambda_[2])
              ]
    }


def get_exposed_inputs(exposed=(0.8318945681908346, 0.4532701740867703, 0.11561966774301881)):
    return {
        'A(H1N1)': dbc.Input(id={'type': 'exposed_io', 'index': 0}, type='number',
                             value=exposed[0], style={'width': '200px'}, step=0.001),
        'A(H3N2)': dbc.Input(id={'type': 'exposed_io', 'index': 1}, type='number',
                             value=exposed[1], style={'width': '200px'}, step=0.001),
        'B': dbc.Input(id={'type': 'exposed_io', 'index': 2}, type='number',
                       value=exposed[2], style={'width': '200px'}, step=0.001)
    }


def get_lambda_inputs(lambda_=(0.13516497447559267, 0.09344715673194291, 0.07693719659160894)):
    return {
        'A(H1N1)': dbc.Input(id={'type': 'lambda_io', 'index': 0}, type='number',
                             value=lambda_[0], style={'width': '200px'}, step=0.001),
        'A(H3N2)': dbc.Input(id={'type': 'lambda_io', 'index': 1}, type='number',
                             value=lambda_[1], style={'width': '200px'}, step=0.001),
        'B': dbc.Input(id={'type': 'lambda_io', 'index': 2}, type='number',
                       value=lambda_[2], style={'width': '200px'}, step=0.001)
    }


def get_multi_strain_c(exposed_params=(0.8318945681908346, 0.4532701740867703, 0.11561966774301881),
                       lambda_params=(0.13516497447559267, 0.09344715673194291, 0.07693719659160894)):
    exposed_inputs = get_exposed_inputs(exposed_params)
    lambda_inputs = get_lambda_inputs(lambda_params)
    exposed_sliders = get_exposed_sliders(exposed_params)
    lambda_sliders = get_lambda_sliders(lambda_params)
    return {'exposed': [
        Row([
            Col(exposed_sliders['A(H1N1)']),
            Col([exposed_inputs['A(H1N1)']], align='center')
        ]),
        Row([
            Col(exposed_sliders['A(H3N2)']),
            Col([exposed_inputs['A(H3N2)']], align='center')
        ]),
        Row([
            Col(exposed_sliders['B']),
            Col([exposed_inputs['B']], align='center')
        ])
    ],
        'lambda': [
            Row([
                Col(lambda_sliders['A(H1N1)']),
                Col([lambda_inputs['A(H1N1)']], align='center')
            ]),
            Row([
                Col(lambda_sliders['A(H3N2)']),
                Col([lambda_inputs['A(H3N2)']], align='center')
            ]),
            Row([
                Col(lambda_sliders['B']),
                Col([lambda_inputs['B']], align='center')
            ]),
        ]}
