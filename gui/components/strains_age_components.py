import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_components import Row, Col


def get_exposed_sliders(exposed=(0.26934818007330236, 0.9,  0.24085705533787935,
                                 0.8835544191576892, 0.5010795376126536, 0.005)):
    return {
        '0-14': {
            'A(H1N1)': [
                        html.P('0-14: A(H1N1)'),
                        dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                                   id={'type': 'exposed', 'index': 0},
                                   tooltip={"placement": "bottom"},
                                   value=exposed[0])
                        ],
            'A(H3N2)': [html.P('0-14: A(H3N2)'),
                        dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                                   id={'type': 'exposed', 'index': 1},
                                   tooltip={"placement": "bottom"},
                                   value=exposed[1])
                        ],
            'B': [html.P('0-14: B'),
                  dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                             id={'type': 'exposed', 'index': 2},
                             tooltip={"placement": "bottom"},
                             value=exposed[2])
                  ]
        },
        '15+': {
            'A(H1N1)': [
                        html.P('15+: A(H1N1)'),
                        dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                                   id={'type': 'exposed', 'index': 3},
                                   tooltip={"placement": "bottom"},
                                   value=exposed[3])
                        ],
            'A(H3N2)': [html.P('15+: A(H3N2)'),
                        dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                                   id={'type': 'exposed', 'index': 4},
                                   tooltip={"placement": "bottom"},
                                   value=exposed[4])
                        ],
            'B': [html.P('15+: B'),
                  dcc.Slider(min=0, max=1, marks={0: '0', 1: '1'},
                             id={'type': 'exposed', 'index': 5},
                             tooltip={"placement": "bottom"},
                             value=exposed[5])
                  ]
        }
    }


def get_lambda_sliders(lambda_=(0.3, 0.09659722555326276, 0.11197409482841345)):
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


def get_exposed_inputs(exposed=(0.26934818007330236, 0.9, 0.24085705533787935,
                                0.8835544191576892, 0.5010795376126536, 0.005)):
    return {
        '0-14': {
            'A(H1N1)': dbc.Input(id={'type': 'exposed_io', 'index': 0}, type='number',
                                 value=exposed[0], style={'width': '200px'}, step=0.001),
            'A(H3N2)': dbc.Input(id={'type': 'exposed_io', 'index': 1}, type='number',
                                 value=exposed[1], style={'width': '200px'}, step=0.001),
            'B': dbc.Input(id={'type': 'exposed_io', 'index': 2}, type='number',
                           value=exposed[2], style={'width': '200px'}, step=0.001)
        },
        '15+': {
            'A(H1N1)': dbc.Input(id={'type': 'exposed_io', 'index': 3}, type='number',
                                 value=exposed[3], style={'width': '200px'}, step=0.001),
            'A(H3N2)': dbc.Input(id={'type': 'exposed_io', 'index': 4}, type='number',
                                 value=exposed[4], style={'width': '200px'}, step=0.001),
            'B': dbc.Input(id={'type': 'exposed_io', 'index': 5}, type='number',
                           value=exposed[5], style={'width': '200px'}, step=0.001)
        }
    }


def get_lambda_inputs(lambda_=(0.3, 0.09659722555326276, 0.11197409482841345)):
    return {
        'A(H1N1)': dbc.Input(id={'type': 'lambda_io', 'index': 0}, type='number',
                             value=lambda_[0], style={'width': '200px'}, step=0.001),
        'A(H3N2)': dbc.Input(id={'type': 'lambda_io', 'index': 1}, type='number',
                             value=lambda_[1], style={'width': '200px'}, step=0.001),
        'B': dbc.Input(id={'type': 'lambda_io', 'index': 2}, type='number',
                       value=lambda_[2], style={'width': '200px'}, step=0.001)
    }


def get_multi_strain_age_c(exposed_params=(0.26934818007330236, 0.9, 0.24085705533787935,
                                           0.8835544191576892, 0.5010795376126536, 0.005),
                           lambda_params=(0.3, 0.09659722555326276, 0.11197409482841345)):
    exposed_inputs = get_exposed_inputs(exposed_params)
    lambda_inputs = get_lambda_inputs(lambda_params)
    exposed_sliders = get_exposed_sliders(exposed_params)
    lambda_sliders = get_lambda_sliders(lambda_params)
    return {'exposed': [
        Row([
            Col(exposed_sliders['0-14']['A(H1N1)']),
            Col([exposed_inputs['0-14']['A(H1N1)']], align='center')
        ]),
        Row([
            Col(exposed_sliders['0-14']['A(H3N2)']),
            Col([exposed_inputs['0-14']['A(H3N2)']], align='center')
        ]),
        Row([
            Col(exposed_sliders['0-14']['B']),
            Col([exposed_inputs['0-14']['B']], align='center')
        ]),
        Row([
            Col(exposed_sliders['15+']['A(H1N1)']),
            Col([exposed_inputs['15+']['A(H1N1)']], align='center')
        ]),
        Row([
            Col(exposed_sliders['15+']['A(H3N2)']),
            Col([exposed_inputs['15+']['A(H3N2)']], align='center')
        ]),
        Row([
            Col(exposed_sliders['15+']['B']),
            Col([exposed_inputs['15+']['B']], align='center')
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
