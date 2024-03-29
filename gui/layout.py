import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_components import Row, Col
import plotly.graph_objects as go

from components import total_c

mode_bar_buttons_to_remove = ['autoScale2d',
                              'pan2d', 'zoom2d', 'select2d', 'lasso2d']
config = dict(displaylogo=False, responsive=True,
              modeBarButtonsToRemove=mode_bar_buttons_to_remove)

cities = [html.P('География', style={'margin': '20px 20px'}),
          dcc.RadioItems(options=[{'label': 'Россия', 'value': 'spb'},
                                  #   {'label': 'Москва', 'value': 'msc',
                                  #       'disabled': True},
                                  #   {'label': 'Новосибирск', 'value': 'novosib', 'disabled': True}
                                  ],
                         value='spb',
                         inputStyle={"marginRight": "10px",
                                     "marginLeft": "5px"},
                         style={'margin': '10px 0px'},
                         id='city')]

years = [html.P('Год', id='yearT'),
         dcc.Dropdown(options=['2010', '2011', '2012', '2013', '2014',
                               '2015', '2016', '2017', '2018', '2019',
                               '2020', '2021', '2022', '2023'],
                      value='2010',
                      id='year',
                      clearable=False)]


def get_incidence_type(default):
    return [html.P('Уровень детализированности', id='detail'),
            dcc.Dropdown(options=[
                # {'label': 'возрастные группы', 'value': 'age-group'},
                #   {'label': 'штаммы', 'value': 'strain'},
                #   {'label': 'возрастные группы и штаммы',
                #   'value': 'strain_age-group'},
                {'label': 'агрегированные данные', 'value': 'total'}],
        value=default, id='incidence', clearable=False)]


'''
        ***OLD INTERFACE VERSION***
def get_data_components(incidence_type_init):
    return Row([
        # yet without cities, only country
        Col([*cities], md=3),
        Col([*years, *get_incidence_type(incidence_type_init)], md=4)
    ], id='data_components', justify='left')
'''


def get_data_components(incidence_type_init):
    return Row([
        Col([*cities], lg=3, width=4, id="geoCol"),
        Col([*years], lg=4, width=4, id='yearCol'),
        Col([*get_incidence_type(incidence_type_init)],
            lg=5, width=4, id='incidenceCol')
    ], id='data_components', justify='left')


data_components = get_data_components('total')


def get_a_layout(value):
    _min = round(value-0.0005, 4)
    _max = round(value+0.0005, 4)
    return Row([
        Col([
            dcc.Slider(min=_min, max=_max, step=0.0001, marks={_min: f'{_min}', _max: f'{_max}'},
                       id='a', tooltip={"placement": "bottom"}, value=value)
        ]),
        Col([
            dbc.Input(id='a_io', type='number',
                      value=value, style={'width': '170px'})
        ], align='end')
    ])


def get_mu_layout(value):
    _min = round(value-0.0005, 4)
    _max = round(value+0.0005, 4)
    return Row([
        Col([
            dcc.Slider(min=_min, max=_max, step=0.0001, marks={_min: f'{_min}', _max: f'{_max}'},
                       id='mu', tooltip={"placement": "bottom"}, value=value)
        ]),
        Col([
            dbc.Input(id='mu_io', type='number',
                      value=value, style={'width': '170px'})
        ], align='end')
    ])


def get_delta_layout(value):
    return Row([
        Col([
            dcc.Slider(min=0, max=200, step=1, marks={0: '0', 100: '100', 200: '200'}, id='delta',
                       tooltip={"placement": "bottom"}, value=value),
        ]),
        Col([
            dbc.Input(id='delta_io', type='number',
                      value=value, style={'width': '170px'})
        ], align='end')
    ])


sample = Row([
    html.P('Размер выборки'),
    Col([
        dcc.Slider(min=2, max=52, step=1, marks={2: '2', 52: '52'}, id='sample',
                   tooltip={"placement": "bottom"}, value=52),
    ]),
    Col([
        dbc.Input(id='sample_io', type='number',
                  value=52, style={'width': '170px'})
    ], align='end')
])

forecasting = Row([
    html.P('Длительность прогноза'),
    Col([
        dcc.Slider(min=1, max=13, step=1, marks={1: '1', 13: '13'}, id='forecast-term',
                   tooltip={"placement": "bottom"}, value=4),
    ]),
    Col([
        dbc.Input(id='forecast-term_io', type='number',
                  value=4, style={'width': '170px'})
    ], align='end'),

    html.P('Параметр инфляции'),
    Col([
        dcc.Slider(min=0, max=2, step=0.05, marks={0: '0', 1: '1', 2: '2'}, id='inflation-parameter',
                   tooltip={"placement": "bottom"}, value=1),
    ]),
    Col([
        dbc.Input(id='inflation-parameter_io', type='number',
                  value=1, style={'width': '170px'})
    ], align='end'),
    dbc.Checklist(
        options=[
            {"label": "Визуализировать структуру ошибок", "value": 1},
        ],
        value=[],
        id="plot-error_structures",
        switch=True
    ),
])

'''
        ***OLD INTERFACE VERSION***
def get_model_params_components(components_inc, a_default=0.01093982936993367, mu_default=0.2, delta_default=30):
    return Row([
        Col([
            dbc.Accordion([
                dbc.AccordionItem(components_inc['exposed'],
                                  title='Доля переболевших',
                                  id='exposed-accordion-item'),
                dbc.AccordionItem(components_inc['lambda'],
                                  title='Вирулентность',
                                  id='lambda-accordion-item'),
                dbc.AccordionItem(
                    [get_a_layout(a_default)], title='Доля населения, теряющего иммунитет за год'),
                dbc.AccordionItem([get_mu_layout(mu_default)],
                                  title='Доля населения, потерявшего иммунитет'),
                dbc.AccordionItem(
                    [get_delta_layout(delta_default)], title='Сдвиг данных'),
                dbc.AccordionItem([sample], title='Доступность данных'),
                dbc.AccordionItem([forecasting], title='Прогнозирование')
            ], start_collapsed=False, style={'margin': '20px', 'padding': '0px'})
        ], id='params-components'),
    ])
'''


def get_model_params_components(components_inc, a_default=0.01093982936993367, mu_default=0.2):
    return dbc.ListGroup([

        dbc.ListGroupItem([
            Row(html.H6('Доля переболевших'), style={
                'margin': '5px 0px 15px 0px'}),
            Row(components_inc['exposed'], id='exposed-accordion-item')
        ], style={'margin': '20px 0px 0px 0px'}),

        dbc.ListGroupItem([
            Row(html.H6('Вирулентность'), style={
                'margin': '5px 0px 15px 0px'}),
            Row(components_inc['lambda'], id='lambda-accordion-item')
        ]),

        dbc.ListGroupItem([
            Row(html.H6('Доля населения, теряющего иммунитет за год'),
                style={'margin': '5px 0px 15px 0px'}),
            Row(get_a_layout(a_default))
        ]),

        dbc.ListGroupItem([
            Row(html.H6('Доля населения, потерявшего иммунитет'),
                style={'margin': '5px 0px 15px 0px'}),
            Row(get_mu_layout(mu_default))
        ]),
    ], id='params-components')


def get_model_advance_params(delta_default=30):
    return dbc.ListGroup([
        dbc.ListGroupItem([
            Row(html.H6('Сдвиг данных')),
            Row(get_delta_layout(delta_default))
        ]),

        dbc.ListGroupItem([
            Row(html.H6('Доступность данных')),
            Row(sample)
        ]),

        dbc.ListGroupItem([
            Row(html.H6('Прогнозирование')),
            Row(forecasting)
        ])
    ], id='params-components-advance')


model_components = get_model_params_components(total_c)

buttons = \
    Row([
        Col([
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.Div(
                            dbc.Spinner(
                                dcc.Download(id="download-preset"),
                                size="lg", color='#ADFF2F'
                            ), style={"position": "relative", "top": "50%"}
                        ), html.P("Запустить калибровку", id='calibration-button-text')
                    ], id='calibration-button'),

                    dbc.Button(html.P('Остановить калибровку', id='calibration-button-stop-text'),
                               id='calibration-button-stop', disabled=True),
                    dbc.Button(html.P('Запустить моделирование', id='forecast-button-text'),
                               id='forecast-button'),
                    dbc.Button(
                        html.P('Сохранить график', id='save_plot-button-text'), id='save_plot-button'),
                    dbc.Button(html.P('Создать таблицу', id='excel-button-text'),
                               id='excel-button', disabled=True),
                    # dbc.Button('Построить прогноз', id='forecast-button'),
                    #             dcc.Loading(
                    #     id="loading",
                    #     type="circle",
                    #     children=html.Div(id="loading"),
                    #     fullscreen=True
                    # ),
                    # html.Div([
                    #     dbc.Button('Генерация бюллетеня', id='ci-button'),
                    #     dcc.Download(id="download-ci-request-json")],)
                ], className='buttons-group')
            ], style={'marginTop': '20px', "height": "15%"})
        ])
    ], id='buttonRow')

preset_components = html.Div([
    dcc.Upload(id="upload-preset",
               children=html.Div(["Перетащите или ",
                                  html.A("выберите файл", href=""),
                                  " в формате JSON"]),
               style={
                   'width': '100%',
                   'height': '60px',
                   'lineHeight': '60px',
                   'borderWidth': '1px',
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   'margin': '10px'
               },
               multiple=False,
               accept='.json'),
    html.Div(id='output-data-upload')
])

source_components = html.Div([
    dcc.Upload(id="upload-source",
               children=html.Div(["Загрузите данные или ",
                                  html.A("выберите файл", href=""),
                                  " в формате excel"]),
               style={
                   'width': '100%',
                   'height': '60px',
                   'lineHeight': '60px',
                   'borderWidth': '1px',
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   'margin': '10px'
               },
               multiple=False,
               accept='.xlsx'),
    html.Div(id='output-source-upload')
])

documentation = html.Div(
    [html.Div('Инструкция',
              style={'textAlign': 'center', 'fontSize': 20, 'font-weight': 'bold'}),
     html.Div(children='''Данный веб-интерфейс предназначен для прогнозирования количества заболеваний 
            гриппом и ОРВИ, а также составления графиков на основе моделирования.
            '''),
     html.Div(children='''Интерфейс состоит из трёх вкладок:'''),
     html.Li(children='Данные', style={'font-weight': 'bold'}),
     html.Div(children='''Раздел позволяет выбрать год для которого необходимо провести моделирование.
             Уровень детализированности позволяет выбрать режим моделирования.'''),
     html.Li(children='Параметры модели', style={'font-weight': 'bold'}),
     html.Div(children=''' Раздел представляет основное окно для просмотра и изменения параметров модели, 
             а также визуального анализа графика. Кроме этого, присутствуют кнопки: запустить калибровку, 
             запустить моделирование, сохранить график и сгенерировать бюллетень.'''),
     html.Ul([html.Li(children='Запустить калибровку'),
              html.Div(children=''' Нажатие кнопки инициирует запуск калибровки модели: для доступных
                     данных по заболеваемости вычисляются оптимальные параметры модели, при 
                     которых кривая графика моделирования будет приближать реальные данные. Во время работы
                     калибровки кнопка становится "неактивной", время калибровки составляет несколько минут. 
                     После окончания калибровки на компьютер скачается файл пресета. Этот файл можно использовать для
                     воссоздания графика во вкладке "пресеты".'''),
              html.Li(children='Запустить моделирование'),
              html.Div(children=''' Нажатие кнопки инициирует запуск моделирования и построение графика. Данную 
                     опцию необходимо использовать после изменения параметров модели, чтобы обновить график с использованием
                     актуальных параметров. '''),
              html.Li(children='Сохранить график'),
              html.Div(
         children=''' После нажатия график сохраняется на сервер для дальнейшего использования. '''),
         # html.Li(children='Сгенерировать бюллетень'),
         # html.Div(children='''Генерирует pdf-бюллетень с графиками и текстовым описанием.'''),
     ]),

     html.Li(children='Пресеты', style={'font-weight': 'bold'}),
     html.Div(children=''' В данном разделе можно загрузить файл пресета - специальный файл, содержащий 
             всю необходимую информацию для воссоздания графика по данным определенного периода. После 
             загрузки файла пресета необходимо перейти во вкладку "Параметры модели" 
             и нажать "запустить моделирование" '''),
     ]
)
'''
        ***OLD INTERFACE VERSION***
upper_row = \
    Row([
        Col([
            html.Div([
                dbc.Tabs([
                    dbc.Tab([model_components, buttons],
                            label='Параметры модели', id='model-components'),
                    dbc.Tab([data_components], label='Данные',
                            id='data-components'),
                    dbc.Tab([preset_components, source_components], label='Пресеты',
                            id='preset-components'),
                    dbc.Tab([documentation], label='Инструкция',
                            id='documentation'),
                ], style={'fontWeight': 'bold'})
            ], className='inputs-container shadow p-5 rounded',
                style={'margin': '0px 60px 50px 60px',
                       'padding': '30px 40px 30px 40px',
                       'backgroundColor': 'white'})
        ], md=11)
    ], justify='center')

lower_row = \
    Row([
        Col(dbc.Spinner([
            html.Div([
                dcc.Graph(id='model-fit', config=config, mathjax=True)
            ], className='graph-container shadow rounded',
                style={'backgroundColor': 'white',
                       'margin': '0px 60px 0px 60px',
                       'padding': '20px 20px'})
        ], size='lg', color="primary", type="border", fullscreen=True, ), md=11)
    ], justify='center')

layout = \
    html.Div([
        upper_row,
        lower_row,
    ], style={'backgroundColor': '#e6e6e6',
              'width': '100%', 'height': '100%',
              'margin': '0px', 'padding': '20px 0px'})
'''

advance_setting = html.Div([
    dbc.Button('Advance setting', size="sm", className='advance_setting', id='advance_setting', style={
               'position': 'relative', 'left': '35%', 'top': '20px', "marginBottom": "20px"}),

    dbc.Offcanvas(
        [
            get_model_advance_params()
        ],
        id="offcanvas",
        title="Advance setting",
        is_open=False,
        style={'width': '40%'}
    ),
])

upper_row = \
    html.Div([
        dbc.Tabs([
            dbc.Tab([model_components, data_components, advance_setting],
                    label='Параметры модели', id='model-components'),
            dbc.Tab([preset_components, source_components], label='Пресеты',
                    id='preset-components'),
            dbc.Tab([documentation], label='Инструкция',
                    id='documentation'),
        ], style={'fontWeight': 'bold'}),
    ],
        style={'padding': '0px 40px 30px 40px',
               'backgroundColor': 'white'})
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[2],
    y=[1],
    text=["Запустите моделирование, чтобы построить график"],
    mode="text",
    textfont=dict(
        family="Verdana",
        size=22,
        color="#808080"
    )
))
lower_row = \
    dbc.Spinner([
        html.Div([
                dcc.Graph(figure=fig, id='model-fit',
                          className='dash-graph-districts', config=config, mathjax=True)
                ], className='graph-container rounded',
            style={'backgroundColor': 'white',
                   'padding': '20px 20px',
                   'border': 'solid black 1px',
                   'marginRight': '10px',
                   'marginLeft': '10px'})
    ], size='lg', color="primary", type="border", fullscreen=True, )

new_r = Row([
    Col(upper_row, lg=5, width=12), Col(
        [Row(lower_row), Row(buttons)], lg=7, width=12)
], justify='center')

layout = \
    html.Div([
        new_r,
    ], style={'backgroundColor': 'white',
              'width': '100%', 'height': '100%',
              'margin': '0px', 'paddingTop': '20px'})
