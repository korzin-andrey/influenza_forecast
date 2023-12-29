import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_bootstrap_components import Row, Col

from components import multi_age

mode_bar_buttons_to_remove = ['autoScale2d',
                              'pan2d', 'zoom2d', 'select2d', 'lasso2d']
config = dict(displaylogo=False, responsive=True,
              modeBarButtonsToRemove=mode_bar_buttons_to_remove)

cities = [html.P('Город', style={'margin': '20px 20px'}),
          dcc.RadioItems(options=[{'label': 'Санкт-Петербург', 'value': 'spb'},
                                #   {'label': 'Москва', 'value': 'msc',
                                #       'disabled': True},
                                #   {'label': 'Новосибирск', 'value': 'novosib', 'disabled': True}
                                  ],
                         value='spb',
                         inputStyle={"marginRight": "10px",
                                     "marginLeft": "15px"},
                         style={'margin': '10px 5px'},
                         id='city')]

years = [html.P('Год', style={'margin': '20px 0px 10px 0px'}),
         dcc.Dropdown(options=['2010', '2011', '2012', '2013', '2014',
                               '2015', '2016', '2017', '2018', '2019'],
                      value='2010',
                      id='year',
                      clearable=False)]


def get_incidence_type(default):
    return [html.P('Уровень детализированности', style={'margin': '20px 0px 10px 0px'}),
            dcc.Dropdown(options=[{'label': 'возрастные группы', 'value': 'age-group'},
                                  {'label': 'штаммы', 'value': 'strain'},
                                  {'label': 'возрастные группы и штаммы',
                                      'value': 'strain_age-group'},
                                  {'label': 'агрегированные данные', 'value': 'total'}],
                         value=default, id='incidence', clearable=False)]


def get_data_components(incidence_type_init):
    return Row([
        Col([*cities], md=3),
        Col([*years, *get_incidence_type(incidence_type_init)], md=4)
    ], id='data_components', justify='left')


data_components = get_data_components('age-group')


def get_a_layout(value):
    return Row([
        Col([
            dcc.Slider(min=0, max=1, step=10e-7, marks={0: '0', 1: '1'},
                       id='a', tooltip={"placement": "bottom"})
        ]),
        Col([
            dbc.Input(id='a_io', type='number',
                      value=value, style={'width': '170px'})
        ], align='end')
    ])


def get_mu_layout(value):
    return Row([
        Col([
            dcc.Slider(min=0, max=1, step=0.001, marks={0: '0', 1: '1'}, id='mu',
                       tooltip={"placement": "bottom"}, value=value)
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


model_components = get_model_params_components(multi_age)

buttons = \
    Row([
        Col([
            html.Div([
                dbc.ButtonGroup([
                    html.Div([dbc.Button('Запустить калибровку',
                               id='calibration-button'),
                               dcc.Download(id="download-preset")],)
                    ,
                    # dbc.Button('Остановить калибровку', id='stop-button'),
                    dbc.Button('Запустить моделирование', id='forecast-button'),
                    dbc.Button('Сохранить график', id='save_plot-button'),
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
            ], style={'margin': '20px 60px 0px 0px'})
        ])
    ], justify='center')

preset_components = html.Div([
    dcc.Upload(id="upload-preset",
               children=html.Div(["Перетащите или ",
                                  html.A("выберите файл",
                                         href="javascript:void(0);"),
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

documentation = html.Div(
    [html.Div('Инструкция',
             style={'textAlign': 'center', 'fontSize': 20,'font-weight': 'bold'}),
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
            html.Div(children=''' После нажатия график сохраняется на сервер для дальнейшего использования. '''),
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

upper_row = \
    Row([
        Col([
            html.Div([
                dbc.Tabs([
                    dbc.Tab([data_components], label='Данные',
                            id='data-components'),
                    dbc.Tab([model_components, buttons],
                            label='Параметры модели', id='model-components'),
                    dbc.Tab([preset_components], label='Пресеты',
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
