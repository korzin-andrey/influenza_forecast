from app import app
from dash import Input, Output, State, callback, ALL
import datetime
import calibration
import json
import os
from optimizers import multiple_model_fit
from dash.exceptions import PreventUpdate
import signal

@app.callback(
    Input('calibration-button', 'n_clicks'),

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
    # calibration_parameters = {'exposed': [0.30148313273350635],
    # 'incidence': 'total',
    # 'lambda': [0.09544288428775363],
    # 'a': [0.13599275369791994],
    # 'delta': 57,
    # 'total_recovered': [483795.443701321],
    # 'R2': [0.5166538711970509],
    # 'r0': [[3.7383068917701014]],
    # 'mu': 0.1}
    # print("Something")

    # exposed_def = exposed_dict_to_inputs(
    #     calibration_parameters['exposed'], incidence)
    # lambda_def = lambda_dict_to_inputs(calibration_parameters['lambda'], incidence)

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
    return 0
