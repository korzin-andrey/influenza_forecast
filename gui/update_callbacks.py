from app import app
from dash import Input, Output, State, callback, ALL, ctx
from components import multi_strain, multi_age, multi_strain_age, total_c
from dash.exceptions import PreventUpdate
from aux_functions import get_data_size
UPDATE_MODE = True


@app.callback(
    Output('sample', 'value', allow_duplicate=True),
    Input('year', 'value'),
    State('incidence', 'value'),
    prevent_initial_call=True
)
def update_year(year, incidence):
    return get_data_size(incidence, int(year))

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
    from preset_callback import PRESET_MODE
    global UPDATE_MODE
    if PRESET_MODE==UPDATE_MODE:
        UPDATE_MODE = not(UPDATE_MODE)
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
     Output({'type': 'exposed_io', 'index': ALL}, 'value'),
     Output({'type': 'exposed', 'index': ALL}, 'min'),
     Output({'type': 'exposed', 'index': ALL}, 'max'),
     Output({'type': 'exposed', 'index': ALL}, 'marks')],

    [Input({'type': 'exposed_io', 'index': ALL}, 'value'),
     Input({'type': 'exposed', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def update_exposed(exposed_io, exposed):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        exposed = exposed_io
    else:
        exposed_io = exposed

    min_ = [round(i-0.0005,4) for i in exposed]
    max_ = [round(i+0.0005,4) for i in exposed]
    marks = [{min_[i]: f'{min_[i]}', max_[i]: f'{max_[i]}'} for i in range(len(min_))]
    return exposed, exposed_io, min_, max_, marks




@app.callback(
    [Output({'type': 'lambda', 'index': ALL}, 'value'),
     Output({'type': 'lambda_io', 'index': ALL}, 'value'),
     Output({'type': 'lambda', 'index': ALL}, 'min'),
     Output({'type': 'lambda', 'index': ALL}, 'max'),
     Output({'type': 'lambda', 'index': ALL}, 'marks')],

    [Input({'type': 'lambda_io', 'index': ALL}, 'value'),
     Input({'type': 'lambda', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def update_lambda(lambda_io, lambda_):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        lambda_ = lambda_io
    else:
        lambda_io = lambda_

    min_ = [round(i-0.00005,5) for i in lambda_]
    max_ = [round(i+0.00005,5) for i in lambda_]
    marks = [{min_[i]: f'{min_[i]}', max_[i]: f'{max_[i]}'} for i in range(len(min_))]
    return lambda_, lambda_io, min_, max_, marks



@app.callback(
    [Output('a', 'value'),
     Output('a_io', 'value'),
     Output('a', 'min'),
     Output('a', 'max'),
     Output('a', 'marks')],

    [Input('a_io', 'value'),
     Input('a', 'value')],
    prevent_initial_call=True
)
def update_a(a_io, a):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        a = a_io
    else:
        a_io = a

    min_ = round(a-0.0005,5)
    max_ = round(a+0.0005,5)
    marks = {min_: f'{min_}', max_: f'{max_}'}
    return a_io, a, min_, max_, marks




@app.callback(
    [Output('mu', 'value'),
     Output('mu_io', 'value'),
     Output('mu', 'min'),
     Output('mu', 'max'),
     Output('mu', 'marks')],

    [Input('mu_io', 'value'),
     Input('mu', 'value')],
    prevent_initial_call=True
)
def update_mu(mu_io, mu):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        mu = mu_io
    else:
        mu_io = mu

    min_ = round(mu-0.0005,5)
    max_ = round(mu+0.0005,5)
    marks = {min_: f'{min_}', max_: f'{max_}'}
    return mu_io, mu, min_, max_, marks




@app.callback(
    [Output('delta', 'value'),
     Output('delta_io', 'value')],

    [Input('delta_io', 'value'),
     Input('delta', 'value')],
    prevent_initial_call=True
)
def update_delta(delta_io, delta):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        delta = delta_io
    else:
        delta_io = delta

    return delta_io, delta




@app.callback(
    [Output('sample', 'value'),
     Output('sample_io', 'value')],

    [Input('sample_io', 'value'),
     Input('sample', 'value')],
    prevent_initial_call=True
)
def update_sample(sample_io, sample):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        sample = sample_io
    else:
        sample_io = sample

    return sample_io, sample




@app.callback(
    [Output('forecast-term', 'value'),
     Output('forecast-term_io', 'value')],

    [Input('forecast-term_io', 'value'),
     Input('forecast-term', 'value')],
    prevent_initial_call=True
)
def update_forecast_term(forecast_term_io, forecast_term):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        forecast_term = forecast_term_io
    else:
        forecast_term_io = forecast_term

    return forecast_term_io, forecast_term




@app.callback(
    [Output('inflation-parameter', 'value'),
     Output('inflation-parameter_io', 'value')],

    [Input('inflation-parameter_io', 'value'),
     Input('inflation-parameter', 'value')],
    prevent_initial_call=True
)
def update_inflation_parameter(inflation_parameter_io, inflation_parameter):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if "_io" in trigger_id:
        inflation_parameter = inflation_parameter_io
    else:
        inflation_parameter_io = inflation_parameter

    return inflation_parameter_io, inflation_parameter




@app.callback(
    Output("offcanvas", "is_open"),
    Input("advance_setting", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open
