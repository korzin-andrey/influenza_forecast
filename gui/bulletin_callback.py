from app import app
from dash import Input, Output, State, callback, ALL
from aux_functions import prepare_exposed_list, get_data_and_model, transform_days_to_weeks, cities
import jsonpickle
import json

@app.callback(Input("ci-button", "n_clicks"),
              Output("download-ci-request-json", "data"),
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
              force_no_output=True, prevent_initial_call=True)
def bulletin_client_call(_, incidence, exposed_values,
                         lambda_values, a, mu, delta, sample_size, city, year, forecast_term, inflation_parameter):
    exposed_list = exposed_values
    lam_list = lambda_values
    a_list = [a]
    year = int(year)
    exposed_list = prepare_exposed_list(incidence, exposed_list)

    epid_data, model_obj, groups = get_data_and_model(mu, incidence, year)

    # if sample_size < len(epid_data.index):
    #     epid_data = epid_data[:sample_size]

    model_obj.init_simul_params(
        exposed_list=exposed_list, lam_list=lam_list, a=a_list)
    model_obj.set_attributes()
    simul_data, _, _, _, _ = model_obj.make_simulation()
    simul_weekly = transform_days_to_weeks(simul_data, groups)

    current_time = str(datetime.datetime.now())

    forecasting = False
    if sample_size <= len(epid_data.index):
        forecasting = True

    bulletin_request = {
        "datetime": current_time,
        "simulation_parameters": {
            "city": city,
            "city_russian": cities[city],
            "year": int(year),
            "incidence": incidence,
            "exposed": exposed_values,
            "lambda": lambda_values,
            "a": a,
            "mu": mu,
            "delta": delta,
            "sample_size": sample_size,
            "forecasting": forecasting,
            "forecast_term": forecast_term,
            "inflation_parameter": inflation_parameter
        },
        "groups": groups,
        "epid_data_pickled": jsonpickle.encode(epid_data.to_csv(index=False)),
        "model_obj_pickled": jsonpickle.encode(model_obj),
        "simul_weekly_pickled": jsonpickle.encode(simul_weekly.to_csv(index=False))
    }

    simulation_type_string = ""  # retrospective or forecast
    if forecasting:
        simulation_type_string = "_forecast"
    else:
        simulation_type_string = "_retrospective"

    # bulletin_generator.generate_bulletin()
    return dict(content=json.dumps(bulletin_request),
                filename=f"request_for_bulletin_{current_time.replace(' ', '_')}{simulation_type_string}.json")
