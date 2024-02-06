from app import app
from dash import Input, Output, State, callback
from components import multi_strain, multi_age, multi_strain_age, total_c
import json
from aux_functions import exposed_dict_to_inputs, lambda_dict_to_inputs
import components.age_groups_components as age_groups_comps
import components.strains_age_components as strains_age_comps
import components.strains_components as strains_comps
import components.total_components as total_comps
from layout import get_model_params_components, get_data_components, get_model_advance_params
import base64

PRESET_MODE = False

@app.callback([Output('data_components', 'children', allow_duplicate=True),
               Output('city', 'value', allow_duplicate=True),
               Output('year', 'value', allow_duplicate=True),
               Output('params-components', 'children'),
               Output('params-components-advance', 'children'),
               Output('upload-preset', 'contents')],
              Input('upload-preset', 'contents'),
              State('upload-preset', 'filename'),
              State('upload-preset', 'last_modified'),
              prevent_initial_call=True
              )
def process_preset(list_of_contents, list_of_names, list_of_dates):
    incidence_default = "total"
    city_default = 'spb'
    year_default = '2023'

    component_bunch = age_groups_comps.get_multi_age_c()

    a_default = 0.01093982936993367
    mu_default = 0.2
    delta_default = 30

    if list_of_contents is not None:
        global PRESET_MODE
        from update_callbacks import UPDATE_MODE
        PRESET_MODE = UPDATE_MODE
        preset = json.loads(base64.b64decode(list_of_contents[29:]))
        print(preset)
        incidence_default = preset['incidence']
        city_default = preset["city"]
        year_default = preset["year"]

        a_default = preset["a"]
        mu_default = preset["mu"]
        delta_default = preset["delta"]

        exposed_def = exposed_dict_to_inputs(
            preset['exposed'], incidence_default)
        lambda_def = lambda_dict_to_inputs(preset['lambda'], incidence_default)

        if incidence_default == 'total':
            component_bunch = total_comps.get_total_c(exposed_def, lambda_def)
        elif incidence_default == 'strain':
            component_bunch = strains_comps.get_multi_strain_c(
                exposed_def, lambda_def)
        elif incidence_default == 'age-group':
            component_bunch = age_groups_comps.get_multi_age_c(
                exposed_def, lambda_def)
        elif incidence_default == 'strain_age-group':
            component_bunch = strains_age_comps.get_multi_strain_age_c(
                exposed_def, lambda_def)
        else:
            raise ValueError(f"can't parse incidence: {incidence_default}")

    return (get_data_components(incidence_default).children,
            city_default, 
            year_default,
            get_model_params_components(component_bunch, a_default, mu_default).children,
            get_model_advance_params(delta_default).children, None)




####### было в app, но нигде не используется, решил оставить на всякий #######
def parse_contents(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded))
    return df
##############################################################################
