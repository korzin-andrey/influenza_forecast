from app import app
from dash import Input, Output, State, callback
import time
from dash.exceptions import PreventUpdate

@app.callback(
    Input('excel-button', 'n_clicks'),

    Output('model-fit', 'figure', allow_duplicate=True),

    prevent_initial_call=True,
)
def excel_create(_):

    from plot_callbacks import _GENERATE
    _GENERATE.generate()

    time.sleep(3)

    if True:
        raise PreventUpdate
    return 0
