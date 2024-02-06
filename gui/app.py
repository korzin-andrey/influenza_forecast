import dash_bootstrap_components as dbc
from dash_extensions.enrich import CycleBreakerTransform
from layout import layout
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager
from dash_extensions.enrich import DashProxy

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)


app = DashProxy(__name__,
                transforms=[CycleBreakerTransform()],
                external_stylesheets=[dbc.themes.MATERIA],
                prevent_initial_callbacks="initial_duplicate",
                long_callback_manager=long_callback_manager,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app._favicon = ("favicon.ico")

app.layout = layout

