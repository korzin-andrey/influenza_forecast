from app import app
from dash import Input, Output, State, callback
import base64
import pandas as pd
from data import excel_preprocessing
import io

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-source', 'contents'),
              State('upload-preset', 'filename'),
              State('upload-preset', 'last_modified'),
              prevent_initial_call=True,
              )
def process_upload_data(contents, list_of_names, list_of_dates):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded), skiprows=[0])
    excel_preprocessing.preprocess_excel_source(df)
    children = ""
    print(1)
    return children
