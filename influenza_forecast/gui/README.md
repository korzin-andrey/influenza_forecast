This is a GUI for manual calibration of the model over the given data.

* Requirements

To set  up all necessary libraries, please, follow guidelines mention in ```README.md``` in the main directory 
```BaroyanAgeMultistrain_v2```. It is a good practice to initialize new virtual environment specific for the project and 
install all dependencies there. 

* Start of the app

GUI may be run directly in IDE(Pycharm tested) using file ```app.py``` or through the console using the command: 
```python gui/app.py```.
Dash will start the app on the following default address  http://127.0.0.1:8050/ in debug mode. It is dedicated for
developers of the app to track all errors that might occur. To switch off the mode change 
```app.run(debug=True)``` to ```app.run(debug=False)``` in the file ```app.py```.

* Structure of the GUI

All components are grouped in the file ```layout.py```. However, there are some dynamic components 
that are changed in the layout of the app. They are components
with ids *exposed-accordion-item* and *lambda-accordion-item* which are dependent from the one 
with id *incidence*. Therefore, a variable number of components are stored in a separate folder ```components```.

In the ```app.py```, there are only callbacks that enable functionality of the widgets. If new component is 
added, a new callback should be written in the file to enable its interactivity. 

* Uploading presets

You can upload preset in .json format *(component-id: output-data-upload)* which contains parameters derived 
from automatic calibration. The preset is generated in ```calculate_bootstra_parameters.py``` script 
if you set flag ```SAVE_PRESET``` to true in ```config.yaml```.

* Manual fit and predict

Once you've chosen the corresponding model parameters (through preset or sliders) you can update graph
with manual fitting curve (*fit-button*) or forecast (*forecast-button*).

* Bulletin and interval estimates

The call to *ci-button* allows you to download the current state of the GUI app for further postprocessing.
The ```request_for_bulletin_YYYY-MM-DD_HH_MM_SS.MS.json``` file contains:
- simulation parameters
- epidemic data (raw json-pickled CSV)
- model object (json-pickled object)
- simulation data (raw json-pickled CSV)

To compute interval estimates via the bootstrap procedure you should use the ```calculate_bootstrap_from_bulletin_request.py```
to process this file.