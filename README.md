# Start of the work

To start the work, please, create and a separate virtual environment by running the following strings 
of code (Windows version):

[venv guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)

1) ```py -m venv <envname>```
2) ```.\env\Scripts\activate```

To install all necessary dependencies, please run the command below from the directory 
*/RSCF_Uncertainty/BaroyanAgeMultistrain_v2:

```python -m pip install -e . ```


# Repository structure

A structure of the repository is the following:
* ```data```  keeps original incidence data for Saint Petersburg and modules for initial 
data preparation; 
* ```models``` contains a module with Baroyan-Rvachev model which is general for all incidence types and allows to
handle different number of age groups and strains;  
* ```optimizers``` consists of the modules involved in calibration process. ```simulated_annealing.py```
is a module with the class ```InitValueFinder``` to run optimization process. *energy()* function is used to process
parameter values according to the incidence type.

* ```base_optimizer.py``` contains a parent class with the main *find_model_fit()*
and *fit_one_outbreak()* methods. It is inherited by classes in ```optimizer_age.py```, 
```optimizer_strain.py```, ```optimizer_strain_age.py```, and ```optimizer_total.py```;

* ```bootstrapping``` maintains modules for bootstrapping procedure execution, module for different 
error structure generation, and bootstrapped curves restoration from the calculated parameters;

* ```visualization``` contains a corresponding module for visualization of the calibration and prediction  
results as well as epidemic indicators such as population immunity and Rt;

* ```output``` saves all the results of the experiments, both data and figures in one unified folder that is generated
for each experiment; 

* ```config.yaml``` lets user define input parameters such as *INCIDENCE_TYPE*, *DATA_DETAIL*, 
*MODEL_DETAIL*, *PREDICT*, etc.;

* ```main.py``` is an entry point(script) of the repository to run an experiment;

* ```calculate_bootstrap_parameters.py``` is a module to perform bootstrap procedure;

* ```experiment_setup``` is a factory class that returns an appropriate model and optimizer objects 
to start experiment;

* ```utils.py``` is a module with auxiliary functions to save results and restore them from either  saved 
data points or parameters;

* ```gui``` contains a corresponding modules for dashboard app and manual calibration.

* ```bulletin``` is folder with python script to crete pdf-bulletin containing epidemiology situation. To successfull run of this script latex package needed on your PC. 


# Multiprocessing
**NOTE:** python.multiprocessing and pathos.multiprocessing may work incorrectly with .ipynb files, use [nbmultitask](https://github.com/micahscopes/nbmultitask) in that case

# calculate_bootstrap_parameters.py

* set up simulation parameters on *config.yaml*
* set up **num_iter_fit** value in file calibration section
* set up **num_iter** value in file bootstrap section
* run the program

