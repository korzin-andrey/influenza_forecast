from setuptools import setup, find_packages

setup(name='uq_influenza_modeling',
      version='1.0.0',
      description='A project represents a framework for influenza modeling '
                  'using Baroyan-Rvachev model with simulated annealing and '
                  'L-BFGS-B optimization techniques. Uncertainty quantification '
                  'is also enabled.',
      packages=find_packages(),
      install_requires=['dash',
                        'dash_bootstrap_components',
                        'dash_extensions',
                        'diskcache'
                        'kaleido',
                        'matplotlib',
                        'numpy',
                        'pylatex',
                        'pandas',
                        'PyYAML',
                        'scikit-learn',
                        'scipy',
                        'simanneal>=0.5.0',
                        'statsmodels',
                        'tqdm',
                        'pathos',
                        'plotly',
                        'setuptools',
                        'dill',
                        'pox',
                        'ppft',
                        'multiprocess',
                        'nbmultitask',
                        'jupyter',
                        'joblib',
                        'jsonpickle',
                        'wheel'])
