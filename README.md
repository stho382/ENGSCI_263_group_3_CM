# ENGSCI 263  - Rotarua Geothermal System Recovery

## Project Overview
For this project, our aim was to develop a computer model to analyse the pressure and temperature changes in the Rotorua geothermal field, providing insights as to the recovery of the Waikite geyser's surface features.

## Included Files
***analytic_solution.py***
  * Helper file for Benchmarking.py

***Benchmarking.py***
  * This file consists of code we used to generate plots for our benchmarks

***main.py***
  * Run this file to retrive all the plots made within our project

***ODE_Model_Function.py***
  * This file includes code we used to develop our forecasts for both pressure & temperature and also generate the misfit

***poetry.lock & pyproject.toml***
  * These files contains details of the dependancies and their version numbers required to execute our code

***test_functions.py***
  * This file includes the unit tests we have developed to ensure that our model is working as intended

***visualisations.py***
  * This file contains the code for the plot we used to visualise all the data provided to us

## Executing files

#### Viewing Plots
To execute and view the plots that we have made, please run the following snippets of code in the order provided:

1. Install poetry package
```bash
pip3 install poetry
```
2. Create a virtual environment with all the packages installed
```bash
poetry install
```
3. Run the `main.py` file in the virtual environment
```bash
poetry run python3 main.py
```

#### Executing tests
To execute and view out test functions in `test_functions.py`:

1. Follow steps 1 and 2 of 'Viewing plots'
2. Run `pytest` in the virtual environment
```bash
poetry run pytest
```