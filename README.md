# Modeling and Forecasting COVID-19 Dynamics in the United States Using Logistic Growth Models

## Project Overview

This project investigates the mathematical modeling and forecasting of COVID-19 dynamics in the United States using two classical population growth formulations: the **discrete logistic growth map** and the **continuous logistic growth model**.

The project explores the mathematical foundations of logistic growth, derives both model formulations, examines their equilibrium solutions and stability properties, and implements numerical simulations using Python and MATLAB.

The continuous logistic model is subsequently applied to historical U.S. COVID-19 cumulative reported case data to estimate model parameters and generate short-term forecasts.

The project emphasizes mathematical derivation, numerical implementation, parameter estimation, and model evaluation at an advanced undergraduate level.

---

## Research Question

Can classical discrete and continuous logistic growth models describe the progression of cumulative reported COVID-19 cases in the United States, and how effectively can the continuous logistic model forecast short-term changes in reported cases?

## Research Objectives

The primary objectives are:

1. Derive the discrete logistic growth recurrence relation.
2. Derive the continuous logistic growth differential equation and its analytical solution.
3. Analyze equilibrium points and stability conditions for both models.
4. Investigate how model parameters influence population growth behavior.
5. Implement numerical simulations using Python and MATLAB.
6. Estimate logistic model parameters from historical U.S. COVID-19 data.
7. Generate 7-day, 14-day, and 30-day forecasts.
8. Evaluate forecasting performance using statistical error metrics.
9. Compare the behavior and limitations of discrete and continuous logistic growth models.

---

# Mathematical Background

## 1. Discrete Logistic Growth Model

The discrete logistic growth model describes population changes over successive time steps.

The recurrence relation is:

$$
P_{n+1}=P_n+rP_n\left(1-\frac{P_n}{K}\right)
$$

Where:

- $P_n$ represents the population at time step $n$.
- $P_{n+1}$ represents the population at the next time step.
- $r$ represents the discrete growth parameter.
- $K$ represents the carrying capacity.

### Mathematical Analysis

The equilibrium points are obtained by setting:

$$
P_{n+1}=P_n=P^*
$$

This produces:

$$
P^*=0
$$

and

$$
P^*=K
$$

The stability of these equilibrium points will be examined using the derivative of the discrete mapping function.

For positive $r$, the carrying-capacity equilibrium is locally asymptotically stable when:

$$
0<r<2
$$

The project will investigate how different parameter values influence the behavior of the discrete logistic map.

## 2. Continuous Logistic Growth Model

The continuous logistic growth model describes population changes over continuous time.

Its governing differential equation is:

$$
\frac{dP}{dt}=rP\left(1-\frac{P}{K}\right)
$$

Where:

- $P(t)$ represents the population at time $t$.
- $r$ represents the continuous growth rate.
- $K$ represents the carrying capacity.

### Analytical Solution

The differential equation can be solved using separation of variables and partial fraction decomposition.

The resulting analytical solution is:

$$
P(t)=\frac{K}{1+Ae^{-rt}}
$$

Where:

$$
A=\frac{K-P_0}{P_0}
$$

and $P_0$ represents the initial population.

### Equilibrium Analysis

The equilibrium solutions satisfy:

$$
\frac{dP}{dt}=0
$$

Therefore:

$$
P^*=0 
$$

and 

$$
P^*=K
$$

For positive growth rates:

- $P^*=0$ is unstable.
- $P^*=K$ is asymptotically stable.

### Inflection Point

The logistic growth curve reaches its maximum growth rate at:

$$
P(t)=\frac{K}{2}
$$

The corresponding inflection time is:

$$
t_{\text{inflection}}=\frac{\ln(A)}{r}
$$

This represents the point at which the modeled population transitions from accelerating growth to decelerating growth.

---

# Discrete vs. Continuous Logistic Growth

An important component of this project is comparing the mathematical behavior of the two logistic formulations.

| Characteristic | Discrete Logistic Model | Continuous Logistic Model |
|---|---|---|
| Mathematical formulation | Difference equation | Differential equation |
| Time representation | Discrete | Continuous |
| Solution method | Iteration | Analytical or numerical |
| Equilibrium analysis | Fixed points | Equilibrium solutions |
| Stability analysis | Discrete dynamical systems | Continuous dynamical systems |
| Numerical implementation | Recurrence iteration | Numerical integration |
| Main application | Mathematical comparison | COVID-19 forecasting |

The discrete recurrence relation can also be interpreted as an explicit Euler discretization of the continuous logistic differential equation with a time step of one.

The two formulations are related but are not generally equivalent for finite time steps.

---

# COVID-19 Data and Forecasting

## Dataset

The project will use historical COVID-19 data for the United States.

The primary proposed data source is:

**Our World in Data: COVID-19 Dataset**

https://ourworldindata.org/coronavirus

The dataset contains reported epidemiological observations, including:

- Date
- Location
- Total confirmed cases
- New confirmed cases
- Smoothed daily cases
- Total deaths
- Vaccination statistics

The primary response variable for this project is:

$$
N(t)=\text{Cumulative reported COVID-19 cases}
$$

The analysis will focus on cumulative reported cases rather than active infections or the total number of actual infections.

## Forecasting Methodology

The continuous logistic model will be fitted to selected historical epidemic-wave data.

The model parameters are:

$$
N(t)=\frac{K}{1+Ae^{-rt}}
$$

The parameters will be estimated using nonlinear least squares.

The estimated parameters are:

- $K$: Fitted upper asymptote of the selected epidemic wave.
- $r$: Estimated continuous growth rate.
- $A$: Logistic integration parameter.

The fitted model will be used to generate forecasts over three time horizons:

| Forecast Horizon | Objective |
|---|---|
| 7 days | Short-term forecasting |
| 14 days | Intermediate short-term forecasting |
| 30 days | Longer short-term forecasting |

The forecasts will be compared against subsequently observed cumulative reported cases.

**Important:** The fitted carrying capacity is an upper asymptote for the selected model and epidemic-wave interval. It should not be interpreted as the maximum number of people who can contract COVID-19.

## Parameter Estimation

The project will investigate nonlinear least-squares estimation using Python's `scipy.optimize.curve_fit`.

The objective is to minimize:

$$
SSE=\sum_{i=1}^{n}\left[N_i-\hat{N}(t_i)\right]^2
$$

Where:

- $N_i$ represents the observed cumulative case count.
- $\hat{N}(t_i)$ represents the model-predicted case count.
- $n$ represents the number of observations.

The estimated parameters will be used to evaluate the model's ability to represent historical epidemic-wave behavior.

---

# Model Evaluation

Forecasting performance will be assessed using the following statistical metrics.

### Mean Absolute Error (MAE)

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

### Root Mean Squared Error (RMSE)

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

### Coefficient of Determination ($R^2$)

$$
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

These metrics will be used to evaluate how closely the model predictions agree with observed cumulative reported cases.

Forecast evaluation will use chronological training and testing periods to avoid data leakage.

---

# Technology Stack

### Programming Languages

- Python
- MATLAB

### Python Libraries

| Library | Purpose |
|---|---|
| NumPy | Numerical calculations and array operations |
| SciPy | Numerical integration and parameter estimation |
| Matplotlib | Data visualization |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | Model evaluation metrics |
| Jupyter Notebook | Exploratory analysis and experimentation |

### MATLAB

MATLAB will be used for independent numerical implementation, simulation, and comparison of model results.

---

# Repository Structure

```text
cpsc-375-project/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt                   # Python packages for the venv / Colab
│
├── data/
│   ├── total_cases.csv                # Raw OWID cumulative cases (wide format)
│   ├── total_deaths.csv               # Raw OWID cumulative deaths (wide format)
│   └── processed/
│       ├── tidy_covid_all_locations.csv   # Tidy weekly data, all locations
│       ├── us_covid_clean.csv             # Cleaned weekly U.S. table
│       └── us_covid_waves.csv             # U.S. table with wave labels (planned, Wk 2)
│
├── notebooks/
│   ├── data_cleaning.ipynb            # Data import, cleaning, tidying (Q1–Q2)
│   ├── exploratory_data_analysis.ipynb    # Initial exploration and single logistic fit
│   ├── wave_analysis.ipynb            # Q3: wave definitions (planned, Wk 2)
│   ├── deaths_per_case.ipynb          # Q5: deaths per case by wave (planned, Wk 2)
│   ├── growth_rate_analysis.ipynb     # Q4: growth-rate regressions (planned, Wk 3)
│   └── logistic_model_fitting.ipynb   # Logistic fits by wave (planned, Wk 2–3)
│
├── Python/
│   ├── continuous_logistic_model.py   # Continuous logistic model
│   ├── discrete_logistic_model.py     # Discrete logistic map and cobweb plots
│   ├── parameter_estimation.py        # Parameter estimation test on synthetic data
│   ├── model_evaluation.py            # Naive baseline and MAE/RMSE/R² (planned, Wk 3)
│   └── forecasting.py                 # 1/2/4-week forecasts (planned, Wk 4)
│
├── figures/                           # (planned, Wk 1–4)
│   ├── fig1_us_cases_deaths.png
│   ├── fig2_wave_cumulative.png
│   ├── fig3_growth_rate.png
│   ├── fig4_deaths_per_case.png
│   ├── fig5_logistic_fits.png
│   └── fig6_forecasts.png
│
├── results/
│   ├── logistic_mle_estimates.csv     # Synthetic-data parameter estimates
│   ├── table2_wave_summary.csv        # (planned, Wk 2)
│   ├── table3_growth_regression.csv   # (planned, Wk 3)
│   ├── estimated_parameters.csv       # (planned, Wk 3)
│   ├── forecast_predictions.csv       # (planned, Wk 4)
│   └── model_metrics.csv              # Table 4 (planned, Wk 4)
│
└── reports/
    ├── checkin_report.md
    ├── checkin_report.pdf
    ├── final_report.pdf               # (planned, Wk 4–5)
    └── final_presentation.pdf         # (planned, Wk 5)
```

---

# Methodology

The project will follow these stages:

### Stage 1: Mathematical Derivation

- Derive the discrete logistic recurrence relation.
- Derive the continuous logistic differential equation.
- Obtain the analytical solution of the continuous model.
- Determine equilibrium points and stability conditions.
- Derive the inflection point and maximum growth rate.

### Stage 2: Numerical Simulation

- Implement the discrete logistic map in Python and MATLAB.
- Implement the continuous logistic differential equation.
- Compare numerical and analytical solutions.
- Investigate the effects of varying growth rates and carrying capacities.

### Stage 3: Data Collection and Preprocessing

- Obtain historical U.S. COVID-19 data.
- Extract cumulative reported case observations.
- Handle missing values and reporting anomalies.
- Select appropriate epidemic-wave intervals.
- Visualize cumulative and daily reported cases.

### Stage 4: Parameter Estimation

- Fit the continuous logistic model to historical observations.
- Estimate $K$, $r$, and $A$.
- Examine the fitted curve and residuals.
- Investigate the sensitivity of parameter estimates to the selected training interval.

### Stage 5: Forecasting

Generate 7-day, 14-day, and 30-day forecasts using the fitted model.

### Stage 6: Model Evaluation

- Compare predictions with actual observations.
- Calculate MAE, RMSE, and $R^2$.
- Evaluate different forecasting horizons.
- Compare Python and MATLAB implementations.

### Stage 7: Research Discussion

Discuss the mathematical behavior, forecasting performance, assumptions, and limitations of the logistic growth models.

---

# Limitations

The classical logistic growth model has several limitations when applied to real-world epidemic data.

1. **Multiple epidemic waves:** COVID-19 experienced multiple waves of transmission, which cannot generally be represented accurately by a single logistic curve.

2. **Changing transmission conditions:** Variants, vaccination, public-health interventions, and behavioral changes influence epidemic progression.

3. **Reporting limitations:** Confirmed case counts depend on testing availability, reporting practices, and surveillance systems.

4. **Parameter sensitivity:** Estimated parameters can vary depending on the selected observation period and initial parameter estimates.

5. **Forecasting uncertainty:** A strong historical fit does not guarantee accurate future predictions.

6. **Carrying capacity interpretation:** The estimated carrying capacity is a fitted mathematical asymptote, not a biological limit on the total number of infections.

These limitations will be considered when interpreting the forecasting results.

---

# Installation and Setup

Clone the repository:

```bash
git clone https://github.com/USERNAME/COVID19-Logistic-Growth-Project.git
```

Navigate into the project directory:

```bash
cd COVID19-Logistic-Growth-Project
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

MATLAB scripts can be executed independently using MATLAB.

## Requirements

The `requirements.txt` file will include:

```text
numpy
scipy
matplotlib
pandas
scikit-learn
jupyter
```

---

# Expected Outcomes

The expected outcomes of this project include:

- Mathematical derivations of discrete and continuous logistic growth models.
- Equilibrium and stability analysis.
- Numerical simulations in Python and MATLAB.
- Comparison of discrete and continuous model behavior.
- Estimated logistic parameters from historical U.S. COVID-19 data.
- Weekly forecasts over 7-, 14-, and 30-week horizons (as used by the weekly cleaned dataset).
- Quantitative evaluation of forecasting performance.
- Discussion of the strengths and limitations of classical logistic growth models.

The project aims to demonstrate how classical mathematical models can be implemented and evaluated using real-world epidemiological observations.

---

# References and Data Sources

1. Verhulst, P. F., "Notice sur la loi que la population suit dans son accroissement," *Correspondance Mathématique et Physique*, vol. 10, pp. 113–121, 1838.

2. Our World in Data, "Coronavirus Pandemic (COVID-19)," https://ourworldindata.org/coronavirus.

3. SciPy Documentation, "scipy.integrate.solve_ivp," https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.

4. SciPy Documentation, "scipy.optimize.curve_fit," https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html.

5. MathWorks Documentation, "Ordinary Differential Equations," https://www.mathworks.com/help/matlab/ordinary-differential-equations.html.

Additional peer-reviewed research papers and relevant preprints will be incorporated as the literature review develops.

---

# Project Status

**Status:** Analysis and implementation substantially complete (reproducibility review pending).

The mathematical models, data cleaning, exploratory analysis, wave segmentation, growth-rate and death/case analyses, logistic fitting, forecasting utilities, and model evaluation are implemented. The notebooks and automated Python smoke tests are provided. Results should be rerun and reviewed before treating this project as a final research report.

---

# Future Improvements

Potential extensions include:

- Comparing different epidemic-wave intervals.
- Investigating the effects of different training-window lengths.
- Examining forecasting performance across different time horizons.
- Comparing numerical implementations between Python and MATLAB.
- Investigating alternative epidemic growth models as future research.

---

## Author

**Author:** Troy Chin

**Project:** Modeling and Forecasting COVID-19 Dynamics in the United States Using Logistic Growth Models

**Academic Level:** Advanced Undergraduate Research

**Primary Tools:** Python, MATLAB, Jupyter Notebook

**Research Focus:** Applied Mathematics, Mathematical Modeling, Numerical Analysis, and Time-Series Forecasting.


## Reproducible analysis and completed components

See [completion notes](README-completion-notes.md) for notebook order, setup, module APIs, and interpretation guidance. The notebook sequence now includes growth-rate analysis, deaths-to-case ratio exploration, and single-wave logistic fitting with chronological holdout and horizon forecasts.

Run the Python smoke tests after installing dependencies:

```bash
pytest -q
```

Forecast horizons in the fitting notebook are measured in **weeks**, matching the weekly cleaned dataset. The model's carrying capacity is an estimated asymptote, not a population-level infection limit. Multi-wave cumulative data should not be fit with one logistic curve.
