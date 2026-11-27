# Project completion / reproducibility notes

## Notebook sequence

1. `notebooks/data_cleaning.ipynb` - source import, quality checks, weekly U.S. table.
2. `notebooks/exploratory_data_analysis.ipynb` - descriptive EDA.
3. `notebooks/wave_analysis.ipynb` - smoothing, peak/valley detection, wave exports.
4. `notebooks/growth_rate_analysis.ipynb` - per-wave growth proxy and linear trend.
5. `notebooks/deaths_per_case.ipynb` - exploratory death/case ratios and lag correlations.
6. `notebooks/logistic_model_fitting.ipynb` - fit one wave, chronological holdout, forecasts and diagnostics.

Run notebooks from the repository root or notebook folder. The analysis notebooks locate the project data using a root fallback. The wave exports already in the repository are included for reproducibility.

## Python functions

The modules in `Python/` are import-safe and have no plotting side effects on import.

- `continuous_logistic_model.py`: analytical solution, ODE simulation, fitting, metrics, plot helper.
- `discrete_logistic_model.py`: recurrence map and optional time-series/cobweb visualization.
- `forecasting.py`: fit and forecast a specified number of time steps.
- `parameter_estimation.py`: Gaussian-error maximum-likelihood demo and reusable fitter.
- `model_evaluation.py`: MAE, MSE, RMSE, MAPE percent, R-squared.

## Install and run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python Python/continuous_logistic_model.py
python Python/parameter_estimation.py
```

The MLE demo writes its parameter table under `results/`. The fitting notebook writes selected-wave metrics, parameters and forecast tables there too.

## Interpretation

This is a mathematical modeling project, not an epidemiological forecasting service. Reported COVID counts are affected by changing test availability, reporting definitions, surveillance, behavior, vaccination, variants and overlapping waves. Logistic fits should be applied to a single wave and interpreted cautiously. Forecasts are conditional extrapolations, not forecasts of future policy or disease dynamics. The death-to-case ratio is a crude aggregate descriptive measure, not a cohort case-fatality risk.
