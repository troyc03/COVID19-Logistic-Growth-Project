"""Smoke tests for the project's core model functions."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Python"))
from continuous_logistic_model import logistic_curve,simulate_logistic_growth,fit_logistic_growth
from discrete_logistic_model import discrete_logistic_growth
from model_evaluation import evaluate_predictions
from forecasting import Forecasting

def test_analytical_matches_numerical_solution():
    t=np.linspace(0,15,80); expected=logistic_curve(t,400,12,.25)
    numerical=simulate_logistic_growth(12,.25,400,t)
    assert np.allclose(expected,numerical,rtol=1e-5,atol=1e-6)

def test_discrete_map_converges_for_small_rate():
    p=discrete_logistic_growth(8,.6,250,100)
    assert p[0]==8 and np.all(np.isfinite(p))
    assert abs(p[-1]-250)<1e-3

def test_fit_recovers_parameters_on_clean_curve():
    t=np.arange(30,dtype=float); y=logistic_curve(t,800,10,.22)
    fitted,_=fit_logistic_growth(t,y)
    assert np.allclose(fitted,[800,10,.22],rtol=1e-3)

def test_metrics():
    result=evaluate_predictions([1,2,3],[1,2,4])
    assert result["MAE"]==1/3
    assert result["RMSE"]==np.sqrt(1/3)
    assert result["R2"]==.5

def test_forecast_returns_future_points():
    t=np.arange(20,dtype=float); y=logistic_curve(t,500,10,.3)
    future,pred,params=Forecasting.forecast(t,y,horizon=5)
    assert len(future)==5 and len(pred)==5
    assert future[0]==20 and np.all(np.diff(pred)>0)

def test_completed_notebooks_have_valid_json_and_python():
    import ast, json
    notebook_names=["growth_rate_analysis.ipynb","deaths_per_case.ipynb","logistic_model_fitting.ipynb"]
    for name in notebook_names:
        nb=json.loads((ROOT/"notebooks"/name).read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell["cell_type"]=="code":
                source="".join(cell.get("source",[]))
                ast.parse(source,filename=name)
