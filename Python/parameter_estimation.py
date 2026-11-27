"""Maximum-likelihood fitting for logistic growth under Gaussian residuals."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from continuous_logistic_model import logistic_curve
from model_evaluation import evaluate_predictions

def logistic_model(t,r,K,P0): return logistic_curve(t,K,P0,r)
def negative_log_likelihood(params,t,y):
    r,K,P0,sigma=params
    if r<=0 or K<=0 or P0<=0 or sigma<=0 or P0>=K: return np.inf
    pred=logistic_model(t,r,K,P0)
    return .5*len(y)*np.log(2*np.pi*sigma**2)+np.sum((y-pred)**2)/(2*sigma**2)
def fit_mle(t,y,initial=None):
    t=np.asarray(t,float); y=np.asarray(y,float)
    if t.ndim!=1 or y.ndim!=1 or t.size!=y.size or y.size<4: raise ValueError("Need matching 1-D arrays with >=4 observations")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(y<=0) or np.any(np.diff(t)<=0): raise ValueError("Data must be finite, positive, and t strictly increasing")
    guess=(.15,max(y.max()*1.3,y[-1]*1.01),float(y[0]),max(float(np.std(y)),1.))
    if initial is not None: guess=initial
    bounds=[(1e-6,5),(y.max()*(1+1e-6),max(y.max()*100,1e9)),(1e-8,float(y[0])),(1e-8,None)]
    result=minimize(negative_log_likelihood,guess,args=(t,y),method="L-BFGS-B",bounds=bounds)
    if not result.success: raise RuntimeError("MLE optimizer failed: "+result.message)
    return result
def run_demo(output_csv="results/logistic_mle_demo.csv"):
    rng=np.random.default_rng(42); t=np.linspace(0,30,40); truth=(.35,500.,10.)
    y=np.maximum(logistic_model(t,*truth)+rng.normal(0,15,len(t)),1e-3)
    fit=fit_mle(t,y); r,K,P0,sigma=fit.x; pred=logistic_model(t,r,K,P0)
    table=pd.DataFrame({"parameter":["r","K","P0","sigma"],"estimate":fit.x,"truth":[*truth,15.]})
    path=Path(output_csv); path.parent.mkdir(parents=True,exist_ok=True); table.to_csv(path,index=False)
    print(table.to_string(index=False)); print("Metrics:",evaluate_predictions(y,pred)); return fit,table
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="results/logistic_mle_demo.csv"); run_demo(parser.parse_args().output)
