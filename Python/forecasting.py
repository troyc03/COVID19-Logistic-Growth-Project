"""Forecasting utilities for fitted continuous logistic growth."""
import numpy as np
from scipy.optimize import curve_fit
from continuous_logistic_model import logistic_curve

class Forecasting:

    @staticmethod
    def fit(t,observed,initial_guess=None):
        t=np.asarray(t,float); y=np.asarray(observed,float)
        if t.ndim!=1 or y.ndim!=1 or t.size!=y.size or t.size<4: raise ValueError("Need matching 1-D arrays with at least 4 points.")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(y<=0) or np.any(np.diff(t)<=0): raise ValueError("Require finite positive observations and strictly increasing t.")
        p0=initial_guess or (max(y.max()*1.2,y[0]*1.5),float(y[0]),.15)
        bounds=([y.max()*(1+1e-8),1e-12,1e-8],[np.inf,y.max(),5.])
        return curve_fit(logistic_curve,t,y,p0=p0,bounds=bounds,maxfev=30000)

    @staticmethod
    def predict(t,params):
        K,P0,r=params; return logistic_curve(t,K,P0,r)

    @staticmethod
    def forecast(t_observed,y_observed,horizon=30,params=None):
        t=np.asarray(t_observed,float)
        if horizon<0: raise ValueError("horizon must be >=0")
        if params is None: params,_=Forecasting.fit(t,y_observed)
        future=np.arange(t[-1]+1,t[-1]+horizon+1,dtype=float)
        return future,Forecasting.predict(future,params),params

