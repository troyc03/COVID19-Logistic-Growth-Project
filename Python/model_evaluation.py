"""Model evaluation metrics."""
import numpy as np

def evaluate_predictions(actual,predicted):
    y=np.asarray(actual,float); p=np.asarray(predicted,float)
    if y.shape!=p.shape or y.size==0: raise ValueError("actual and predicted must have equal non-empty shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)): raise ValueError("Inputs must be finite")
    e=y-p; mse=float(np.mean(e**2)); denom=float(np.sum((y-y.mean())**2))
    mape=float(np.mean(np.abs(e[y!=0]/y[y!=0]))*100) if np.any(y!=0) else float("nan")
    return {"MAE":float(np.mean(np.abs(e))),"MSE":mse,"RMSE":float(np.sqrt(mse)),"MAPE_percent":mape,"R2":float(1-np.sum(e**2)/denom) if denom else float("nan")}
