"""Discrete logistic growth map and visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt

def discrete_logistic_growth(p0, r, K, t_max):
    if p0 < 0 or K <= 0 or t_max < 0 or not np.isfinite(r):
        raise ValueError("Require p0 >= 0, K > 0, finite r and t_max >= 0.")
    values=np.empty(int(t_max)+1,float); values[0]=p0
    for i in range(int(t_max)):
        values[i+1]=values[i]+r*values[i]*(1-values[i]/K)
    return values

def mapping(p,r,K):
    p=np.asarray(p,float); return p+r*p*(1-p/K)

def plot_dynamics(p0=10,K=500,t_max=50,r_values=(.7,1.9,2.6)):
    fig,axes=plt.subplots(len(r_values),2,figsize=(12,4*len(r_values)))
    if len(r_values)==1: axes=np.array([axes])
    grid=np.linspace(0,1.5*K,500)
    for i,r in enumerate(r_values):
        series=discrete_logistic_growth(p0,r,K,t_max); ax=axes[i,0]
        ax.plot(np.arange(len(series)),series,marker=".",ms=3); ax.axhline(K,ls="--",label="K")
        ax.set(title=f"Time series r={r:g}",xlabel="n",ylabel="P_n"); ax.grid(alpha=.25); ax.legend()
        ax=axes[i,1]; ax.plot(grid,mapping(grid,r,K),label="f(P)"); ax.plot(grid,grid,"k--",label="identity")
        for p,q in zip(series[:-1],series[1:]):
            ax.plot([p,p],[p,q],color="tab:red",alpha=.4,lw=.8); ax.plot([p,q],[q,q],color="tab:red",alpha=.4,lw=.8)
        ax.set(title=f"Cobweb r={r:g}",xlabel="P_n",ylabel="P_(n+1)"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); return fig,axes
if __name__=="__main__":
    plot_dynamics(); plt.show()
