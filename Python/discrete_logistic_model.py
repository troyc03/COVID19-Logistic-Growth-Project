"""Discrete logistic growth model and fitting utilities.

The recurrence is
    P[t + 1] = P[t] + r * P[t] * (1 - P[t] / K)

Time is measured in discrete steps (for example, weeks for weekly COVID data).
The model is most useful for a single approximately logistic growth wave; it
should not be fit blindly to a multi-wave cumulative COVID-19 series.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def discrete_logistic_growth(P0, r, K, n_steps):
    """Simulate n_steps of discrete logistic growth and return n_steps + 1 values."""
    if K <= 0:
        raise ValueError("K must be positive.")
    if n_steps < 0 or int(n_steps) != n_steps:
        raise ValueError("n_steps must be a non-negative integer.")
    if P0 < 0:
        raise ValueError("P0 must be non-negative.")

    population = np.empty(int(n_steps) + 1, dtype=float)
    population[0] = float(P0)
    for step in range(int(n_steps)):
        current = population[step]
        population[step + 1] = current + r * current * (1.0 - current / K)
    return population


def simulate_discrete_logistic_growth(P0, r, K, t):
    """Evaluate the recurrence at integer time steps supplied in t."""
    times = np.asarray(t)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("t must be a finite one-dimensional array.")
    if np.any(times < 0) or not np.all(times == np.floor(times)):
        raise ValueError("All time values must be non-negative integers.")
    if times.size == 0:
        return np.array([], dtype=float)
    trajectory = discrete_logistic_growth(P0, r, K, int(times.max()))
    return trajectory[times.astype(int)]


def fit_discrete_logistic_growth(t, observed, initial_guess=None, maxfev=20000):
    """Fit r and K to observations; the first observed value is treated as P0."""
    times = np.asarray(t, dtype=float)
    values = np.asarray(observed, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise ValueError("t and observed must be one-dimensional arrays of equal length.")
    if times.size < 3:
        raise ValueError("At least three observations are required to fit r and K.")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("t and observed must contain only finite values.")
    if np.any(times < 0) or not np.all(times == np.floor(times)):
        raise ValueError("All time values must be non-negative integers.")
    if np.any(values < 0):
        raise ValueError("Observed population values must be non-negative.")
    if np.unique(times).size != times.size:
        raise ValueError("Time values must be unique.")

    order = np.argsort(times)
    times = times[order].astype(int)
    values = values[order]
    if times[0] != 0:
        raise ValueError("Include the initial observation at t=0 so P0 is defined.")
    if not np.array_equal(times, np.arange(times[-1] + 1)):
        raise ValueError("Observations must cover consecutive time steps.")

    P0 = float(values[0])
    guess = initial_guess or (0.2, max(float(values.max()) * 1.2, P0 + 1.0))

    def model(time, r, K):
        return simulate_discrete_logistic_growth(P0, r, K, time)

    parameters, covariance = curve_fit(
        model,
        times,
        values,
        p0=guess,
        bounds=([0.0, max(P0, np.finfo(float).eps)], [3.0, np.inf]),
        maxfev=maxfev,
    )
    return {"r": float(parameters[0]), "K": float(parameters[1]),
            "covariance": covariance, "P0": P0}


def calculate_metrics(observed, predicted):
    """Return common regression metrics for observed and predicted values."""
    actual = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if actual.shape != estimate.shape or actual.size == 0:
        raise ValueError("observed and predicted must have the same non-empty shape.")
    errors = actual - estimate
    mse = float(np.mean(errors ** 2))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = float(1.0 - np.sum(errors ** 2) / denominator) if denominator else float("nan")
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r_squared,
    }


def plot_discrete_fit(t, observed, predicted, title="Discrete Logistic Growth Fit"):
    """Plot observed data against a fitted discrete logistic trajectory."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(t, observed, label="Observed", color="black", s=24)
    ax.plot(t, predicted, label="Discrete logistic model", linewidth=2)
    ax.set(title=title, xlabel="Time steps", ylabel="Population")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_cobweb(r, K, P0, n_steps=50, ax=None):
    """Plot a cobweb diagram for the discrete recurrence."""
    if K <= 0:
        raise ValueError("K must be positive.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    p_values = np.linspace(0, K * 1.5, 500)
    update = p_values + r * p_values * (1.0 - p_values / K)
    ax.plot(p_values, update, label=r"$P_{t+1}=f(P_t)$")
    ax.plot(p_values, p_values, "k--", alpha=0.6, label=r"$P_{t+1}=P_t$")

    trajectory = discrete_logistic_growth(P0, r, K, n_steps)
    for current, following in zip(trajectory[:-1], trajectory[1:]):
        ax.plot([current, current], [current, following], "r-", linewidth=0.8, alpha=0.6)
        ax.plot([current, following], [following, following], "r-", linewidth=0.8, alpha=0.6)

    ax.set(xlabel=r"Current population $P_t$", ylabel=r"Next population $P_{t+1}$",
           title=f"Cobweb diagram (r={r:g}, K={K:g})")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    return ax


if __name__ == "__main__":
    # Synthetic self-check: the fitting routine should recover known parameters.
    t = np.arange(31)
    actual = discrete_logistic_growth(P0=10, r=0.35, K=500, n_steps=30)
    fitted = fit_discrete_logistic_growth(t, actual, initial_guess=(0.25, 450))
    prediction = simulate_discrete_logistic_growth(fitted["P0"], fitted["r"], fitted["K"], t)
    print("Fitted parameters:", {key: fitted[key] for key in ("P0", "r", "K")})
    print("Fit metrics:", calculate_metrics(actual, prediction))

    plot_discrete_fit(t, actual, prediction, title="Synthetic Discrete Logistic Model Check")
    plt.show()
