"""Continuous logistic growth model and fitting utilities.

The model is dP/dt = r * P * (1 - P/K), where:
- P(t) is cumulative population/case count,
- r is the growth rate (per time unit),
- K is the carrying capacity (upper asymptote).

For this project, time is measured in weeks when fitting weekly COVID-19 data.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


def logistic_curve(t, K, P0, r):
    """Evaluate the analytical logistic solution at times t."""
    t = np.asarray(t, dtype=float)
    if K <= 0 or P0 <= 0 or r <= 0:
        raise ValueError("K, P0, and r must all be positive.")
    if P0 >= K:
        raise ValueError("P0 must be smaller than K for logistic growth.")

    return K / (1.0 + ((K - P0) / P0) * np.exp(-r * t))


def logistic_growth(t, population, r, K):
    """Derivative function for solve_ivp: dP/dt."""
    return r * population * (1.0 - population / K)


def simulate_logistic_growth(P0, r, K, t):
    """Numerically solve the logistic differential equation at times t."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a one-dimensional array with at least 2 points.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if P0 <= 0 or K <= 0 or r <= 0 or P0 >= K:
        raise ValueError("Require 0 < P0 < K and r > 0.")

    solution = solve_ivp(
        fun=logistic_growth,
        t_span=(float(t[0]), float(t[-1])),
        y0=[float(P0)],
        t_eval=t,
        args=(float(r), float(K)),
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Logistic ODE solver failed: {solution.message}")
    return solution.y[0]


def fit_logistic_growth(t, observed, initial_guess=None, maxfev=20_000):
    """Fit K, P0, and r to observations using nonlinear least squares.

    Returns (parameters, covariance), where parameters are ordered [K, P0, r].
    """
    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if t.ndim != 1 or observed.ndim != 1 or t.size != observed.size:
        raise ValueError("t and observed must be one-dimensional arrays of equal length.")
    if t.size < 4:
        raise ValueError("At least four observations are required to fit three parameters.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(observed)):
        raise ValueError("t and observed must contain only finite values.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if np.any(observed <= 0):
        raise ValueError("All observed values must be positive for this logistic model.")

    p0 = float(observed[0])
    if initial_guess is None:
        k0 = max(float(observed.max()) * 1.2, p0 * 1.5)
        r0 = 0.2
        initial_guess = (k0, p0, r0)

    lower_bounds = (max(float(observed.max()), p0) * (1.0 + 1e-9), 1e-12, 1e-8)
    upper_bounds = (np.inf, float(observed.max()), 5.0)

    parameters, covariance = curve_fit(
        logistic_curve,
        t,
        observed,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=maxfev,
    )
    return parameters, covariance


def evaluate_predictions(actual, predicted):
    """Return common regression metrics without requiring scikit-learn."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must have the same non-empty shape.")

    errors = actual - predicted
    mse = float(np.mean(errors**2))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = float(1.0 - np.sum(errors**2) / denominator) if denominator > 0 else float("nan")

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r_squared,
    }


def plot_logistic_fit(t, observed, parameters, title="Continuous Logistic Growth Fit"):
    """Plot observed values and the fitted logistic curve."""
    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)
    K, P0, r = parameters

    t_smooth = np.linspace(float(t.min()), float(t.max()), 300)
    plt.figure(figsize=(10, 6))
    plt.scatter(t, observed, label="Observed data", alpha=0.75)
    plt.plot(t_smooth, logistic_curve(t_smooth, K, P0, r), label="Logistic fit")
    plt.axhline(K, linestyle="--", alpha=0.7, label=f"Carrying capacity K={K:,.0f}")
    plt.xlabel("Time (weeks, if using weekly data)")
    plt.ylabel("Cumulative cases")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Small synthetic example to verify that the analytical and numerical
    # solutions agree. Replace this demo with one epidemic wave for analysis.
    time = np.linspace(0, 20, 100)
    initial_population, growth_rate, carrying_capacity = 10.0, 0.35, 500.0

    numerical = simulate_logistic_growth(
        initial_population, growth_rate, carrying_capacity, time
    )
    analytical = logistic_curve(
        time, carrying_capacity, initial_population, growth_rate
    )

    max_difference = float(np.max(np.abs(numerical - analytical)))
    print(f"Maximum analytical/numerical difference: {max_difference:.6g}")
    assert np.allclose(numerical, analytical, rtol=1e-5, atol=1e-7)

    fitted_parameters, _ = fit_logistic_growth(time, analytical)
    print(
        "Recovered parameters [K, P0, r]: "
        f"{fitted_parameters[0]:.3f}, "
        f"{fitted_parameters[1]:.3f}, "
        f"{fitted_parameters[2]:.4f}"
    )
    plot_logistic_fit(time, analytical, fitted_parameters, title="Synthetic Logistic Model Check")
