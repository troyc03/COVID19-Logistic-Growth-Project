import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# 1. Define the deterministic logistic growth model in terms of P(t) and P0
def logistic_model(t, r, K, P0):
    return (K * P0 * np.exp(r * t)) / (K + P0 * (np.exp(r * t) - 1))

# 2. Define the negative log-likelihood function
def neg_log_likelihood(params, t, y):
    r, K, P0, sigma = params
    
    # Boundary constraints / Penalties for physical realism
    if r <= 0 or K <= 0 or P0 <= 0 or sigma <= 0:
        return 1e10
        
    # Model predictions
    y_pred = logistic_model(t, r, K, P0)
    
    # Calculate negative log-likelihood components
    n = len(y)
    nll = (n / 2) * np.log(2 * np.pi) + n * np.log(sigma) + (1 / (2 * sigma**2)) * np.sum((y - y_pred)**2)
    return nll

# Set seed for reproducibility
np.random.seed(42)

true_r = 0.35      # Growth rate
true_K = 500.0     # Carrying capacity
true_P0 = 10.0     # Initial population
true_sigma = 15.0  # Noise standard deviation

# Generate simulated synthetic data
t_data = np.linspace(0, 30, 40)
clean_y = logistic_model(t_data, true_r, true_K, true_P0)
y_data = clean_y + np.random.normal(0, true_sigma, size=len(t_data))

# Initial guesses: [r, K, P0, sigma]
initial_guess = [0.1, max(y_data), y_data[0], np.std(y_data)]

# Run optimization using L-BFGS-B via SciPy
result = minimize(
    neg_log_likelihood, 
    initial_guess, 
    args=(t_data, y_data), 
    method='L-BFGS-B',
    bounds=[(1e-5, 2.0), (1e-5, 2000.0), (1e-5, 500.0), (1e-5, 100.0)]
)

# Extract estimates
est_r, est_K, est_P0, est_sigma = result.x

# Save estimated parameters as a CSV file using pandas
df_estimates = pd.DataFrame({
    'Parameter': ['r', 'K', 'P0', 'sigma'],
    'True_Value': [true_r, true_K, true_P0, true_sigma],
    'MLE_Estimate': [est_r, est_K, est_P0, est_sigma]
})
df_estimates.to_csv('logistic_mle_estimates.csv', index=False)
print("\n[Success] Estimates exported to 'logistic_mle_estimates.csv'")

# Generate fine grid for smooth plotting
t_fine = np.linspace(0, 30, 200)
y_fit = logistic_model(t_fine, est_r, est_K, est_P0)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(t_data, y_data, color='darkorange', alpha=0.7, label='Observed Data (Noisy)')
plt.plot(t_fine, logistic_model(t_fine, true_r, true_K, true_P0), 'g--', label='True Curve')
plt.plot(t_fine, y_fit, 'b-', linewidth=2, label=f'MLE Fit (r={est_r:.2f}, K={est_K:.1f})')
plt.fill_between(t_fine, y_fit - 1.96 * est_sigma, y_fit + 1.96 * est_sigma, color='blue', alpha=0.1, label='95% Data Bounds')

plt.xlabel('Time (t)')
plt.ylabel('Population P(t)')
plt.title('Maximum Likelihood Estimation of Logistic Growth')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Generate predictions for the actual observed data time points to compute performance metrics
y_pred_data = logistic_model(t_data, est_r, est_K, est_P0)


