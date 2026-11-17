import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from parameter_estimation import *

mse = mean_squared_error(y_data, y_pred_data)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_data, y_pred_data)
r2 = r2_score(y_data, y_pred_data)

# Print the final performance report
print("\n=== Model Performance Evaluation ===")
print(f"Mean Squared Error (MSE):           {mse:.4f}")
print(f"Root Mean Squared Error (RMSE):     {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")
print(f"R² Coefficient of Determination:     {r2:.4f}")