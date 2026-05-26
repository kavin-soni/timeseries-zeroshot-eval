import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_metrics(y_true, y_pred, y_train_hist, seasonality=1):
    """
    Calculates standard forecasting metrics: MAE, RMSE, sMAPE, MASE, and Bias.

    Args:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
        y_train_hist (np.array): Historical training data (used for MASE).
        seasonality (int): Seasonality period for MASE calculation.
    """
    # 1. MAE & RMSE
    mae = np.mean(np.abs(y_true - y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # 2. sMAPE
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    smape = np.mean(200 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) if np.sum(mask) > 0 else 0.0

    # 3. MASE
    if len(y_train_hist) > seasonality:
        naive_errors = np.abs(y_train_hist[seasonality:] - y_train_hist[:-seasonality])
        mae_naive = np.mean(naive_errors)
    elif len(y_train_hist) > 1:
        mae_naive = np.mean(np.abs(np.diff(y_train_hist)))
    else:
        mae_naive = 1.0

    mase = mae / mae_naive if mae_naive > 1e-9 else np.nan

    # 4. Forecast Bias
    sum_actual = np.sum(y_true)
    bias = (np.sum(y_pred) - sum_actual) / (sum_actual + 1e-10) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'sMAPE': smape,
        'MASE': mase,
        'Bias': bias
    }
