import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Replication-padded moving average kernel for trend extraction.
    Preserves input sequence length by padding with the first/last value.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_pad = torch.cat([
            x[:, :1].expand(-1, pad_left),
            x,
            x[:, -1:].expand(-1, pad_right),
        ], dim=1)  # (batch, seq_len + kernel_size - 1)
        return self.avg(x_pad.unsqueeze(1)).squeeze(1)  # (batch, seq_len)


class DLinear(nn.Module):
    """
    DLinear: decomposition-based linear forecasting model.
    Zeng et al. 2023 – "Are Transformers Effective for Time Series Forecasting?"

    Channel-independent: each input is a single univariate series (batch, lookback).
    The same two linear layers are shared across all channels/series.

    Args:
        lookback:    Input context length.
        horizon:     Forecast horizon.
        kernel_size: Moving average kernel size for trend extraction (default: 25).
    """

    def __init__(self, lookback: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)
        self.trend_linear = nn.Linear(lookback, horizon)
        self.seasonal_linear = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback)
        trend = self.moving_avg(x)       # (batch, lookback)
        seasonal = x - trend             # (batch, lookback)
        return self.trend_linear(trend) + self.seasonal_linear(seasonal)  # (batch, horizon)
