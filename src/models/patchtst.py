import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """
    PatchTST: Patch-based Transformer for time series forecasting.
    Nie et al. 2023 — "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

    Channel-independent: input is a single univariate series (batch, lookback).
    Each series is divided into overlapping patches, embedded, then processed by a
    standard Transformer encoder. A linear head maps the flattened output to the horizon.

    Args:
        lookback:    Input context length.
        horizon:     Forecast horizon.
        patch_len:   Length of each patch (default: 16).
        stride:      Step between consecutive patches (default: 8).
        d_model:     Transformer embedding dimension (default: 128).
        n_heads:     Number of attention heads (default: 16).
        e_layers:    Number of Transformer encoder layers (default: 3).
        d_ff:        Feedforward dimension inside each encoder layer (default: 256).
        dropout:     Dropout rate (default: 0.2).
    """

    def __init__(
        self,
        lookback:  int,
        horizon:   int,
        patch_len: int = 16,
        stride:    int = 8,
        d_model:   int = 128,
        n_heads:   int = 16,
        e_layers:  int = 3,
        d_ff:      int = 256,
        dropout:   float = 0.2,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride

        n_patches = (lookback - patch_len) // stride + 1
        self.n_patches = n_patches

        # Patch embedding: project each patch vector to d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.embed_dropout   = nn.Dropout(dropout)

        # Learned positional embedding (one vector per patch position)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer encoder (pre-norm / norm_first=True, as in the paper)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = d_ff,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm        = nn.LayerNorm(d_model)

        # Prediction head: flatten all patch representations → horizon
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback)

        # 1. Patchify: (batch, n_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 2. Embed + positional encoding
        out = self.patch_embedding(patches) + self.pos_embedding  # (batch, n_patches, d_model)
        out = self.embed_dropout(out)

        # 3. Transformer encoder
        out = self.transformer(out)   # (batch, n_patches, d_model)
        out = self.norm(out)

        # 4. Flatten and project
        out = out.flatten(1)          # (batch, n_patches * d_model)
        return self.head(out)         # (batch, horizon)
