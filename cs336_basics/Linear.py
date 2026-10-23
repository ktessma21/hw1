

import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        super().__init__()

        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, 0, std, -3*std, 3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,oi->...o", x, self.W)
