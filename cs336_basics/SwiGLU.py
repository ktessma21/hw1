import torch
from jaxtyping import Float
from torch import Tensor, nn

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, "d_ff d_model"],
        w2_weight: Float[Tensor, "d_model d_ff"],
        w3_weight: Float[Tensor, "d_ff d_model"],
    ):
        super().__init__()

        self.w1 = torch.nn.Parameter(w1_weight.detach().clone())
        self.w2 = torch.nn.Parameter(w2_weight.detach().clone())
        self.w3 = torch.nn.Parameter(w3_weight.detach().clone())
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
