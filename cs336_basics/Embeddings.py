import torch 
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        torch.nn.init.trunc_normal_(self.W, 0, 1, -3, 3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]