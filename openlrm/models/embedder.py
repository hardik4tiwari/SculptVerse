import torch
import torch.nn as nn


class CameraEmbedder(nn.Module):
    """
    Embed camera features to a high-dimensional vector.
    
    Reference:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, raw_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.compile
    def forward(self, x):
        return self.mlp(x)
