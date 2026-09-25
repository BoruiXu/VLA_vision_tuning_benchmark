"""LoRA for Florence's right-multiplied bare image_projection matrix."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class MatrixLoRA(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.scale = float(alpha) / rank
        # W has shape [in, out], hence Delta W = (B @ A).T.
        self.low_rank_A = nn.Parameter(weight.new_empty(rank, weight.shape[0]))
        self.low_rank_B = nn.Parameter(weight.new_zeros(weight.shape[1], rank))
        nn.init.normal_(self.low_rank_A, std=1.0 / rank)

    def forward(self, original):
        return original + (self.low_rank_B @ self.low_rank_A).T * self.scale


def install(vlm, rank, alpha):
    if parametrize.is_parametrized(vlm, "image_projection"):
        raise RuntimeError("projector already parametrized")
    vlm.image_projection.requires_grad_(False)
    parametrize.register_parametrization(vlm, "image_projection", MatrixLoRA(vlm.image_projection, rank, alpha))
    vlm.image_proj_norm.requires_grad_(False)


def merge(vlm):
    parametrize.remove_parametrizations(vlm, "image_projection", leave_parametrized=True)
