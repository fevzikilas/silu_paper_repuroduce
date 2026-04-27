from __future__ import annotations

import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class dSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = torch.sigmoid(x)
        ctx.save_for_backward(x, sigmoid_x)
        return sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, sigmoid_x = ctx.saved_tensors
        dsilu_dx = sigmoid_x * (1.0 - sigmoid_x)
        dsilu_dx = dsilu_dx + sigmoid_x + x * dsilu_dx * (1.0 - 2.0 * sigmoid_x)
        return grad_output * dsilu_dx


class dSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dSiLUFunction.apply(x)


def build_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return SiLU()
    if normalized == "dsilu":
        return dSiLU()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class ShallowNetwork(nn.Module):
    def __init__(self, input_size: int = 460, hidden_size: int = 50, activation: str = "relu"):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.hidden(x))
        return self.output(x).squeeze(-1)