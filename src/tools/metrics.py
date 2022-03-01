import torch.nn as nn
import torch


class R2Score(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - torch.sum((input - target)**2)/torch.sum((target - target.mean())**2)


class MAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 100*torch.sum(torch.abs((input-target)/target))/target.shape[0]


class WAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 100*torch.sum(torch.abs(input-target))/torch.sum(torch.abs(target))
