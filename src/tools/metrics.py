import torch.nn as nn
import torch.nn.functional as F
import torch


class R2Score(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-15
        return 1 - torch.sum((input - target)**2)/(torch.sum((target - target.mean())**2)+eps)


class MAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-15
        return 100*torch.sum(torch.abs((input-target)/(target+eps)))/target.shape[0]


class WAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-15
        return 100*torch.sum(torch.abs(input-target))/(torch.sum(torch.abs(target))+eps)


class MyMetric(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps=1e-15
        return F.mse_loss(inputs[-1], target)/(F.mse_loss(inputs[0], target) + eps)
