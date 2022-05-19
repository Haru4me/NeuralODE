import torch.nn as nn
import torch.nn.functional as F
import torch


class R2Score(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return 1 - torch.sum((input - target)**2)/(torch.sum((target - target.mean())**2)+eps)


class MAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return torch.mean(torch.abs((input-target)/(target+eps)))


class WAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return torch.sum(torch.abs(input-target))/(torch.sum(torch.abs(target))+eps)


class SMAPE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return torch.mean(torch.abs(2*(input-target))/(input+target+eps))


class RMSE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(input, target))


class MAGE(nn.Module):

    def __init__(self, dz: float=0.5):

        super().__init__()
        self.dz = dz

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(target-input) > self.dz)


class MyMetric(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps=1e-8
        return F.mse_loss(inputs[-1], target)/(F.mse_loss(inputs[0], target) + eps)

class WAE(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        weight = 1/(torch.abs(inputs - target[:, 0])+eps)
        ae = torch.abs(inputs - target[:, 1])
        return torch.mean(ae) + 0.01*torch.sum(weight)
