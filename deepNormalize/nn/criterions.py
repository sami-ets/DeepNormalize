import torch
from torch.nn.modules.loss import _Loss
from typing import Union


class MeanLoss(_Loss):
    def __init__(self):
        super(MeanLoss, self).__init__()
        pass

    def forward(self, inputs: torch.Tensor, target: Union[None, torch.Tensor]):
        return torch.mean(inputs)
