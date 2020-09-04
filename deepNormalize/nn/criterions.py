import torch
from kerosene.nn.criterions import CriterionFactory
from torch.nn.modules.loss import _Loss
from deepNormalize.utils.constants import EPSILON


class MeanLoss(_Loss):
    def __init__(self):
        super(MeanLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return torch.mean(inputs)


class CustomCriterionFactory(CriterionFactory):
    def __init__(self):
        super().__init__()
        self.register("MeanLoss", MeanLoss)
        self.register("DualDatasetLoss", DualDatasetLoss)
        self.register("MultipleDatasetLoss", MultipleDatasetLoss)


class MultipleDatasetLoss(_Loss):
    def __init__(self, reduction="sum"):
        super(MultipleDatasetLoss, self).__init__()
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = torch.nn.functional.softmax(inputs, dim=1)

        target_fake = torch.Tensor().new_full(targets.size(), inputs.shape[1] - 1, dtype=torch.long,
                                              device=inputs.device)

        pred_real = inputs.gather(1, targets.view(-1, 1))
        pred_fake = inputs.gather(1, target_fake.view(-1, 1))

        ones = torch.Tensor().new_ones(pred_fake.shape, device=inputs.device, dtype=pred_real.dtype,
                                       requires_grad=False)

        if self._reduction == "sum":
            # return torch.log(pred_real + pred_fake).sum()
            return -torch.log(ones - ((pred_real + EPSILON) + (pred_fake + EPSILON))).sum()
        elif self._reduction == "mean":
            # return torch.log(pred_real + pred_fake).mean()
            return -torch.log(ones - ((pred_real + EPSILON) + (pred_fake + EPSILON))).mean()
        else:
            return NotImplementedError


class DualDatasetLoss(_Loss):
    def __init__(self):
        super(DualDatasetLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        ones = torch.Tensor().new_ones(targets.shape, device=targets.device, dtype=targets.dtype)
        inverse_target = (ones - targets)
        return torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(inputs, dim=1), inverse_target)
