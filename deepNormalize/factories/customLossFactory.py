from enum import Enum
from deepNormalize.nn.criterions import MultipleDatasetLoss


class LossType(Enum):
    MultipleDatasetLoss = "MultipleDatasetLoss"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        else:
            return False


class CustomLossFactory(object):
    def create(self, criterion, params):
        if criterion == LossType.MultipleDatasetLoss:
            return MultipleDatasetLoss(**params)
