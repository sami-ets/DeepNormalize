#  -*- coding: utf-8 -*-
#  Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import abc
import torch

from samitorch.metrics.metrics import Dice
from samitorch.losses.losses import DiceLoss


class AbstractLayerFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_layer(self, function: str, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator):
        raise NotImplementedError


class AbstractMetricFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_metric(self, function: str, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, function: str, creator):
        raise NotImplementedError


class MetricsFactory(AbstractMetricFactory):
    def __init__(self):
        super(MetricsFactory, self).__init__()

        self._metrics = {
            "Dice": Dice
        }

    def create_metric(self, metric: str, *args):
        """
        Instanciate an optimizer based on its name.

        Args:
            metric (Enum): The optimizer's name.
            *args: Other arguments.

        Returns:
            The metric, a torch or ignite module.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        metric = self._metrics[metric]
        return metric

    def register(self, metric: str, creator):
        """
        Add a new activation layer.

        Args:
           metric (str): Metric's name.
           creator: A torch or ignite module object wrapping the new custom metric function.
        """
        self._metrics[metric] = creator


class OptimizerFactory(AbstractLayerFactory):

    def __init__(self):
        super(OptimizerFactory, self).__init__()

        self._optimizers = {
            "Adam": torch.optim.Adam,
            "Adagrad": torch.optim.Adagrad,
            "SGD": torch.optim.SGD
        }

    def create_layer(self, function: str, *args):
        """
        Instanciate an optimizer based on its name.

        Args:
            function (Enum): The optimizer's name.
            *args: Other arguments.

        Returns:
            :obj:`torch.optim.Optimizer`: The optimizer.

        Raises:
            KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        optimizer = self._optimizers[function]
        return optimizer

    def register(self, function: str, creator: torch.optim.Optimizer):
        """
        Add a new activation layer.

        Args:
           function (str): Activation layer name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom optimizer function.
        """
        self._optimizers[function] = creator


class CriterionFactory(AbstractLayerFactory):
    def __init__(self):
        super(CriterionFactory, self).__init__()

        self._criterion = {
            "DiceLoss": DiceLoss,
            "Cross_Entropy": torch.nn.functional.cross_entropy
        }

    def create_layer(self, function: str, *args):
        """
        Instanciate a loss function based on its name.

        Args:
           function (Enum): The criterion's name.
           *args: Other arguments.

        Returns:
           :obj:`torch.nn.Module`: The criterion.

        Raises:
           KeyError: Raises KeyError Exception if Activation Function is not found.
        """
        optimizer = self._criterion[function]
        return optimizer

    def register(self, function: str, creator):
        """
        Add a new criterion.

        Args:
           function (str): Criterion's name.
           creator (:obj:`torch.nn.Module`): A torch module object wrapping the new custom criterion function.
        """
        self._criterion[function] = creator
