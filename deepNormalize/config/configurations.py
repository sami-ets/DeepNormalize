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

import torch

from typing import Union, List
from samitorch.configs.configurations import Configuration


class ModelConfiguration(Configuration):
    """
       A base class for storing a ModelTrainer configuration as an object.
       """

    def __init__(self, config):
        super(ModelConfiguration, self).__init__()
        for key in config:
            setattr(self, key, config[key])


class RunningConfiguration(Configuration):
    def __init__(self, config: dict):
        super(RunningConfiguration, self).__init__()
        self._opt_level = config["opt_level"]
        self._num_workers = config["num_workers"]
        self._local_rank = config["local_rank"]
        self._sync_batch_norm = config["sync_batch_norm"]
        self._keep_batch_norm_fp32 = config["keep_batch_norm_fp32"]
        self._loss_scale = config["loss_scale"]
        self._num_gpus = config["num_gpus"]
        self._device = torch.device("cuda:" + str(self._local_rank)) if torch.cuda.is_available() else torch.device(
            "cpu")
        self._is_distributed = config["is_distributed"]
        self._world_size = None

    @property
    def opt_level(self) -> str:
        """
        The optimization level.
            - 00: FP32 training
            - 01: Mixed Precision (recommended)
            - 02: Almost FP16 Mixed Precision
            - 03: FP16 Training.

        Returns:
            str: The optimization level.
        """
        return self._opt_level

    @property
    def num_workers(self) -> int:
        """
        The number of data loading workers (default: 4).

        Returns:
            int: The number of parallel threads.
        """
        return self._num_workers

    @property
    def local_rank(self) -> int:
        """
        The local rank of the distributed node.

        Returns:
            int: The local rank.
        """
        return self._local_rank

    @property
    def sync_batch_norm(self) -> bool:
        """
        Enables the APEX sync of batch normalization.

        Returns:
            bool: Whether if synchronization is enabled or not.
        """
        return self._sync_batch_norm

    @property
    def keep_batch_norm_fp32(self) -> bool:
        """
        Whether to keep the batch normalization in 32-bit floating point (Mixed Precision).

        Returns:
            bool: True will keep 32-bit FP batch norm, False will convert it to 16-bit FP.
        """
        return self._keep_batch_norm_fp32

    @property
    def loss_scale(self) -> str:
        """
        The loss scale in Mixed Precision training.

        Returns:
            The loss scale.
        """
        return self._loss_scale

    @property
    def num_gpus(self) -> int:
        """
        The number of GPUs on the Node to be used for training.

        Returns:
            int: The number of allowed GPU to be used for training.
        """
        return self._num_gpus

    @property
    def device(self) -> torch.device:
        """
        Get the device where Tensors are going to be transfered.

        Returns:
            :obj:`torch.device`: A Torch Device object.
        """
        return self._device

    @property
    def is_distributed(self):
        """
        Whether or not the execution is distributed.

        Returns:
            bool: True if distributed, False otherwise.
        """
        return self._is_distributed

    @is_distributed.setter
    def is_distributed(self, is_distributed):
        self._is_distributed = is_distributed

    @property
    def world_size(self) -> int:
        return self._world_size

    @world_size.setter
    def world_size(self, world_size):
        self._world_size = world_size


class DatasetConfiguration(Configuration):

    def __init__(self, dataset_name, path, validation_split, training_patch_size, training_patch_step,
                 validation_patch_size, validation_patch_step):
        super(DatasetConfiguration, self).__init__()

        self._dataset_name = dataset_name
        self._path = path
        self._validation_split = validation_split
        self._training_patch_size = training_patch_size
        self._training_patch_step = training_patch_step
        self._validation_patch_size = validation_patch_size
        self._validation_patch_step = validation_patch_step

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def path(self):
        """
        Dataset's path.

        Returns:
            str: The dataset's path.
        """
        return self._path

    @property
    def validation_split(self) -> int:
        """
        The number of validation samples.

        Returns:
            int: The number of samples in validation set.
        """
        return self._validation_split

    @property
    def training_patch_size(self):
        """
        The size of a training patch.

        Returns:
            int: The size of a training patch size.
        """
        return self._training_patch_size

    @property
    def training_patch_step(self):
        """
        The step between two training patches.

        Returns:
            int: The step.
        """
        return self._training_patch_step

    @property
    def validation_patch_size(self):
        """
        The size of a validation patch.

        Returns:
            int: The size of a validation patch size.
        """
        return self._validation_patch_size

    @property
    def validation_patch_step(self):
        """
        The step between two validation patches.

        Returns:
            int: The step.
        """
        return self._validation_patch_step

    @classmethod
    def from_dict(cls, dataset_name, config_dict):
        return cls(dataset_name, config_dict["path"], config_dict["validation_split"],
                   config_dict["training"]["patch_size"], config_dict["training"]["step"],
                   config_dict["validation"]["patch_size"], config_dict["validation"]["step"])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Dataset Configuration</h2> \n {}".format(configuration_values)

class DeepNormalizeTrainingConfiguration(Configuration):

    def __init__(self, config: dict):
        super(DeepNormalizeTrainingConfiguration, self).__init__(config)
        self._debug = config["debug"]
        self._batch_size = config["batch_size"]
        self._checkpoint_every = config["checkpoint_every"]
        self._metrics = config["metrics"]
        self._max_epochs = config["max_epochs"]

    @property
    def checkpoint_every(self):
        return self._checkpoint_every

    @property
    def batch_size(self) -> int:
        """
        The batch size used during training.

        Returns:
            int: The number of elements constituting a training batch.
        """
        return self._batch_size

    @property
    def max_iterations(self) -> int:
        """
        The maximum number of epochs during training.

        Returns:
            int: The maximum number of epochs.
        """
        return self._max_epochs

    @property
    def metrics(self) -> Union[list, str]:
        """
        The metrics used during learning.

        Returns:
            str_or_list: The metric used.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def debug(self) -> bool:
        """
        Define if model is in debug mode. If so, run verifications.

        Returns:
            bool: If model is in debug mode.
        """
        return self._debug

    @property
    def max_epochs(self):
        return self._max_epochs


class PretrainingConfiguration(Configuration):

    def __init__(self, config: dict):
        super(PretrainingConfiguration, self).__init__()
        self._pretrained = config["pretrained"]
        self._model_paths = config["model_paths"]

    @property
    def pretrained(self) -> bool:
        return self._pretrained

    @property
    def model_paths(self):
        return self._model_paths


class VariableConfiguration(Configuration):

    def __init__(self, config: dict):
        super(VariableConfiguration, self).__init__()
        self._lambda = config["lambda"]
        self._alpha = config["alpha"]

    @property
    def lambda_(self) -> float:
        """
        The lambda variable between losses.

        Returns:
            float: The lambda variable.
        """
        return self._lambda

    @lambda_.setter
    def lambda_(self, lambda_):
        self._lambda = lambda_

    @property
    def alpha_(self) -> float:
        """
        The autoencoder variable factor.

        Returns:
            float: The alpha variable.
        """
        return self._alpha

    @alpha_.setter
    def alpha_(self, alpha_):
        self._alpha = alpha_


class ModelTrainerConfig(Configuration):
    def __init__(self,
                 model: torch.nn.Module,
                 variables: Configuration,
                 pretraining_config: Configuration,
                 checkpoint_every: int,
                 max_epoch: int,
                 metric,
                 optimizer,
                 criterion
                 ) -> None:
        super(ModelTrainerConfig, self).__init__()
        self._model = model

        self._checkpoint_every = checkpoint_every
        self._max_epoch = max_epoch
        self._metric = metric
        self._optimizer = optimizer
        self._criterion = criterion
        self._variables = variables

        self._pretraining_config = pretraining_config

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def checkpoint_every(self):
        return self._checkpoint_every

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def metric(self):
        return self._metric

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    @property
    def variables(self):
        return self._variables

    @property
    def pretraining_config(self):
        return self._pretraining_config


class LoggerConfiguration(Configuration):

    def __init__(self, config: dict):
        super(LoggerConfiguration, self).__init__()
        self._path = config["path"]
        self._frequency = config["log_after_iterations"]

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def frequency(self) -> int:
        return self._frequency


class VisdomConfiguration(Configuration):

    def __init__(self, config: dict):
        super(VisdomConfiguration, self).__init__()
        self._server = config["server"]
        self._port = config["port"]

    @property
    def server(self):
        return self._server

    @property
    def port(self):
        return self._port


class OptimizerConfiguration(Configuration):
    def __init__(self, config: dict):
        super(OptimizerConfiguration, self).__init__()
        self._type = config["type"]
        self._lr = config["lr"]

        if self._type == "SGD":
            self._momentum = config["momentum"]
        else:
            self._momentum = None

    @property
    def type(self):
        return self._type

    @property
    def lr(self):
        return self._lr

    @property
    def momentum(self):
        return self._momentum


class SchedulerConfiguration(Configuration):
    def __init__(self, config: dict):
        super(SchedulerConfiguration, self).__init__()
        self._mode = config["mode"]
        self._factor = config["factor"]
        self._patience = config["patience"]

    @property
    def mode(self):
        return self._mode

    @property
    def factor(self):
        return self._factor

    @property
    def patience(self):
        return self._patience
