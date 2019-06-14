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

from samitorch.configs.configurations import DatasetConfiguration, TrainingConfiguration, Configuration


class RunningConfiguration(Configuration):
    def __init__(self, config: dict):
        self._opt_level = config["opt_level"]
        self._num_workers = config["num_workers"]
        self._local_rank = config["local_rank"]
        self._sync_batch_norm = config["sync_batch_norm"]
        self._keep_batch_norm_fp32 = config["keep_batch_norm_fp32"]
        self._loss_scale = config["loss_scale"]
        self._num_gpus = config["num_gpus"]
        self._device = torch.device("cuda:" + str(self._local_rank)) if torch.cuda.is_available() else torch.device(
            "cpu")

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


class DeepNormalizeDatasetConfiguration(DatasetConfiguration):

    def __init__(self, config: dict):
        super(DeepNormalizeDatasetConfiguration, self).__init__()

        self._path = config["path"]
        self._training_patch_size = config["training"]["patch_size"]
        self._validation_patch_size = config["validation"]["patch_size"]

    @property
    def path(self):
        """
        Dataset's path.

        Returns:
            str: The dataset's path.
        """
        return self._path

    @property
    def training_patch_size(self):
        """
        The size of a training patch.

        Returns:
            int: The size of a training patch size.
        """
        return self._training_patch_size

    @property
    def validation_patch_size(self):
        """
        The size of a validation patch.

        Returns:
            int: The size of a validation patch size.
        """
        return self._validation_patch_size


class DeepNormalizeTrainingConfiguration(TrainingConfiguration):

    def __init__(self, config: dict):
        super(DeepNormalizeTrainingConfiguration, self).__init__()

        self._batch_size = config["batch_size"]
        self._checkpoint_every = config["checkpoint_every"]
        self._criterions = config["criterions"]
        self._metrics = config["metrics"]
        self._optimizer = config["optimizer"]
        self._max_epoch = config["max_epoch"]

    @property
    def batch_size(self):
        """
        The batch size used during training.

        Returns:
            int: The number of elements constituting a training batch.
        """
        return self._batch_size

    @property
    def max_epoch(self):
        """
        The maximum number of epochs during training.

        Returns:
            int: The maximum number of epochs.
        """
        return self._max_epoch

    @property
    def checkpoint_every(self):
        """
        The frequency at which we want to save a checkpoint of the model.

        Returns:
            int: The frequency (in epoch).
        """
        return self._checkpoint_every

    @property
    def criterions(self):
        """
        The criterion used for model optimization.

        Returns:
            str: A criterion used.
        """
        return self._criterions

    @property
    def metrics(self):
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
    def optimizer(self):
        """
        The optimizer used for learning.

        Returns:
            str: The optimizer used.
        """
        return self._optimizer
