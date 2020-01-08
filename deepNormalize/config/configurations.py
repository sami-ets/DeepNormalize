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


class DatasetConfiguration(Configuration):
    def __init__(self, dataset_name, path, modalities, validation_split, training_patch_size, training_patch_step,
                 validation_patch_size, validation_patch_step, test_patch_size, test_patch_step):
        super(DatasetConfiguration, self).__init__()

        self._dataset_name = dataset_name
        self._path = path
        self._modalities = modalities
        self._validation_split = validation_split
        self._training_patch_size = training_patch_size
        self._training_patch_step = training_patch_step
        self._validation_patch_size = validation_patch_size
        self._validation_patch_step = validation_patch_step
        self._test_patch_size = test_patch_size
        self._test_patch_step = test_patch_step

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
    def modalities(self):
        """
        Dataset's modalities.

        Returns:
            str or List of str: The dataset's modalities.
        """
        return self._modalities

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

    @property
    def test_patch_size(self):
        return self._test_patch_size

    @property
    def test_patch_step(self):
        return self._test_patch_step

    @classmethod
    def from_dict(cls, dataset_name, config_dict):
        return cls(dataset_name, config_dict["path"], config_dict["validation_split"],
                   config_dict["training"]["patch_size"], config_dict["training"]["step"],
                   config_dict["validation"]["patch_size"], config_dict["validation"]["step"],
                   config_dict["test"]["patch_size"], config_dict["test"]["step"])

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
        self._lambda = config["disc_ratio"]
        self._alpha = config["alpha"]

    @property
    def lambda_(self) -> float:
        """
        The disc_ratio variable between losses.

        Returns:
            float: The disc_ratio variable.
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
