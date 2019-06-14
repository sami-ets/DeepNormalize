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

import yaml
import logging

from ignite.metrics.accuracy import Accuracy
from samitorch.configs.configurations import Configuration
from samitorch.factories.parsers import AbstractConfigurationParserFactory
from samitorch.configs.configurations import UNetModelConfiguration, ResNetModelConfiguration, DiceMetricConfiguration

from deepNormalize.config.configurations import DeepNormalizeDatasetConfiguration, DeepNormalizeTrainingConfiguration


class DeepNormalizeModelsParserFactory(AbstractConfigurationParserFactory):

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                unets = [UNetModelConfiguration(config["models"][i][element]) for i, element in
                         enumerate(["preprocessor", "segmenter"])]
                resnet = [ResNetModelConfiguration(config["models"][2]["discriminator"])]
                return unets + resnet
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class DeepNormalizeDatasetConfigurationParserFactory(AbstractConfigurationParserFactory):
    def __init__(self) -> None:
        pass

    def parse(self, path: str):
        """
        Parse a dataset configuration file.

        Args:
           path (str): Configuration YAML file path.

        Returns:
           :obj:`samitorch.config.configurations.DatasetConfiguration`: An object containing dataset's properties.

        """
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                return [DeepNormalizeDatasetConfiguration(config["dataset"][i][element]) for i, element in
                        enumerate(["iSEG", "MRBrainS"])]
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class: Configuration):
        pass


class TrainingConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        """
        Parse a training configuration file.

        Args:
          path (str): Configuration YAML file path.

        Returns:
          :obj:`samitorch.config.configurations.DatasetConfiguration`: An object containing dataset's properties.

        """
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                metrics = [config["training"]["metrics"][0], config["training"]["metrics"][1]]
                config = DeepNormalizeTrainingConfiguration(config["training"])
                config.metrics = metrics
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass
