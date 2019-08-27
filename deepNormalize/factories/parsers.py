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

from samitorch.configs.configurations import Configuration
from samitorch.parsers.parsers import AbstractConfigurationParserFactory
from samitorch.configs.configurations import UNetModelConfiguration, ResNetModelConfiguration

from deepNormalize.config.configurations import DeepNormalizeDatasetConfiguration, DeepNormalizeTrainingConfiguration, \
    VariableConfiguration, LoggerConfiguration, VisdomConfiguration, PretrainingConfiguration, OptimizerConfiguration, \
    SchedulerConfiguration, ModelConfiguration


class DeepNormalizeModelsParserFactory(AbstractConfigurationParserFactory):

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                generator = UNetModelConfiguration(config["models"]["Generator"])
                optimizer_g = OptimizerConfiguration(config["models"]["Generator"]["optimizer"])
                scheduler_g = SchedulerConfiguration(config["models"]["Generator"]["scheduler"])
                criterion_g = config["models"]["Generator"]["criterion"]
                segmenter = UNetModelConfiguration(config["models"]["Segmenter"])
                optimizer_s = OptimizerConfiguration(config["models"]["Segmenter"]["optimizer"])
                scheduler_s = SchedulerConfiguration(config["models"]["Segmenter"]["scheduler"])
                criterion_s = config["models"]["Segmenter"]["criterion"]
                discriminator = ResNetModelConfiguration(config["models"]["Discriminator"])
                optimizer_d = OptimizerConfiguration(config["models"]["Discriminator"]["optimizer"])
                scheduler_d = SchedulerConfiguration(config["models"]["Discriminator"]["scheduler"])
                criterion_d = config["models"]["Discriminator"]["criterion"]

                generator_config = ModelConfiguration(config={"model": generator,
                                                              "optimizer": optimizer_g,
                                                              "scheduler": scheduler_g,
                                                              "criterion": criterion_g})
                segmenter_config = ModelConfiguration(config={"model": segmenter,
                                                              "optimizer": optimizer_s,
                                                              "scheduler": scheduler_s,
                                                              "criterion": criterion_s})
                discriminator_config = ModelConfiguration(config={"model": discriminator,
                                                                  "optimizer": optimizer_d,
                                                                  "scheduler": scheduler_d,
                                                                  "criterion": criterion_d})

                return [generator_config, segmenter_config, discriminator_config]
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
                return [DeepNormalizeDatasetConfiguration(config["dataset"][element]) for element in
                        ["iSEG", "MRBrainS"]]
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
                config = DeepNormalizeTrainingConfiguration(config["training"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class PreTrainingConfigurationParserFactory(AbstractConfigurationParserFactory):

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
                config = PretrainingConfiguration(config["pretraining"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class VariableConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                config = VariableConfiguration(config["variables"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class LoggerConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                config = LoggerConfiguration(config["logger"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass


class VisdomConfigurationParserFactory(AbstractConfigurationParserFactory):

    def __init__(self):
        pass

    def parse(self, path: str):
        with open(path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                config = VisdomConfiguration(config["visdom"])
                return config
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    def register(self, model_type: str, configuration_class):
        pass
