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

import logging
from argparse import ArgumentParser

import yaml
from samitorch.configs.configurations import Configuration
from samitorch.inputs.images import Modality, ImageType

from deepNormalize.config.configurations import DatasetConfiguration, DeepNormalizeTrainingConfiguration, \
    VariableConfiguration, LoggerConfiguration, VisdomConfiguration, PretrainingConfiguration


class DatasetConfigurationParser(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def parse(path: str):
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
                dataset_configs = list(
                    map(lambda dataset_name: DatasetConfiguration.from_dict(dataset_name,
                                                                            config["dataset"][dataset_name]),
                        config["dataset"]))
                return dataset_configs
            except yaml.YAMLError as e:
                logging.error(
                    "Unable to read the config file: {} with error {}".format(path, e))

    @staticmethod
    def parse_section(config_file_path, yml_tag):
        with open(config_file_path, 'r') as config_file:
            try:
                config = yaml.load(config_file, Loader=yaml.FullLoader)

                return config[yml_tag]
            except yaml.YAMLError as e:
                DatasetConfigurationParser.LOGGER.warning(
                    "Unable to read the training config file: {} with error {}".format(config_file_path, e))

    def register(self, model_type: str, configuration_class: Configuration):
        pass


class TrainingConfigurationParserFactory(object):

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


class PreTrainingConfigurationParserFactory(object):

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


class VariableConfigurationParserFactory(object):

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


class LoggerConfigurationParserFactory(object):

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


class VisdomConfigurationParserFactory(object):

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


class ArgsParserType(object):
    MODEL_TRAINING = "training"
    BRAIN_EXTRACTION = "brain_extraction"
    PRE_PROCESSING_PIPELINE = "pre_processing"
    PRE_TRAINED = "pre_trained"


class ArgsParserFactory(object):

    @staticmethod
    def create_parser(parser_type):
        parser = ArgumentParser(description='DeepNormalize Training')
        parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
        parser.add_argument("--amp-opt-level", dest="amp_opt_level", type=str, default="O1",
                            help="O0 - FP32 training, O1 - Mixed Precision (recommended), O2 - Almost FP16 Mixed Precision, O3 - FP16 Training.")
        parser.add_argument("--num-workers", dest="num_workers", default=8, type=int,
                            help="Number of data loading workers for each dataloader object (default: 4).")
        parser.add_argument("--local_rank", dest="local_rank", default=0, type=int, help="The local_rank of the GPU.")

        if parser_type is ArgsParserType.MODEL_TRAINING:
            parser.add_argument("--modality", dest="modality", default="T1", type=str,
                                help="The modality to be used (default: T1).")
            parser.add_argument("--config-file", dest="config_file", required=True)
        elif parser_type is ArgsParserType.BRAIN_EXTRACTION:
            parser.add_argument("--root_dir", dest="root_dir", required=True)
            parser.add_argument("--modality", dest="modality", required=True, choices=Modality.ALL)
            parser.add_argument("--image_type", dest="image_type", required=True, choices=ImageType.ALL)
        elif parser_type is ArgsParserType.PRE_PROCESSING_PIPELINE:
            parser.add_argument("--root_dir", dest="root_dir", required=True)
            parser.add_argument("--output_dir", dest="output_dir", required=True)
            parser.add_argument("--input_modality", dest="input_modality", required=True, choices=Modality.ALL)
            parser.add_argument("--output_modality", dest="output_modality", required=False, choices=Modality.ALL)
        elif parser_type is ArgsParserType.PRE_TRAINED:
            parser.add_argument("--modality", dest="modality", required=True, choices=Modality.ALL)
            parser.add_argument("--config_file", dest="config_file", required=True)
        return parser
