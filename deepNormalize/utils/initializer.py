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
import os
import copy
from datetime import datetime

from samitorch.factories.factories import *
from samitorch.factories.enums import *
from samitorch.inputs.datasets import NiftiPatchDataset
from samitorch.inputs.transformers import ToNDTensor
from samitorch.inputs.dataloaders import DataLoader
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset
from deepNormalize.factories.parsers import *
from deepNormalize.utils.utils import split_dataset


class Initializer(object):

    def __init__(self, path: str):
        self._path = path
        self._logger = logging.getLogger("Initializer")
        self._logger.setLevel(logging.INFO)
        # Logging to console
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(process)d] [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        self._logger.propagate = False

    def create_configs(self):
        datasets_configs = DeepNormalizeDatasetConfigurationParserFactory().parse(self._path)
        model_config = DeepNormalizeModelsParserFactory().parse(self._path)
        training_config = TrainingConfigurationParserFactory().parse(self._path)
        variable_config = VariableConfigurationParserFactory().parse(self._path)
        logger_config = LoggerConfigurationParserFactory().parse(self._path)

        return datasets_configs, model_config, training_config, variable_config, logger_config

    def create_metrics(self, training_config):
        dice_metric = training_config.metrics["dice"]
        accuracy_metric = training_config.metrics["accuracy"]
        factory = MetricsFactory()

        dice = factory.create_metric(Metrics.Dice,
                                              num_classes=dice_metric["num_classes"],
                                              ignore_index=dice_metric["ignore_index"],
                                              average=dice_metric["average"],
                                              reduction=dice_metric["reduction"])
        accuracy = factory.create_metric(Metrics.TopKCategoricalAccuracy, k=2)
        metrics = [dice, accuracy]
        return metrics

    def create_criterions(self, training_config):
        criterion_segmenter = CriterionFactory().create_criterion(training_config.criterions[0])
        criterion_discriminator = CriterionFactory().create_criterion(training_config.criterions[1])
        criterions = [criterion_segmenter, criterion_discriminator]
        return criterions

    def create_models(self, model_config):
        preprocessor = ModelFactory().create_model(UNetModels.UNet3D, model_config[0])
        segmenter = ModelFactory().create_model(UNetModels.UNet3D, model_config[1])
        discriminator = ModelFactory().create_model(ResNetModels.ResNet34, model_config[2])
        models = [preprocessor, segmenter, discriminator]
        return models

    def create_optimizers(self, training_config, models):
        optimizer_preprocessor = OptimizerFactory().create_optimizer(
            training_config.optimizers["preprocessor"]["type"],
            models[0].parameters(),
            lr=training_config.optimizers["preprocessor"]["lr"])
        optimizer_segmenter = OptimizerFactory().create_optimizer(training_config.optimizers["segmenter"]["type"],
                                                                  models[1].parameters(),
                                                                  lr=training_config.optimizers["segmenter"]["lr"])
        optimizer_discriminator = OptimizerFactory().create_optimizer(
            training_config.optimizers["discriminator"]["type"],
            models[2].parameters(),
            lr=training_config.optimizers["discriminator"]["lr"])
        optimizers = [optimizer_preprocessor, optimizer_segmenter, optimizer_discriminator]
        return optimizers

    def create_dataset(self, dataset_config):
        dataset_iSEG = NiftiPatchDataset(source_dir=dataset_config[0].path + "/Training/Source",
                                         target_dir=dataset_config[0].path + "/Training/Target",
                                         transform=Compose([ToNDTensor()]),
                                         patch_shape=dataset_config[0].training_patch_size,
                                         step=dataset_config[0].training_patch_step)

        dataset_MRBrainS = NiftiPatchDataset(source_dir=dataset_config[1].path + "/TrainingData/Source",
                                             target_dir=dataset_config[1].path + "/TrainingData/Target",
                                             transform=Compose([ToNDTensor()]),
                                             patch_shape=dataset_config[1].training_patch_size,
                                             step=dataset_config[1].training_patch_step)
        return [dataset_iSEG, dataset_MRBrainS]

    def init_process_group(self, running_config):
        if 'WORLD_SIZE' in os.environ:
            running_config.world_size = int(os.environ['WORLD_SIZE']) > 1

        gpu = 0
        running_config.world_size = 1

        if running_config.is_distributed:
            gpu = running_config.local_rank
            torch.cuda.set_device(gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            running_config.world_size = torch.distributed.get_world_size()
        else:
            torch.cuda.set_device(gpu)

        self._logger.info("Running in {} mode with WORLD_SIZE of {}.".format(
            "distributed" if running_config.is_distributed else "non-distributed",
            running_config.world_size))

    def create_dataloader(self, datasets, batch_size, num_workers, dataset_configs, is_distributed):
        if is_distributed:
            self._logger.info("Initializing distributed Dataloader.")

            dataloaders = list()

            for dataset, config in zip(datasets, dataset_configs):
                train_dataset, valid_dataset = split_dataset(dataset, config.validation_split)
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

                dataloaders.append(DataLoader(dataset, shuffle=True, validation_split=config.validation_split,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              samplers=(train_sampler, valid_sampler)))
            return dataloaders

        else:
            self._logger.info("Initializing Dataloader.")

            dataloaders = list()

            for dataset, config in zip(datasets, dataset_configs):
                dataloaders.append(DataLoader(dataset, shuffle=True, validation_split=config.validation_split,
                                              num_workers=num_workers,
                                              batch_size=batch_size))
            return dataloaders

    def create_log_folder(self, config):
        config_ = copy.copy(config)
        log_path = os.path.join(config.path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(log_path)
        config_.path = log_path
        return config_
