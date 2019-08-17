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
import torch

from samitorch.optimizers.optimizers import OptimizerFactory
from samitorch.criterions.criterions import CriterionFactory
from samitorch.metrics.metric import MetricsFactory, Metric
from samitorch.models.unet3d import UNet3DModelFactory, UNetModel
from samitorch.models.resnet3d import ResNet3DModelFactory, ResNetModel
from samitorch.inputs.images import Modality
from samitorch.inputs.datasets import PatchDatasetFactory
from samitorch.inputs.utils import patch_collate
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
        pretraining_config = PreTrainingConfigurationParserFactory().parse(self._path)
        variable_config = VariableConfigurationParserFactory().parse(self._path)
        logger_config = LoggerConfigurationParserFactory().parse(self._path)
        visdom_config = VisdomConfigurationParserFactory().parse(self._path)
        return datasets_configs, model_config, training_config, pretraining_config, variable_config, logger_config, visdom_config

    def create_metrics(self, training_config):
        dice_metric = training_config.metrics["dice"]
        accuracy_metric = training_config.metrics["accuracy"]
        factory = MetricsFactory()
        dice = factory.create_metric(Metric.Dice,
                                     num_classes=dice_metric["num_classes"],
                                     ignore_index=dice_metric["ignore_index"],
                                     average=dice_metric["average"],
                                     reduction=dice_metric["reduction"])
        accuracy = factory.create_metric(Metric.Accuracy, is_multilabel=accuracy_metric["is_multilabel"])
        metrics = [dice, accuracy]
        return metrics

    def create_criterions(self, model_config):
        criterion_generator = CriterionFactory().create(model_config[0].criterion)
        criterion_segmenter = CriterionFactory().create(model_config[1].criterion)
        criterion_discriminator = CriterionFactory().create(model_config[2].criterion)
        criterions = [criterion_generator, criterion_segmenter, criterion_discriminator]
        return criterions

    def create_models(self, model_config):
        generator = UNet3DModelFactory().create_model(UNetModel.UNet3D, model_config[0].model)
        segmenter = UNet3DModelFactory().create_model(UNetModel.UNet3D, model_config[1].model)
        discriminator = ResNet3DModelFactory().create_model(ResNetModel.ResNet34, model_config[2].model)
        models = [generator, segmenter, discriminator]
        return models

    def create_optimizers(self, model_config, models):
        optimizer_generator = OptimizerFactory().create(
            model_config[0].optimizer.type,
            models[0].parameters(),
            lr=model_config[0].optimizer.lr)
        optimizer_segmenter = OptimizerFactory().create(model_config[1].optimizer.type,
                                                        models[1].parameters(),
                                                        lr=model_config[1].optimizer.lr)
        optimizer_discriminator = OptimizerFactory().create(
            model_config[2].optimizer.type, models[2].parameters(),
            lr=model_config[2].optimizer.lr)
        optimizers = [optimizer_generator, optimizer_segmenter, optimizer_discriminator]
        return optimizers

    def create_dataset(self, dataset_config):
        iSEG_train, iSEG_valid = PatchDatasetFactory.create_train_test(
            source_dir=dataset_config[0].path + "/Training/Source",
            target_dir=dataset_config[0].path + "/Training/Target",
            dataset_id=0,
            patch_size=dataset_config[0].training_patch_size,
            step=dataset_config[0].training_patch_step,
            modality=Modality.T1,
            test_size=0.2,
            keep_centered_on_foreground=True)

        MRBrainS_train, MRBrains_valid = PatchDatasetFactory.create_train_test(
            source_dir=dataset_config[1].path + "/TrainingData/Source",
            target_dir=dataset_config[1].path + "/TrainingData/Target",
            dataset_id=1,
            patch_size=dataset_config[1].training_patch_size,
            step=dataset_config[1].training_patch_step,
            test_size=0.2,
            keep_centered_on_foreground=True,
            modality=Modality.T1)
        return [iSEG_train, MRBrainS_train, iSEG_valid, MRBrains_valid]

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

    def create_dataloader(self, train_datasets, valid_datasets, batch_size, num_workers, is_distributed):
        if is_distributed:
            self._logger.info("Initializing distributed Dataloader.")

            dataloaders = list()

            for train_dataset, valid_dataset in zip(train_datasets, valid_datasets):
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

                dataloaders.append(torch.utils.data.DataLoader(dataset=train_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               num_workers=num_workers,
                                                               sampler=train_sampler,
                                                               collate_fn=patch_collate))
                dataloaders.append(torch.utils.data.DataLoader(dataset=valid_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               num_workers=num_workers,
                                                               sampler=valid_sampler,
                                                               collate_fn=patch_collate))
            return dataloaders

        else:
            self._logger.info("Initializing Dataloader.")

            dataloaders = list()

            for train_dataset, valid_dataset in zip(train_datasets, valid_datasets):
                dataloaders.append(torch.utils.data.DataLoader(dataset=train_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               num_workers=num_workers,
                                                               collate_fn=patch_collate))
                dataloaders.append(torch.utils.data.DataLoader(dataset=valid_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               num_workers=num_workers,
                                                               collate_fn=patch_collate))
            return dataloaders
