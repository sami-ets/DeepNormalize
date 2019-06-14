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

from samitorch.factories.factories import *
from samitorch.factories.enums import *
from samitorch.inputs.datasets import NiftiDataset
from samitorch.inputs.transformers import ToNumpyArray, RandomCrop3D
from samitorch.inputs.dataloaders import DataLoader
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset

from deepNormalize.factories.parsers import *


class Initializer(object):
    VALIDATION_SPLIT = 5
    NUM_WORKERS = 8

    def __init__(self, path: str):
        self._path = path

    def create_configs(self):
        datasets_configs = DeepNormalizeDatasetConfigurationParserFactory().parse(self._path)
        model_config = DeepNormalizeModelsParserFactory().parse(self._path)
        training_config = TrainingConfigurationParserFactory().parse(self._path)

        return datasets_configs, model_config, training_config

    def create_metrics(self, training_config):
        dice_metric = training_config.metrics[0]["dice"]
        accuracy_metric = training_config.metrics[1]["accuracy"]

        dice = MetricsFactory().create_metric(Metrics.Dice,
                                              num_classes=dice_metric["num_classes"],
                                              ignore_index=dice_metric["ignore_index"],
                                              average=dice_metric["average"],
                                              reduction=dice_metric["reduction"])
        accuracy = MetricsFactory().create_metric(Metrics.Accuracy, accuracy_metric["is_multilabel"])
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
        optimizer_preprocessor = OptimizerFactory().create_optimizer(training_config.optimizer["type"],
                                                                     models[0].parameters(),
                                                                     lr=training_config.optimizer["lr"])
        optimizer_segmenter = OptimizerFactory().create_optimizer(training_config.optimizer["type"],
                                                                  models[1].parameters(),
                                                                  lr=training_config.optimizer["lr"])
        optimizer_discriminator = OptimizerFactory().create_optimizer(training_config.optimizer["type"],
                                                                      models[2].parameters(),
                                                                      lr=training_config.optimizer["lr"])
        optimizers = [optimizer_preprocessor, optimizer_segmenter, optimizer_discriminator]
        return optimizers

    def create_dataset(self, dataset_config):
        dataset_iSEG = NiftiDataset(source_dir=dataset_config[0].path + "/Training/Source",
                                    target_dir=dataset_config[0].path + "/Training/Target",
                                    transform=Compose([ToNumpyArray(),
                                                       RandomCrop3D(dataset_config[0].training_patch_size)]))
        dataset_MRBrainS = NiftiDataset(source_dir=dataset_config[1].path + "/TrainingData/Source",
                                        target_dir=dataset_config[1].path + "/TrainingData/Target",
                                        transform=Compose([ToNumpyArray(),
                                                           RandomCrop3D(dataset_config[0].training_patch_size)]))

        dataset = ConcatDataset([dataset_iSEG, dataset_MRBrainS])
        return dataset

    def create_dataloader(self, dataset, batch_size):
        return DataLoader(dataset, shuffle=True, validation_split=self.VALIDATION_SPLIT, num_workers=self.NUM_WORKERS,
                          batch_size=batch_size)
