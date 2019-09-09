# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import multiprocessing

import torch
import torch.backends.cudnn as cudnn
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.config.trainers import RunConfiguration
from kerosene.dataloaders.factories import DataloaderFactory
from kerosene.events import Event
from kerosene.events.handlers.console import ConsoleLogger
from kerosene.events.handlers.checkpoints import ModelCheckpointIfBetter
from kerosene.events.handlers.visdom.config import VisdomConfiguration
from kerosene.events.handlers.visdom.data import VisdomData, PlotFrequency
from kerosene.events.handlers.visdom.visdom import VisdomLogger
from kerosene.events.preprocessors.visdom import PlotAllModelStateVariables, PlotLR, PlotCustomVariables, PlotType
from kerosene.training.trainers import ModelTrainerFactory
from samitorch.inputs.datasets import PatchDatasetFactory
from samitorch.inputs.utils import patch_collate
from torch.utils.data import DataLoader

from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType, DatasetConfigurationParser
from deepNormalize.events.preprocessor.console_preprocessor import PrintTrainLoss
from deepNormalize.factories.customModelFactory import CustomModelFactory
from deepNormalize.factories.customCriterionFactory import CustomCriterionFactory
from deepNormalize.training.trainer import DeepNormalizeTrainer

ISEG_ID = 0
MRBRAINS_ID = 1

cudnn.benchmark = True
cudnn.enabled = True

if __name__ == '__main__':
    # Basic settings
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(multiprocessing.cpu_count())
    args = ArgsParserFactory.create_parser(ArgsParserType.MODEL_TRAINING).parse_args()

    # Create configurations.
    run_config = RunConfiguration(args.use_amp, args.amp_opt_level, args.local_rank)
    model_trainer_configs, training_config = YamlConfigurationParser.parse(args.config_file)
    dataset_config = DatasetConfigurationParser().parse(args.config_file)
    config_html = [training_config.to_html(), list(map(lambda config: config.to_html(), model_trainer_configs))]

    # Prepare the data.
    iSEG_train, iSEG_valid = PatchDatasetFactory.create_train_test(
        source_dir=dataset_config[0].path + "/Training/Source",
        target_dir=dataset_config[0].path + "/Training/Target",
        dataset_id=ISEG_ID,
        patch_size=dataset_config[0].training_patch_size,
        step=dataset_config[0].training_patch_step,
        modality=args.modality,
        test_size=dataset_config[0].validation_split,
        keep_centered_on_foreground=True)

    MRBrainS_train, MRBrains_valid = PatchDatasetFactory.create_train_test(
        source_dir=dataset_config[1].path + "/TrainingData/Source",
        target_dir=dataset_config[1].path + "/TrainingData/Target",
        dataset_id=MRBRAINS_ID,
        patch_size=dataset_config[1].training_patch_size,
        step=dataset_config[1].training_patch_step,
        test_size=dataset_config[1].validation_split,
        keep_centered_on_foreground=True,
        modality=args.modality)

    # Concat datasets.
    training_datasets = torch.utils.data.ConcatDataset((iSEG_train, MRBrainS_train))
    validation_datasets = torch.utils.data.ConcatDataset((iSEG_valid, MRBrains_valid))

    # Initialize the model trainers
    model_trainer_factory = ModelTrainerFactory(model_factory=CustomModelFactory(),
                                                criterion_factory=CustomCriterionFactory())
    model_trainers = list(map(lambda config: model_trainer_factory.create(config, run_config), model_trainer_configs))

    # Create loaders.
    train_loader, valid_loader = DataloaderFactory(training_datasets, validation_datasets).create(run_config,
                                                                                                  training_config,
                                                                                                  collate_fn=patch_collate)

    # Initialize the loggers.
    if run_config.local_rank == 0:
        visdom_logger = VisdomLogger(VisdomConfiguration.from_yml(args.config_file, "visdom"))
        console_logger = ConsoleLogger()
        visdom_logger(VisdomData("Experiment", "Experiment Config", PlotType.TEXT_PLOT, PlotFrequency.EVERY_EPOCH, None,
                                 config_html))

    # Train with the training strategy.
    if run_config.local_rank == 0:
        trainer = DeepNormalizeTrainer(training_config, model_trainers, train_loader, valid_loader, run_config,
                                       dataset_config) \
            .with_event_handler(console_logger, Event.ON_BATCH_END) \
            .with_event_handler(console_logger, Event.ON_BATCH_END, PrintTrainLoss()) \
            .with_event_handler(visdom_logger, Event.ON_EPOCH_END, PlotAllModelStateVariables()) \
            .with_event_handler(visdom_logger, Event.ON_EPOCH_END, PlotLR()) \
            .with_event_handler(visdom_logger, Event.ON_100_TRAIN_STEPS,
                                PlotCustomVariables("Generated Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4, "opts": {"title": "Generated Patches"}})) \
            .with_event_handler(visdom_logger, Event.ON_100_TRAIN_STEPS,
                                PlotCustomVariables("Input Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4, "opts": {"title": "Input Patches"}})) \
            .with_event_handler(visdom_logger, Event.ON_100_TRAIN_STEPS,
                                PlotCustomVariables("Segmented Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4, "opts": {"title": "Segmented Patches"}})) \
            .with_event_handler(visdom_logger, Event.ON_TRAIN_BATCH_END,
                                PlotCustomVariables("Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                                    params={
                                                        "opts": {"title": "Generated Intensity Histogram",
                                                                 "nbins": 50}})) \
            .with_event_handler(visdom_logger, Event.ON_TRAIN_BATCH_END,
                                PlotCustomVariables("Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                                    params={
                                                        "opts": {"title": "Inputs Intensity Histogram", "nbins": 50}})) \
            .with_event_handler(ModelCheckpointIfBetter("saves/"), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)
    else:
        trainer = DeepNormalizeTrainer(training_config, model_trainers, train_loader, valid_loader, run_config,
                                       dataset_config) \
            .train(training_config.nb_epochs)
