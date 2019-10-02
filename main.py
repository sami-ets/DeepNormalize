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
import os
import torch
import torch.backends.cudnn as cudnn
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.config.trainers import RunConfiguration
from kerosene.dataloaders.factories import DataloaderFactory
from kerosene.events import Event
from kerosene.events.handlers.checkpoints import ModelCheckpointIfBetter
from kerosene.events.handlers.console import PrintTrainingStatus
from kerosene.events.handlers.visdom import PlotAllModelStateVariables, PlotLR, PlotCustomVariables, PlotGradientFlow
from kerosene.loggers.visdom import PlotType, PlotFrequency
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger, VisdomData
from kerosene.training.trainers import ModelTrainerFactory
from deepNormalize.inputs.datasets import iSEGSegmentationFactory, MRBrainSSegmentationFactory
from samitorch.inputs.utils import sample_collate
from torch.utils.data import DataLoader

from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType, DatasetConfigurationParser
from deepNormalize.events.handlers.console import PrintTrainLoss
from deepNormalize.factories.customCriterionFactory import CustomCriterionFactory
from deepNormalize.factories.customModelFactory import CustomModelFactory
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
    run_config = RunConfiguration(args.use_amp, args.amp_opt_level, args.local_rank, args.num_workers)
    model_trainer_configs, training_config = YamlConfigurationParser.parse(args.config_file)
    dataset_config = DatasetConfigurationParser().parse(args.config_file)
    config_html = [training_config.to_html(), list(map(lambda config: config.to_html(), dataset_config)),
                   list(map(lambda config: config.to_html(), [model_trainer_configs]))]

    # Prepare the data.
    iSEG_train = None
    iSEG_valid = None
    MRBrainS_train = None
    MRBrainS_valid = None

    if "iSEG" in [dataset_config[i].dataset_name for i in range(len(dataset_config))]:
        iSEG_train, iSEG_valid = iSEGSegmentationFactory.create_train_test(source_dir=dataset_config[0].path,
                                                                           target_dir=dataset_config[0].path + "/label",
                                                                           modality=args.modality,
                                                                           dataset_id=ISEG_ID,
                                                                           test_size=dataset_config[0].validation_split)

    if "MRBrainS" in [dataset_config[i].dataset_name for i in range(len(dataset_config))]:
        MRBrainS_train, MRBrainS_valid = MRBrainSSegmentationFactory.create_train_test(
            source_dir=dataset_config[1 if len(dataset_config) == 2 else 0].path,
            target_dir=dataset_config[1 if len(dataset_config) == 2 else 0].path,
            modality=args.modality,
            dataset_id=MRBRAINS_ID,
            test_size=dataset_config[1 if len(dataset_config) == 2 else 0].validation_split)

    # Concat datasets.
    if len(dataset_config) == 2:
        training_dataset = torch.utils.data.ConcatDataset((iSEG_train, MRBrainS_train))
        validation_dataset = torch.utils.data.ConcatDataset((iSEG_valid, MRBrainS_valid))
    else:
        training_dataset = iSEG_train if iSEG_train is not None else MRBrainS_train
        validation_dataset = iSEG_valid if iSEG_valid is not None else MRBrainS_valid

    # Initialize the model trainers
    model_trainer_factory = ModelTrainerFactory(model_factory=CustomModelFactory(),
                                                criterion_factory=CustomCriterionFactory())
    model_trainers = list(map(lambda config: model_trainer_factory.create(config, run_config), [model_trainer_configs]))

    # Create loaders.
    train_loader, valid_loader = DataloaderFactory(training_dataset, validation_dataset).create(run_config,
                                                                                                training_config,
                                                                                                collate_fn=sample_collate)
    # Initialize the loggers.
    visdom_config = VisdomConfiguration.from_yml(args.config_file, "visdom")
    visdom_logger = VisdomLogger(visdom_config)

    # Train with the training strategy.
    if run_config.local_rank == 0:
        visdom_logger(VisdomData("Experiment", "Experiment Config", PlotType.TEXT_PLOT, PlotFrequency.EVERY_EPOCH, None,
                                 config_html))
        save_folder = "saves/" + os.path.basename(os.path.normpath(visdom_config.env))
        [os.makedirs("{}/{}".format(save_folder, model), exist_ok=True) for model in ["Segmenter"]]

        trainer = DeepNormalizeTrainer(training_config, model_trainers, train_loader, valid_loader, run_config) \
            .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
            .with_event_handler(PrintTrainLoss(every=25), Event.ON_BATCH_END) \
            .with_event_handler(PlotAllModelStateVariables(visdom_logger), Event.ON_EPOCH_END) \
            .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Input Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4, "opts": {"title": "Input Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Segmented Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4, "opts": {"title": "Segmented Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Segmentation Ground Truth Batch", PlotType.IMAGES_PLOT,
                                params={"nrow": 4, "opts": {"title": "Ground Truth Patches"}},
                                every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                                    params={
                                                        "opts": {"title": "Inputs Intensity Histogram", "nbins": 75}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                    params={"opts": {"title": "Batch data distribution",
                                                                     "legend": ["iSEG", "MRBrainS", "Fake Class"]}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                    params={"opts": {"title": "Mean Hausdorff Distance"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GPU {} Memory".format(run_config.local_rank), PlotType.LINE_PLOT,
                                params={"opts": {"title": "GPU {} Memory Usage".format(run_config.local_rank)}},
                                every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                    params={"opts": {"title": "Metric Table"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                    params={"opts": {"columnnames": ["Background", "CSF", "GM", "WM"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotGradientFlow(visdom_logger, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(ModelCheckpointIfBetter(save_folder), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)


    else:
        trainer = DeepNormalizeTrainer(training_config, model_trainers, train_loader, valid_loader, run_config) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GPU {} Memory".format(run_config.local_rank), PlotType.LINE_PLOT,
                                params={"opts": {"title": "GPU {} Memory Usage".format(run_config.local_rank)}},
                                every=1), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)
