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
from kerosene.dataloaders.dataloaders import DataloaderFactory
from kerosene.events import Event
from kerosene.events.handlers.checkpoints import ModelCheckpointIfBetter
from kerosene.events.handlers.console import PrintTrainingStatus, PrintModelTrainersStatus
from kerosene.events.handlers.visdom import PlotAllModelStateVariables, PlotLR, PlotCustomVariables, PlotGradientFlow
from kerosene.loggers.visdom import PlotType, PlotFrequency
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger, VisdomData
from kerosene.training.trainers import ModelTrainerFactory
from samitorch.inputs.utils import sample_collate
from torch.utils.data import DataLoader

from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType, DatasetConfigurationParser
from deepNormalize.factories.customCriterionFactory import CustomCriterionFactory
from deepNormalize.factories.customModelFactory import CustomModelFactory
from deepNormalize.inputs.datasets import iSEGSegmentationFactory, MRBrainSSegmentationFactory
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
                   list(map(lambda config: config.to_html(), model_trainer_configs))]

    # Prepare the data.
    iSEG_train = None
    iSEG_valid = None
    iSEG_test = None
    iSEG_CSV = None
    MRBrainS_train = None
    MRBrainS_valid = None
    MRBrainS_test = None
    MRBrainS_CSV = None

    if "iSEG" in [dataset_config[i].dataset_name for i in range(len(dataset_config))]:
        iSEG_train, iSEG_valid, iSEG_test, iSEG_CSV = iSEGSegmentationFactory.create_train_valid_test(
            source_dir=dataset_config[0].path,
            target_dir=dataset_config[0].path + "/label",
            modality=args.modality,
            dataset_id=ISEG_ID,
            test_size=dataset_config[0].validation_split)

    if "MRBrainS" in [dataset_config[i].dataset_name for i in range(len(dataset_config))]:
        MRBrainS_train, MRBrainS_valid, MRBrainS_test, MRBrainS_CSV = MRBrainSSegmentationFactory.create_train_valid_test(
            source_dir=dataset_config[1 if len(dataset_config) == 2 else 0].path,
            target_dir=dataset_config[1 if len(dataset_config) == 2 else 0].path,
            modality=args.modality,
            dataset_id=MRBRAINS_ID,
            test_size=dataset_config[1 if len(dataset_config) == 2 else 0].validation_split)

    # Concat datasets.
    if len(dataset_config) == 2:
        training_dataset = torch.utils.data.ConcatDataset((iSEG_train, MRBrainS_train))
        validation_dataset = torch.utils.data.ConcatDataset((iSEG_valid, MRBrainS_valid))
        test_dataset = torch.utils.data.ConcatDataset((iSEG_test, MRBrainS_test))
    else:
        training_dataset = iSEG_train if iSEG_train is not None else MRBrainS_train
        validation_dataset = iSEG_valid if iSEG_valid is not None else MRBrainS_valid
        test_dataset = iSEG_test if iSEG_test is not None else MRBrainS_test

    # Initialize the model trainers
    model_trainer_factory = ModelTrainerFactory(model_factory=CustomModelFactory(),
                                                criterion_factory=CustomCriterionFactory(run_config))
    model_trainers = list(map(lambda config: model_trainer_factory.create(config, run_config), model_trainer_configs))

    # Create loaders.
    train_loader, valid_loader, test_loader = DataloaderFactory(training_dataset, validation_dataset,
                                                                test_dataset).create(run_config,
                                                                                     training_config,
                                                                                     collate_fn=sample_collate)

    # Initialize the loggers.
    visdom_config = VisdomConfiguration.from_yml(args.config_file, "visdom")
    visdom_logger = VisdomLogger(visdom_config)

    # Train with the training strategy.
    if run_config.local_rank == 0:
        visdom_logger(VisdomData("Experiment", "Experiment Config", PlotType.TEXT_PLOT, PlotFrequency.EVERY_EPOCH, None,
                                 config_html))
        visdom_logger(VisdomData("Experiment", "Patch count", PlotType.BAR_PLOT, PlotFrequency.EVERY_EPOCH,
                                 x=[len(iSEG_train) if iSEG_train is not None else 0,
                                    len(MRBrainS_train) if MRBrainS_train is not None else 0],
                                 y=["iSEG", "MRBrainS"], params={"opts": {"title": "Patch count"}}))
        visdom_logger(VisdomData("Experiment", "Center Voxel Class Count", PlotType.BAR_PLOT, PlotFrequency.EVERY_EPOCH,
                                 x=[iSEG_CSV.groupby('center_class').count().get_values().flatten() if iSEG_train is not None else 0,
                                    MRBrainS_CSV.groupby('center_class').count().get_values().flatten() if MRBrainS_train is not None else 0],
                                 y=["iSEG", "MRBrainS"], params={"opts": {"title": "Center Voxel Class Count", "stacked": True, "legend":["CSF", "GM", "WM"]}}))

        save_folder = "saves/" + os.path.basename(os.path.normpath(visdom_config.env))
        [os.makedirs("{}/{}".format(save_folder, model), exist_ok=True) for model in
         ["Discriminator", "Generator", "Segmenter"]]

        trainer = DeepNormalizeTrainer(training_config, model_trainers, train_loader, valid_loader, test_loader,
                                       run_config) \
            .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
            .with_event_handler(PrintModelTrainersStatus(every=25), Event.ON_BATCH_END) \
            .with_event_handler(PlotAllModelStateVariables(visdom_logger), Event.ON_EPOCH_END) \
            .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Generated Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4,
                                                            "opts": {"store_history": True,
                                                                     "title": "Generated Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Input Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4,
                                                            "opts": {"store_history": True,
                                                                     "title": "Input Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Segmented Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4,
                                                            "opts": {"store_history": True,
                                                                     "title": "Segmented Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Segmentation Ground Truth Batch", PlotType.IMAGES_PLOT,
                                params={"nrow": 4,
                                        "opts": {"store_history": True,
                                                 "title": "Ground Truth Patches"}}, every=100),
            Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "Generated Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "Background Generated Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "CSF Generated Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "GM Generated Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "WM Generated Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={
                                    "opts": {"title": "Inputs Intensity Histogram",
                                             "numbins": 128}},
                                every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "Background Input Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "CSF Input Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "GM Input Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                params={"opts": {"title": "WM Input Intensity Histogram",
                                                 "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot", PlotType.PIE_PLOT,
                                                    params={"opts": {"title": "Classification hit per classes",
                                                                     "legend": ["iSEG", "MRBrainS", "Fake Class"]}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                    params={"opts": {"title": "Batch data distribution",
                                                                     "legend": ["iSEG", "MRBrainS", "Fake Class"]}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "D(G(X)) | X", PlotType.LINE_PLOT,
                                                    params={"name": "training", "opts": {"title": "Loss D(G(X)) | X"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "D(G(X)) | X Valid", PlotType.LINE_PLOT,
                                                    params={"name": "validation",
                                                            "opts": {"title": "Loss D(G(X)) | X Validation"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "D(G(X)) | X Test", PlotType.LINE_PLOT,
                                                    params={"name": "validation",
                                                            "opts": {"title": "Loss D(G(X)) | X Test"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Jensen-Shannon Divergence Inputs", PlotType.LINE_PLOT,
                                params={"opts": {"title": "Jensen-Shannon Divergence Inputs"}},
                                every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Jensen-Shannon Divergence Generated", PlotType.LINE_PLOT,
                                params={"opts": {"title": "Jensen-Shannon Divergence Generated"}},
                                every=1), Event.ON_EPOCH_END) \
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
            .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                    params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                    params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "MRBrainS Confusion Matrix"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                    params={"opts": {"title": "Runtime"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotGradientFlow(visdom_logger, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(ModelCheckpointIfBetter(save_folder), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)


    else:
        trainer = DeepNormalizeTrainer(training_config, model_trainers, train_loader, valid_loader, test_loader,
                                       run_config) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GPU {} Memory".format(run_config.local_rank), PlotType.LINE_PLOT,
                                params={"opts": {"title": "GPU {} Memory Usage".format(run_config.local_rank)}},
                                every=1), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)
