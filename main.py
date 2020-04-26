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

import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
from kerosene.configs.configs import RunConfiguration, DatasetConfiguration
from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.events import MonitorMode
from kerosene.events.handlers.checkpoints import Checkpoint
from kerosene.events.handlers.console import PrintTrainingStatus, PrintMonitors
from kerosene.events.handlers.visdom import PlotMonitors, PlotLR, PlotCustomVariables, PlotAvgGradientPerLayer
from kerosene.loggers.visdom import PlotType, PlotFrequency
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger, VisdomData
from kerosene.training.events import Event
from kerosene.training.trainers import ModelTrainerFactory
from kerosene.utils.devices import on_multiple_gpus
from samitorch.inputs.augmentation.strategies import AugmentInput
from samitorch.inputs.augmentation.transformers import AddNoise, AddBiasField, ShiftHistogram
from samitorch.inputs.utils import augmented_sample_collate
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType
from deepNormalize.events.handlers.handlers import PlotGPUMemory, PlotCustomLinePlotWithLegend, PlotCustomLoss
from deepNormalize.factories.customCriterionFactory import CustomCriterionFactory
from deepNormalize.factories.customModelFactory import CustomModelFactory
from deepNormalize.inputs.datasets import iSEGSegmentationFactory, MRBrainSSegmentationFactory, ABIDESegmentationFactory
from deepNormalize.training.trainer import DeepNormalizeTrainer
from deepNormalize.utils.constants import *
from deepNormalize.utils.image_slicer import ImageReconstructor

cudnn.benchmark = True
cudnn.enabled = True

np.random.seed(42)
random.seed(42)

if __name__ == '__main__':
    # Basic settings
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(multiprocessing.cpu_count())
    args = ArgsParserFactory.create_parser(ArgsParserType.MODEL_TRAINING).parse_args()

    # Create configurations.
    run_config = RunConfiguration(use_amp=args.use_amp, local_rank=args.local_rank, amp_opt_level=args.amp_opt_level)
    model_trainer_configs, training_config = YamlConfigurationParser.parse(args.config_file)
    dataset_configs = YamlConfigurationParser.parse_section(args.config_file, "dataset")
    dataset_configs = {k: DatasetConfiguration(v) for k, v, in dataset_configs.items()}
    config_html = [training_config.to_html(), list(map(lambda config: config.to_html(), dataset_configs.values())),
                   list(map(lambda config: config.to_html(), model_trainer_configs))]

    # Prepare the data.
    train_datasets = list()
    valid_datasets = list()
    test_datasets = list()
    reconstruction_datasets = list()
    normalized_reconstructors = list()
    segmentation_reconstructors = list()
    input_reconstructors = list()
    gt_reconstructors = list()
    augmented_input_reconstructors = list()

    iSEG_train = None
    iSEG_CSV = None
    MRBrainS_train = None
    MRBrainS_CSV = None
    ABIDE_train = None
    ABIDE_CSV = None

    iSEG_augmentation_strategy = None
    MRBrainS_augmentation_strategy = None
    ABIDE_augmentation_strategy = None

    # Initialize the model trainers
    model_trainer_factory = ModelTrainerFactory(model_factory=CustomModelFactory(),
                                                criterion_factory=CustomCriterionFactory(run_config))
    model_trainers = model_trainer_factory.create(model_trainer_configs)

    # Create datasets
    if dataset_configs.get("iSEG", None) is not None:
        if dataset_configs["iSEG"].hist_shift_augmentation:
            iSEG_augmentation_strategy = AugmentInput(
                Compose([ShiftHistogram(exec_probability=0.15, min_lambda=-5, max_lambda=5)]))
        elif training_config.data_augmentation:
            iSEG_augmentation_strategy = AugmentInput(Compose([AddNoise(exec_probability=1.0, noise_type="rician"),
                                                               AddBiasField(exec_probability=1.0, alpha=0.001)]))

        iSEG_train, iSEG_valid, iSEG_test, iSEG_reconstruction, iSEG_CSV = iSEGSegmentationFactory.create_train_valid_test(
            source_dir=dataset_configs["iSEG"].path,
            modalities=dataset_configs["iSEG"].modalities,
            dataset_id=ISEG_ID,
            test_size=dataset_configs["iSEG"].validation_split,
            max_subjects=dataset_configs["iSEG"].max_subjects,
            max_num_patches=dataset_configs["iSEG"].max_num_patches,
            augmentation_strategy=iSEG_augmentation_strategy)
        train_datasets.append(iSEG_train)
        valid_datasets.append(iSEG_valid)
        test_datasets.append(iSEG_test)
        reconstruction_datasets.append(iSEG_reconstruction)
        normalized_reconstructors.append(ImageReconstructor(dataset_configs["iSEG"].reconstruction_size,
                                                            dataset_configs['iSEG'].patch_size,
                                                            dataset_configs["iSEG"].step,
                                                            [model_trainers[GENERATOR]],
                                                            normalize=True))
        segmentation_reconstructors.append(
            ImageReconstructor(dataset_configs["iSEG"].reconstruction_size,
                               dataset_configs['iSEG'].patch_size,
                               dataset_configs["iSEG"].step,
                               [model_trainers[GENERATOR],
                                model_trainers[SEGMENTER]],
                               segment=True))
        input_reconstructors.append(ImageReconstructor(dataset_configs["iSEG"].reconstruction_size,
                                                       dataset_configs['iSEG'].patch_size,
                                                       dataset_configs["iSEG"].step))

        gt_reconstructors.append(ImageReconstructor(dataset_configs["iSEG"].reconstruction_size,
                                                    dataset_configs['iSEG'].patch_size,
                                                    dataset_configs["iSEG"].step))

    if dataset_configs.get("MRBrainS", None) is not None:
        if dataset_configs["MRBrainS"].hist_shift_augmentation:
            if training_config.data_augmentation:
                MRBRainS_augmentation_strategy = AugmentInput(
                    Compose([AddNoise(exec_probability=1.0, noise_type="rician"),
                             AddBiasField(exec_probability=1.0, alpha=0.001),
                             ShiftHistogram(exec_probability=0.50, min_lambda=-5,
                                            max_lambda=5)]))
            else:
                MRBRainS_augmentation_strategy = AugmentInput(
                    Compose([ShiftHistogram(exec_probability=0.15, min_lambda=-5, max_lambda=5)]))
        elif training_config.data_augmentation:
            MRBRainS_augmentation_strategy = AugmentInput(Compose([AddNoise(exec_probability=1.0, noise_type="rician"),
                                                                   AddBiasField(exec_probability=1.0, alpha=0.001)]))
        MRBrainS_train, MRBrainS_valid, MRBrainS_test, MRBrainS_reconstruction, MRBrainS_CSV = MRBrainSSegmentationFactory.create_train_valid_test(
            source_dir=dataset_configs["MRBrainS"].path,
            modalities=dataset_configs["MRBrainS"].modalities,
            dataset_id=MRBRAINS_ID,
            test_size=dataset_configs["MRBrainS"].validation_split,
            max_subjects=dataset_configs["MRBrainS"].max_subjects,
            max_num_patches=dataset_configs["MRBrainS"].max_num_patches,
            augmentation_strategy=MRBrainS_augmentation_strategy)
        train_datasets.append(MRBrainS_train)
        valid_datasets.append(MRBrainS_valid)
        test_datasets.append(MRBrainS_test)
        reconstruction_datasets.append(MRBrainS_reconstruction)
        normalized_reconstructors.append(
            ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                               dataset_configs['MRBrainS'].patch_size,
                               dataset_configs["MRBrainS"].step,
                               [model_trainers[GENERATOR]], normalize=True))
        segmentation_reconstructors.append(
            ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                               dataset_configs['MRBrainS'].patch_size,
                               dataset_configs["MRBrainS"].step,
                               [model_trainers[GENERATOR],
                                model_trainers[SEGMENTER]], segment=True))
        input_reconstructors.append(ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                                                       dataset_configs['MRBrainS'].patch_size,
                                                       dataset_configs["MRBrainS"].step))
        gt_reconstructors.append(ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                                                    dataset_configs['MRBrainS'].patch_size,
                                                    dataset_configs["MRBrainS"].step))


    if dataset_configs.get("ABIDE", None) is not None:
        if dataset_configs["ABIDE"].hist_shift_augmentation:
            if training_config.data_augmentation:
                ABIDE_augmentation_strategy = AugmentInput(Compose([AddNoise(exec_probability=1.0, noise_type="rician"),
                                                                    AddBiasField(exec_probability=1.0, alpha=0.001),
                                                                    ShiftHistogram(exec_probability=0.05, min_lambda=-5,
                                                                                   max_lambda=5)]))
            else:
                ABIDE_augmentation_strategy = AugmentInput(
                    Compose([ShiftHistogram(exec_probability=0.15, min_lambda=-5, max_lambda=5)]))
        elif training_config.data_augmentation:
            ABIDE_augmentation_strategy = AugmentInput(Compose([AddNoise(exec_probability=1.0, noise_type="rician"),
                                                                AddBiasField(exec_probability=1.0, alpha=0.001)]))
        ABIDE_train, ABIDE_valid, ABIDE_test, ABIDE_reconstruction, ABIDE_CSV = ABIDESegmentationFactory.create_train_valid_test(
            source_dir=dataset_configs["ABIDE"].path,
            modalities=dataset_configs["ABIDE"].modalities,
            dataset_id=ABIDE_ID,
            sites=dataset_configs["ABIDE"].sites,
            max_subjects=dataset_configs["ABIDE"].max_subjects,
            test_size=dataset_configs["ABIDE"].validation_split,
            max_num_patches=dataset_configs["ABIDE"].max_num_patches,
            augmentation_strategy=ABIDE_augmentation_strategy)
        train_datasets.append(ABIDE_train)
        valid_datasets.append(ABIDE_valid)
        test_datasets.append(ABIDE_test)
        reconstruction_datasets.append(ABIDE_reconstruction)
        normalized_reconstructors.append(ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size,
                                                            dataset_configs['ABIDE'].patch_size,
                                                            dataset_configs["ABIDE"].step,
                                                            [model_trainers[GENERATOR]], normalize=True))
        segmentation_reconstructors.append(
            ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size,
                               dataset_configs['ABIDE'].patch_size,
                               dataset_configs["ABIDE"].step,
                               [model_trainers[GENERATOR],
                                model_trainers[SEGMENTER]], segment=True))
        input_reconstructors.append(ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size,
                                                       dataset_configs['ABIDE'].patch_size,
                                                       dataset_configs["ABIDE"].step))
        gt_reconstructors.append(ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size,
                                                    dataset_configs['ABIDE'].patch_size,
                                                    dataset_configs["ABIDE"].step))

    # Concat datasets.
    if len(dataset_configs) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    else:
        train_dataset = train_datasets[0]
        valid_dataset = valid_datasets[0]
        test_dataset = test_datasets[0]

    # Create samplers
    if on_multiple_gpus(run_config.devices):
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, run_config.world_size,
                                                            run_config.local_rank)
        valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, run_config.world_size,
                                                            run_config.local_rank)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, run_config.world_size,
                                                           run_config.local_rank)
    else:
        train_sampler, valid_sampler, test_sampler = None, None, None

    # Create loaders.
    dataloaders = list(map(lambda dataset, sampler: DataLoader(dataset,
                                                               training_config.batch_size,
                                                               sampler=sampler,
                                                               shuffle=False if sampler is not None else True,
                                                               num_workers=args.num_workers,
                                                               collate_fn=augmented_sample_collate,
                                                               drop_last=True,
                                                               pin_memory=True),
                           [train_dataset, valid_dataset, test_dataset],
                           [train_sampler, valid_sampler, test_sampler]))

    # Initialize the loggers.
    visdom_config = VisdomConfiguration.from_yml(args.config_file, "visdom")
    visdom_logger = VisdomLogger(visdom_config)

    visdom_logger(VisdomData("Experiment", "Experiment Config", PlotType.TEXT_PLOT, PlotFrequency.EVERY_EPOCH, None,
                             config_html))
    visdom_logger(VisdomData("Experiment", "Patch count", PlotType.BAR_PLOT, PlotFrequency.EVERY_EPOCH,
                             x=[len(iSEG_train) if iSEG_train is not None else 0,
                                len(MRBrainS_train) if MRBrainS_train is not None else 0,
                                len(ABIDE_train) if ABIDE_train is not None else 0],
                             y=["iSEG", "MRBrainS", "ABIDE"], params={"opts": {"title": "Patch count"}}))

    save_folder = "saves/" + os.path.basename(os.path.normpath(visdom_config.env))
    [os.makedirs("{}/{}".format(save_folder, model), exist_ok=True)
     for model in
     ["Discriminator", "Generator", "Segmenter"]]

    trainer = DeepNormalizeTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1], dataloaders[2],
                                   reconstruction_datasets, normalized_reconstructors, input_reconstructors,
                                   segmentation_reconstructors, augmented_input_reconstructors, gt_reconstructors,
                                   run_config, dataset_configs, save_folder) \
        .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
        .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
        .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
        .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Generated Batch Process {}".format(run_config.local_rank),
                            PlotType.IMAGES_PLOT,
                            params={"nrow": 4,
                                    "opts": {"store_history": True,
                                             "title": "Generated Patches Process {}".format(
                                                 run_config.local_rank)}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Input Batch Process {}".format(run_config.local_rank),
                            PlotType.IMAGES_PLOT,
                            params={"nrow": 4,
                                    "opts": {"store_history": True,
                                             "title": "Input Patches Process {}".format(run_config.local_rank)}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Segmented Batch Process {}".format(run_config.local_rank),
                            PlotType.IMAGES_PLOT,
                            params={"nrow": 4,
                                    "opts": {"store_history": True,
                                             "title": "Segmented Patches Process {}".format(
                                                 run_config.local_rank)}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger,
                            "Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                            PlotType.IMAGES_PLOT,
                            params={"nrow": 4,
                                    "opts": {"store_history": True,
                                             "title": "Ground Truth Patches Process {}".format(
                                                 run_config.local_rank)}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Label Map Batch Process {}".format(run_config.local_rank),
                            PlotType.IMAGES_PLOT,
                            params={"nrow": 4,
                                    "opts": {"store_history": True,
                                             "title": "Label Map Patches Process {}".format(
                                                 run_config.local_rank)}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "Generated Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "Background Generated Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "CSF Generated Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "GM Generated Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "WM Generated Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={
                                "opts": {"title": "Inputs Intensity Histogram",
                                         "store_history": True,
                                         "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "Background Input Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "CSF Input Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "GM Input Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                            params={"opts": {"title": "WM Input Intensity Histogram",
                                             "store_history": True,
                                             "numbins": 128}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot", PlotType.PIE_PLOT,
                                                params={"opts": {"title": "Classification hit per classes",
                                                                 "legend": list(map(lambda key: key,
                                                                                    dataset_configs.keys())) + [
                                                                               "Fake Class"]}},
                                                every=25), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                params={"opts": {"title": "Batch data distribution",
                                                                 "legend": list(map(lambda key: key,
                                                                                    dataset_configs.keys())) + [
                                                                               "Fake Class"]}},
                                                every=25), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                params={"opts": {"title": "Mean Hausdorff Distance",
                                                                 "legend": ["Test"]}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                params={"opts": {"title": "Metric Table"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                 "rownames": ["Background", "CSF", "GM", "WM"],
                                                                 "title": "Confusion Matrix"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                 "rownames": ["Background", "CSF", "GM", "WM"],
                                                                 "title": "iSEG Confusion Matrix"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                 "rownames": ["Background", "CSF", "GM", "WM"],
                                                                 "title": "MRBrainS Confusion Matrix"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                 "rownames": ["Background", "CSF", "GM", "WM"],
                                                                 "title": "ABIDE Confusion Matrix"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix", PlotType.HEATMAP_PLOT,
                            params={"opts": {
                                "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                "rownames": list(dataset_configs.keys()) + ["Generated"],
                                "title": "Discriminator Confusion Matrix"}},
                            every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix Training", PlotType.HEATMAP_PLOT,
                            params={"opts": {
                                "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                "rownames": list(dataset_configs.keys()) + ["Generated"],
                                "title": "Discriminator Confusion Matrix Training"}},
                            every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                params={"opts": {"title": "Runtime"}},
                                                every=1), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotCustomLoss(visdom_logger, "D(G(X)) | X", every=1), Event.ON_EPOCH_END) \
        .with_event_handler(PlotCustomLoss(visdom_logger, "Total Loss", every=1), Event.ON_EPOCH_END) \
        .with_event_handler(
        PlotCustomLinePlotWithLegend(visdom_logger, "Jensen-Shannon Divergence", every=1,
                                     params={"title": "Jensen-Shannon Divergence on test data per Epoch",
                                             "legend": ["Inputs", "Normalized"]}), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                     params={"title": "Dice score per class per epoch",
                                             "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch on reconstructed iSEG image",
                                     every=1,
                                     params={"title": "Dice score per class per epoch on reconstructed iSEG image",
                                             "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomLinePlotWithLegend(visdom_logger,
                                     "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                     params={
                                         "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                         "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch on reconstructed ABIDE image",
                                     every=1,
                                     params={"title": "Dice score per class per epoch on reconstructed ABIDE image",
                                             "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Input iSEG Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Normalized iSEG Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Normalized iSEG Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Segmented iSEG Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Ground Truth iSEG Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise iSEG Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Initial Noise iSEG Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Noise iSEG After Normalization", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Noise iSEG After Normalization"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Input MRBrainS Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Normalized MRBrainS Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Normalized MRBrainS Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Segmented MRBrainS Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Ground Truth MRBrainS Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise MRBrainS Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Initial Noise MRBrainS Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Noise MRBrainS After Normalization", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Noise MRBrainS After Normalization"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Input ABIDE Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Normalized ABIDE Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Normalized ABIDE Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Ground Truth ABIDE Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True,
                                             "title": "Reconstructed Segmented ABIDE Image"}},
                            every=10), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Conv1 FM", PlotType.IMAGES_PLOT,
                            params={"nrow": 8, "opts": {"store_history": True,
                                                        "title": "Conv1 FM"}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Layer1 FM", PlotType.IMAGES_PLOT,
                            params={"nrow": 8, "opts": {"store_history": True,
                                                        "title": "Layer1 FM"}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Layer2 FM", PlotType.IMAGES_PLOT,
                            params={"nrow": 12, "opts": {"store_history": True,
                                                         "title": "Layer2 FM"}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Layer3 FM", PlotType.IMAGES_PLOT,
                            params={"nrow": 16, "opts": {"store_history": True,
                                                         "title": "Layer3 FM"}},
                            every=500), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Per-Dataset Histograms", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True}}, every=100), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        PlotCustomVariables(visdom_logger, "Reconstructed Images Histograms", PlotType.IMAGE_PLOT,
                            params={"opts": {"store_history": True}}, every=5), Event.ON_TEST_EPOCH_END) \
        .with_event_handler(PlotGPUMemory(visdom_logger, "GPU {} Memory".format(run_config.local_rank),
                                          {"local_rank": run_config.local_rank}, every=50),
                            Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(
        Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                   mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
        .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END) \
        .train(training_config.nb_epochs)
