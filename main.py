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
import torch
import torch.backends.cudnn as cudnn
from kerosene.configs.configs import RunConfiguration, DatasetConfiguration
from kerosene.configs.parsers import YamlConfigurationParser
from kerosene.training.events import Event
from kerosene.events import MonitorMode
from kerosene.events.handlers.checkpoints import Checkpoint
from kerosene.events.handlers.console import PrintTrainingStatus
from kerosene.events.handlers.visdom import PlotMonitors, PlotLR, PlotCustomVariables, PlotAvgGradientPerLayer
from kerosene.loggers.visdom import PlotType, PlotFrequency
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger, VisdomData
from kerosene.training.trainers import ModelTrainerFactory
from kerosene.utils.devices import on_multiple_gpus
from samitorch.inputs.augmentation.strategies import AugmentInput
from samitorch.inputs.augmentation.transformers import AddNoise, AddBiasField
from samitorch.inputs.images import Modality
from samitorch.inputs.utils import sample_collate
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType
from deepNormalize.factories.customCriterionFactory import CustomCriterionFactory
from deepNormalize.factories.customModelFactory import CustomModelFactory
from deepNormalize.inputs.datasets import iSEGSegmentationFactory, MRBrainSSegmentationFactory, ABIDESegmentationFactory
from deepNormalize.training.trainer import DeepNormalizeTrainer

from constants import ISEG_ID, MRBRAINS_ID, ABIDE_ID

cudnn.benchmark = True
cudnn.enabled = True

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
    dataset_configs = list(map(lambda dataset_config: DatasetConfiguration(dataset_config), dataset_configs))
    config_html = [training_config.to_html(), list(map(lambda config: config.to_html(), dataset_configs)),
                   list(map(lambda config: config.to_html(), model_trainer_configs))]

    # Prepare the data.
    train_datasets = list()
    valid_datasets = list()
    test_datasets = list()

    augmentation_strategy = AugmentInput(Compose([AddNoise(exec_probability=0.3, noise_type="rician"),
                                                  AddBiasField(exec_probability=0.3)]))

    if "iSEG" in [dataset_config.name for dataset_config in dataset_configs]:
        iSEG_train, iSEG_valid, iSEG_test, iSEG_CSV = iSEGSegmentationFactory.create_train_valid_test(
            source_dir=dataset_configs[0].path,
            target_dir=dataset_configs[0].path + "/label",
            modalities=dataset_configs[0].modalities,
            dataset_id=ISEG_ID,
            test_size=dataset_configs[0].validation_split,
            augmentation_strategy=augmentation_strategy)
        train_datasets.append(iSEG_train)
        valid_datasets.append(iSEG_valid)
        test_datasets.append(iSEG_test)

    if "MRBrainS" in [dataset_config.name for dataset_config in dataset_configs]:
        MRBrainS_train, MRBrainS_valid, MRBrainS_test, MRBrainS_CSV = MRBrainSSegmentationFactory.create_train_valid_test(
            source_dir=dataset_configs[1 if len(dataset_configs) == 3 else 0].path,
            target_dir=dataset_configs[1 if len(dataset_configs) == 3 else 0].path,
            modalities=dataset_configs[1].modalities,
            dataset_id=MRBRAINS_ID,
            test_size=dataset_configs[1 if len(dataset_configs) == 3 else 0].validation_split,
            augmentation_strategy=augmentation_strategy)
        train_datasets.append(MRBrainS_train)
        valid_datasets.append(MRBrainS_valid)
        test_datasets.append(MRBrainS_test)

    if "ABIDE" in [dataset_config.name for dataset_config in dataset_configs]:
        ABIDE_train, ABIDE_valid, ABIDE_test, ABIDE_CSV = ABIDESegmentationFactory.create_train_valid_test(
            source_dir=dataset_configs[2 if len(dataset_configs) == 3 else 0].path,
            target_dir=dataset_configs[2 if len(dataset_configs) == 3 else 0].path,
            modalities=dataset_configs[2].modalities,
            dataset_id=ABIDE_ID,
            sites=dataset_configs[2 if len(dataset_configs) == 3 else 0].sites,
            test_size=dataset_configs[2 if len(dataset_configs) == 2 else 0].validation_split,
            augmentation_strategy=augmentation_strategy)
        train_datasets.append(ABIDE_train)
        valid_datasets.append(ABIDE_valid)
        test_datasets.append(ABIDE_test)

    # Concat datasets.
    if len(dataset_configs) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    else:
        train_dataset = train_datasets[0]
        valid_dataset = valid_datasets[0]
        test_dataset = test_datasets[0]

    iSEG_reconstruction_dataset = iSEGSegmentationFactory.create(
        "/mnt/md0/Data/Preprocessed/iSEG/TestingData/Patches/Aligned/T1/11",
        None, Modality.T1, ISEG_ID)
    MRBrainS_reconstruction_dataset = MRBrainSSegmentationFactory.create(
        "/mnt/md0/Data/Preprocessed/MRBrainS/TesTData/Patches/Aligned/T1/1", None, Modality.T1, MRBRAINS_ID)
    reconstruction_datasets = [iSEG_reconstruction_dataset, MRBrainS_reconstruction_dataset]

    # Initialize the model trainers
    model_trainer_factory = ModelTrainerFactory(model_factory=CustomModelFactory(),
                                                criterion_factory=CustomCriterionFactory(run_config))
    model_trainers = model_trainer_factory.create(model_trainer_configs)

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
                                                               num_workers=2,
                                                               collate_fn=sample_collate, pin_memory=True),
                           [train_dataset, valid_dataset, test_dataset],
                           [train_sampler, valid_sampler, test_sampler]))

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
                                 x=[np.asarray(
                                     iSEG_CSV.groupby('center_class').count()) if iSEG_train is not None else 0,
                                    np.asarray(MRBrainS_CSV.groupby(
                                        'center_class').count()) if MRBrainS_train is not None else 0],
                                 y=["iSEG", "MRBrainS"], params={
                "opts": {"title": "Center Voxel Class Count", "stacked": True, "legend": ["CSF", "GM", "WM"]}}))

        save_folder = "saves/" + os.path.basename(os.path.normpath(visdom_config.env))
        [os.makedirs("{}/{}".format(save_folder, model), exist_ok=True)
         for model in
         ["Discriminator", "Generator", "Segmenter"]]

        trainer = DeepNormalizeTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1], dataloaders[2],
                                       reconstruction_datasets, run_config) \
            .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
            .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
            .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Generated Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4,
                                                            "opts": {"store_history": False,
                                                                     "title": "Generated Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Input Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4,
                                                            "opts": {"store_history": False,
                                                                     "title": "Input Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Segmented Batch", PlotType.IMAGES_PLOT,
                                                    params={"nrow": 4,
                                                            "opts": {"store_history": False,
                                                                     "title": "Segmented Patches"}},
                                                    every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Segmentation Ground Truth Batch", PlotType.IMAGES_PLOT,
                                params={"nrow": 4,
                                        "opts": {"store_history": False,
                                                 "title": "Ground Truth Patches"}}, every=100),
            Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Reconstructed Segmented Image", PlotType.IMAGE_PLOT,
                                params={"opts": {"store_history": True,
                                                 "title": "Reconstructed Segmented Image"}},
                                every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Reconstructed Input Image", PlotType.IMAGE_PLOT,
                                params={"opts": {"store_history": True,
                                                 "title": "Reconstructed Input Image"}},
                                every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Reconstructed Normalized Image", PlotType.IMAGE_PLOT,
                                params={"opts": {"store_history": True,
                                                 "title": "Reconstructed Normalized Image"}},
                                every=1), Event.ON_EPOCH_END) \
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
                                                    params={"opts": {"title": "Loss D(G(X)) | X",
                                                                     "legend": ["Training",
                                                                                "Validation",
                                                                                "Test"]}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Jensen-Shannon Divergence", PlotType.LINE_PLOT,
                                params={"opts": {"title": "Jensen-Shannon Divergence",
                                                 "legend": ["Inputs", "Normalized"]}},
                                every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                    params={"opts": {"title": "Mean Hausdorff Distance",
                                                                     "legend": ["Test"]}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GPU {} Memory".format(run_config.local_rank), PlotType.LINE_PLOT,
                                params={"opts": {"title": "GPU {} Memory Usage".format(run_config.local_rank),
                                                 "legend": ["Total", "Free", "Used"]}},
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
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix", PlotType.HEATMAP_PLOT,
                                params={"opts": {"columnnames": ["iSEG", "MRBrainS", "Generated"],
                                                 "rownames": ["Generated", "MRBrainS", "iSEG"],
                                                 "title": "Discriminator Confusion Matrix"}},
                                every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                    params={"opts": {"title": "Runtime"}},
                                                    every=1), Event.ON_EPOCH_END) \
            .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=100), Event.ON_TRAIN_BATCH_END) \
            .with_event_handler(
            Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                       mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)

    else:
        trainer = DeepNormalizeTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1], dataloaders[2],
                                       iSEG_reconstruction_dataset, run_config) \
            .with_event_handler(
            PlotCustomVariables(visdom_logger, "GPU {} Memory".format(run_config.local_rank), PlotType.LINE_PLOT,
                                params={"opts": {"title": "GPU {} Memory Usage".format(run_config.local_rank),
                                                 "legend": ["Total", "Free", "Used"]}},
                                every=1), Event.ON_EPOCH_END) \
            .train(training_config.nb_epochs)
