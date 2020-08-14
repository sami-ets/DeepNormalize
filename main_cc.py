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
from kerosene.loggers.visdom import PlotType, PlotFrequency
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger, VisdomData
from kerosene.training.trainers import ModelTrainerFactory
from kerosene.utils.devices import on_multiple_gpus
from samitorch.inputs.augmentation.strategies import AugmentInput
from samitorch.inputs.augmentation.transformers import ShiftHistogram
from samitorch.inputs.utils import augmented_sample_collate
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType
from deepNormalize.factories.customModelFactory import CustomModelFactory
from deepNormalize.factories.customTrainerFactory import TrainerFactory
from deepNormalize.inputs.datasets import iSEGSliceDatasetFactory, MRBrainSSliceDatasetFactory, ABIDESliceDatasetFactory
from deepNormalize.nn.criterions import CustomCriterionFactory
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
    augmented_reconstruction_datasets = list()
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
                                                criterion_factory=CustomCriterionFactory())
    model_trainers = model_trainer_factory.create(model_trainer_configs)
    if not isinstance(model_trainers, list):
        model_trainers = [model_trainers]

    # Create datasets
    if dataset_configs.get("iSEG", None) is not None:
        if dataset_configs["iSEG"].hist_shift_augmentation:
            iSEG_augmentation_strategy = AugmentInput(
                Compose([ShiftHistogram(exec_probability=0.50, min_lambda=-5, max_lambda=5)]))
        iSEG_train, iSEG_valid, iSEG_test, iSEG_reconstruction = iSEGSliceDatasetFactory.create_train_valid_test(
            source_dir=dataset_configs["iSEG"].path,
            modalities=dataset_configs["iSEG"].modalities,
            dataset_id=ISEG_ID,
            test_size=dataset_configs["iSEG"].validation_split,
            max_subjects=dataset_configs["iSEG"].max_subjects,
            max_num_patches=dataset_configs["iSEG"].max_num_patches,
            augmentation_strategy=iSEG_augmentation_strategy,
            patch_size=dataset_configs["iSEG"].patch_size,
            step=dataset_configs["iSEG"].step,
            augmented_path=dataset_configs["iSEG"].path_augmented,
            test_patch_size=dataset_configs["iSEG"].test_patch_size,
            test_step=dataset_configs["iSEG"].test_step)
        train_datasets.append(iSEG_train)
        valid_datasets.append(iSEG_valid)
        test_datasets.append(iSEG_test)
        reconstruction_datasets.append(iSEG_reconstruction)
        normalized_reconstructors.append(
            ImageReconstructor(dataset_configs["iSEG"].reconstruction_size, dataset_configs['iSEG'].test_patch_size,
                               dataset_configs["iSEG"].test_step, [model_trainers[GENERATOR]], normalize=True,
                               test_image=iSEG_reconstruction._augmented_images[
                                   0] if iSEG_reconstruction._augmented_images is not None else
                               iSEG_reconstruction._source_images[0], is_multimodal=True))
        segmentation_reconstructors.append(
            ImageReconstructor(dataset_configs["iSEG"].reconstruction_size, dataset_configs['iSEG'].test_patch_size,
                               dataset_configs["iSEG"].test_step, [model_trainers[GENERATOR],
                                                                   model_trainers[SEGMENTER]],
                               normalize_and_segment=True, test_image=iSEG_reconstruction._augmented_images[
                    0] if iSEG_reconstruction._augmented_images is not None else
                iSEG_reconstruction._source_images[0]))
        input_reconstructors.append(
            ImageReconstructor(dataset_configs["iSEG"].reconstruction_size, dataset_configs['iSEG'].test_patch_size,
                               dataset_configs["iSEG"].test_step, test_image=iSEG_reconstruction._source_images[0], is_multimodal=True))

        gt_reconstructors.append(
            ImageReconstructor(dataset_configs["iSEG"].reconstruction_size, dataset_configs['iSEG'].test_patch_size,
                               dataset_configs["iSEG"].test_step, test_image=iSEG_reconstruction._target_images[0]))

        if dataset_configs["iSEG"].path_augmented is not None:
            augmented_input_reconstructors.append(
                ImageReconstructor(dataset_configs["iSEG"].reconstruction_size, dataset_configs['iSEG'].test_patch_size,
                                   dataset_configs["iSEG"].test_step,
                                   test_image=iSEG_reconstruction._augmented_images[0]))

    if dataset_configs.get("MRBrainS", None) is not None:
        if dataset_configs["MRBrainS"].hist_shift_augmentation:
            MRBrainS_augmentation_strategy = AugmentInput(
                Compose([ShiftHistogram(exec_probability=0.50, min_lambda=-5, max_lambda=5)]))

        MRBrainS_train, MRBrainS_valid, MRBrainS_test, MRBrainS_reconstruction = MRBrainSSliceDatasetFactory.create_train_valid_test(
            source_dir=dataset_configs["MRBrainS"].path,
            modalities=dataset_configs["MRBrainS"].modalities,
            dataset_id=MRBRAINS_ID,
            test_size=dataset_configs["MRBrainS"].validation_split,
            max_subjects=dataset_configs["MRBrainS"].max_subjects,
            max_num_patches=dataset_configs["MRBrainS"].max_num_patches,
            augmentation_strategy=MRBrainS_augmentation_strategy,
            patch_size=dataset_configs["MRBrainS"].patch_size,
            step=dataset_configs["MRBrainS"].step,
            augmented_path=dataset_configs["MRBrainS"].path_augmented,
            test_patch_size=dataset_configs["MRBrainS"].test_patch_size,
            test_step=dataset_configs["MRBrainS"].test_step)
        train_datasets.append(MRBrainS_train)
        valid_datasets.append(MRBrainS_valid)
        test_datasets.append(MRBrainS_test)
        reconstruction_datasets.append(MRBrainS_reconstruction)
        normalized_reconstructors.append(ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                                                            dataset_configs['MRBrainS'].test_patch_size,
                                                            dataset_configs["MRBrainS"].test_step,
                                                            [model_trainers[GENERATOR]], normalize=True,
                                                            test_image=MRBrainS_reconstruction._augmented_images[
                                                                0] if MRBrainS_reconstruction._augmented_images is not None else
                                                            MRBrainS_reconstruction._source_images[0], is_multimodal=True))
        segmentation_reconstructors.append(
            ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                               dataset_configs['MRBrainS'].test_patch_size, dataset_configs["MRBrainS"].test_step,
                               [model_trainers[GENERATOR],
                                model_trainers[SEGMENTER]], normalize_and_segment=True,
                               test_image=MRBrainS_reconstruction._augmented_images[
                                   0] if MRBrainS_reconstruction._augmented_images is not None else
                               MRBrainS_reconstruction._source_images[0]))
        input_reconstructors.append(ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                                                       dataset_configs['MRBrainS'].test_patch_size,
                                                       dataset_configs["MRBrainS"].test_step,
                                                       test_image=MRBrainS_reconstruction._source_images[0], is_multimodal=True))

        gt_reconstructors.append(ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                                                    dataset_configs['MRBrainS'].test_patch_size,
                                                    dataset_configs["MRBrainS"].test_step,
                                                    test_image=MRBrainS_reconstruction._target_images[0]))

        if dataset_configs["MRBrainS"].path_augmented is not None:
            augmented_input_reconstructors.append(
                ImageReconstructor(dataset_configs["MRBrainS"].reconstruction_size,
                                   dataset_configs['MRBrainS'].test_patch_size, dataset_configs["MRBrainS"].test_step,
                                   test_image=MRBrainS_reconstruction._augmented_images[0]))

    if dataset_configs.get("ABIDE", None) is not None:
        if dataset_configs["ABIDE"].hist_shift_augmentation:
            ABIDE_augmentation_strategy = AugmentInput(
                Compose([ShiftHistogram(exec_probability=0.50, min_lambda=-5, max_lambda=5)]))

        ABIDE_train, ABIDE_valid, ABIDE_test, ABIDE_reconstruction = ABIDESliceDatasetFactory.create_train_valid_test(
            source_dir=dataset_configs["ABIDE"].path,
            modalities=dataset_configs["ABIDE"].modalities,
            dataset_id=ABIDE_ID,
            sites=dataset_configs["ABIDE"].sites,
            max_subjects=dataset_configs["ABIDE"].max_subjects,
            test_size=dataset_configs["ABIDE"].validation_split,
            max_num_patches=dataset_configs["ABIDE"].max_num_patches,
            augmentation_strategy=ABIDE_augmentation_strategy,
            patch_size=dataset_configs["ABIDE"].patch_size,
            step=dataset_configs["ABIDE"].step,
            test_patch_size=dataset_configs["ABIDE"].test_patch_size,
            test_step=dataset_configs["ABIDE"].test_step)
        train_datasets.append(ABIDE_train)
        valid_datasets.append(ABIDE_valid)
        test_datasets.append(ABIDE_test)
        reconstruction_datasets.append(ABIDE_reconstruction)
        normalized_reconstructors.append(
            ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size, dataset_configs['ABIDE'].test_patch_size,
                               dataset_configs["ABIDE"].test_step, [model_trainers[GENERATOR]], normalize=True,
                               test_image=ABIDE_reconstruction._augmented_images[
                                   0] if ABIDE_reconstruction._augmented_images is not None else
                               ABIDE_reconstruction._source_images[0]))
        segmentation_reconstructors.append(
            ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size, dataset_configs['ABIDE'].test_patch_size,
                               dataset_configs["ABIDE"].test_step, [model_trainers[GENERATOR],
                                                                    model_trainers[SEGMENTER]],
                               normalize_and_segment=True, test_image=ABIDE_reconstruction._augmented_images[
                    0] if ABIDE_reconstruction._augmented_images is not None else
                ABIDE_reconstruction._source_images[0]))
        input_reconstructors.append(
            ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size, dataset_configs['ABIDE'].test_patch_size,
                               dataset_configs["ABIDE"].test_step, test_image=ABIDE_reconstruction._source_images[0]))

        gt_reconstructors.append(
            ImageReconstructor(dataset_configs["ABIDE"].reconstruction_size, dataset_configs['ABIDE'].test_patch_size,
                               dataset_configs["ABIDE"].test_step, test_image=ABIDE_reconstruction._target_images[0]))

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

    trainer = TrainerFactory(training_config.trainer).create(training_config, model_trainers, dataloaders,
                                                             reconstruction_datasets, normalized_reconstructors,
                                                             input_reconstructors,
                                                             segmentation_reconstructors,
                                                             augmented_input_reconstructors, gt_reconstructors,
                                                             run_config, dataset_configs, save_folder,
                                                             visdom_logger)

    trainer.train(training_config.nb_epochs)
