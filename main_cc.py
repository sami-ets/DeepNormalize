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
from samitorch.inputs.utils import augmented_sample_collate
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader

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
    data_augmentation_config = YamlConfigurationParser.parse_section(args.config_file, "data_augmentation")
    config_html = [training_config.to_html(), list(map(lambda config: config.to_html(), dataset_configs.values())),
                   list(map(lambda config: config.to_html(), model_trainer_configs))]

    # Prepare the data.
    train_datasets = list()
    valid_datasets = list()
    test_datasets = list()
    reconstruction_datasets = list()

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
        iSEG_train, iSEG_valid, iSEG_test, iSEG_reconstruction = iSEGSliceDatasetFactory.create_train_valid_test(
            source_dir=dataset_configs["iSEG"].path,
            modalities=dataset_configs["iSEG"].modalities,
            dataset_id=ISEG_ID,
            test_size=dataset_configs["iSEG"].validation_split,
            max_subjects=dataset_configs["iSEG"].max_subjects,
            max_num_patches=dataset_configs["iSEG"].max_num_patches,
            augment=dataset_configs["iSEG"].augment,
            patch_size=dataset_configs["iSEG"].patch_size,
            step=dataset_configs["iSEG"].step,
            test_patch_size=dataset_configs["iSEG"].test_patch_size,
            test_step=dataset_configs["iSEG"].test_step,
            data_augmentation_config=data_augmentation_config)
        train_datasets.append(iSEG_train)
        valid_datasets.append(iSEG_valid)
        test_datasets.append(iSEG_test)
        reconstruction_datasets.append(iSEG_reconstruction)

    if dataset_configs.get("MRBrainS", None) is not None:
        MRBrainS_train, MRBrainS_valid, MRBrainS_test, MRBrainS_reconstruction = MRBrainSSliceDatasetFactory.create_train_valid_test(
            source_dir=dataset_configs["MRBrainS"].path,
            modalities=dataset_configs["MRBrainS"].modalities,
            dataset_id=MRBRAINS_ID,
            test_size=dataset_configs["MRBrainS"].validation_split,
            max_subjects=dataset_configs["MRBrainS"].max_subjects,
            max_num_patches=dataset_configs["MRBrainS"].max_num_patches,
            augment=dataset_configs["MRBrainS"].augment,
            patch_size=dataset_configs["MRBrainS"].patch_size,
            step=dataset_configs["MRBrainS"].step,
            test_patch_size=dataset_configs["MRBrainS"].test_patch_size,
            test_step=dataset_configs["MRBrainS"].test_step,
            data_augmentation_config=data_augmentation_config)
        train_datasets.append(MRBrainS_train)
        valid_datasets.append(MRBrainS_valid)
        test_datasets.append(MRBrainS_test)
        reconstruction_datasets.append(MRBrainS_reconstruction)

    if dataset_configs.get("ABIDE", None) is not None:
        ABIDE_train, ABIDE_valid, ABIDE_test, ABIDE_reconstruction = ABIDESliceDatasetFactory.create_train_valid_test(
            source_dir=dataset_configs["ABIDE"].path,
            modalities=dataset_configs["ABIDE"].modalities,
            dataset_id=ABIDE_ID,
            sites=dataset_configs["ABIDE"].sites,
            max_subjects=dataset_configs["ABIDE"].max_subjects,
            test_size=dataset_configs["ABIDE"].validation_split,
            max_num_patches=dataset_configs["ABIDE"].max_num_patches,
            augment=dataset_configs["ABIDE"].augment,
            patch_size=dataset_configs["ABIDE"].patch_size,
            step=dataset_configs["ABIDE"].step,
            test_patch_size=dataset_configs["ABIDE"].test_patch_size,
            test_step=dataset_configs["ABIDE"].test_step,
            data_augmentation_config=data_augmentation_config)
        train_datasets.append(ABIDE_train)
        valid_datasets.append(ABIDE_valid)
        test_datasets.append(ABIDE_test)
        reconstruction_datasets.append(ABIDE_reconstruction)

    if len(list(dataset_configs.keys())) == 2:
        normalized_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0]],
            patch_size=dataset_configs["iSEG"].test_patch_size,
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            models=[model_trainers[GENERATOR]],
            normalize=True,
            batch_size=5)
        segmentation_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0]],
            patch_size=dataset_configs["iSEG"].test_patch_size,
            reconstructed_image_size=(1, 256, 256, 192), step=dataset_configs["iSEG"].test_step,
            models=[model_trainers[GENERATOR], model_trainers[SEGMENTER]],
            normalize_and_segment=True,
            batch_size=5)
        input_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0]],
            patch_size=dataset_configs["iSEG"].test_patch_size,
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            batch_size=50)
        gt_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._target_images[0], MRBrainS_reconstruction._target_images[0]],
            patch_size=dataset_configs["iSEG"].test_patch_size,
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            is_ground_truth=True,
            batch_size=50)
        if dataset_configs["iSEG"].augment:
            augmented_input_reconstructor = ImageReconstructor(
                [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0]],
                patch_size=dataset_configs["iSEG"].test_patch_size,
                reconstructed_image_size=(1, 256, 256, 192),
                step=dataset_configs["iSEG"].test_step,
                batch_size=50,
                alpha=data_augmentation_config["test"]["bias_field"]["alpha"][0],
                prob_bias=data_augmentation_config["test"]["bias_field"]["prob_bias"],
                snr=data_augmentation_config["test"]["noise"]["snr"],
                prob_noise=data_augmentation_config["test"]["noise"]["prob_noise"])
            augmented_normalized_input_reconstructor = ImageReconstructor(
                [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0]],
                patch_size=dataset_configs["iSEG"].test_patch_size,
                reconstructed_image_size=(1, 256, 256, 192),
                step=dataset_configs["iSEG"].test_step,
                models=[model_trainers[GENERATOR]],
                batch_size=5,
                alpha=data_augmentation_config["test"]["bias_field"]["alpha"][0],
                prob_bias=data_augmentation_config["test"]["bias_field"]["prob_bias"],
                snr=data_augmentation_config["test"]["noise"]["snr"],
                prob_noise=data_augmentation_config["test"]["noise"]["prob_noise"])
        else:
            augmented_input_reconstructor = None
            augmented_normalized_input_reconstructor = None
    else:
        normalized_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0],
             ABIDE_reconstruction._source_images[0]],
            patch_size=(1, 32, 32, 32),
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            models=[model_trainers[GENERATOR]],
            normalize=True,
            batch_size=4)
        segmentation_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0],
             ABIDE_reconstruction._source_images[0]],
            patch_size=(1, 32, 32, 32),
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            models=[model_trainers[GENERATOR], model_trainers[SEGMENTER]],
            normalize_and_segment=True,
            batch_size=4)
        input_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0],
             ABIDE_reconstruction._source_images[0]],
            patch_size=(1, 32, 32, 32),
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            batch_size=50)
        gt_reconstructor = ImageReconstructor(
            [iSEG_reconstruction._target_images[0], MRBrainS_reconstruction._target_images[0],
             ABIDE_reconstruction._target_images[0]],
            patch_size=(1, 32, 32, 32),
            reconstructed_image_size=(1, 256, 256, 192),
            step=dataset_configs["iSEG"].test_step,
            batch_size=50,
            is_ground_truth=True)
        if dataset_configs["iSEG"].augment:
            augmented_input_reconstructor = ImageReconstructor(
                [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0],
                 ABIDE_reconstruction._source_images[0]],
                patch_size=(1, 32, 32, 32),
                reconstructed_image_size=(1, 256, 256, 192),
                step=dataset_configs["iSEG"].test_step,
                batch_size=50,
                alpha=data_augmentation_config["test"]["bias_field"]["alpha"][0],
                prob_bias=data_augmentation_config["test"]["bias_field"]["prob_bias"],
                snr=data_augmentation_config["test"]["noise"]["snr"],
                prob_noise=data_augmentation_config["test"]["noise"]["prob_noise"])
            augmented_normalized_input_reconstructor = ImageReconstructor(
                [iSEG_reconstruction._source_images[0], MRBrainS_reconstruction._source_images[0],
                 ABIDE_reconstruction._source_images[0]],
                patch_size=(1, 32, 32, 32),
                reconstructed_image_size=(1, 256, 256, 192),
                step=dataset_configs["iSEG"].test_step,
                models=[model_trainers[GENERATOR]],
                batch_size=4,
                alpha=data_augmentation_config["test"]["bias_field"]["alpha"][0],
                prob_bias=data_augmentation_config["test"]["bias_field"]["prob_bias"],
                snr=data_augmentation_config["test"]["noise"]["snr"],
                prob_noise=data_augmentation_config["test"]["noise"]["prob_noise"])
        else:
            augmented_input_reconstructor = None
            augmented_normalized_input_reconstructor = None

    # Concat datasets.
    if len(dataset_configs) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    else:
        train_dataset = train_datasets[0]
        valid_dataset = valid_datasets[0]
        test_dataset = test_datasets[0]

    # Create loaders.
    dataloaders = list(map(lambda dataset: DataLoader(dataset,
                                                      training_config.batch_size,
                                                      sampler=None,
                                                      shuffle=True,
                                                      num_workers=args.num_workers,
                                                      collate_fn=augmented_sample_collate,
                                                      drop_last=True,
                                                      pin_memory=True),
                           [train_dataset, valid_dataset, test_dataset]))

    # Initialize the loggers.
    visdom_config = VisdomConfiguration.from_yml(args.config_file, "visdom")
    exp = args.config_file.split("/")[-3:]
    if visdom_config.save_destination is not None:
        save_folder = visdom_config.save_destination + os.path.join(exp[0], exp[1],
                                                                    os.path.basename(
                                                                        os.path.normpath(visdom_config.env)))
    else:
        save_folder = "saves/{}".format(os.path.basename(os.path.normpath(visdom_config.env)))

    [os.makedirs("{}/{}".format(save_folder, model), exist_ok=True)
     for model in
     ["Discriminator", "Generator", "Segmenter"]]
    visdom_logger = VisdomLogger(visdom_config)

    visdom_logger(VisdomData("Experiment", "Experiment Config", PlotType.TEXT_PLOT, PlotFrequency.EVERY_EPOCH, None,
                             config_html))
    visdom_logger(VisdomData("Experiment", "Patch count", PlotType.BAR_PLOT, PlotFrequency.EVERY_EPOCH,
                             x=[len(iSEG_train) if iSEG_train is not None else 0,
                                len(MRBrainS_train) if MRBrainS_train is not None else 0,
                                len(ABIDE_train) if ABIDE_train is not None else 0],
                             y=["iSEG", "MRBrainS", "ABIDE"], params={"opts": {"title": "Patch count"}}))

    trainer = TrainerFactory(training_config.trainer).create(training_config,
                                                             model_trainers,
                                                             dataloaders,
                                                             reconstruction_datasets,
                                                             normalized_reconstructor,
                                                             input_reconstructor,
                                                             segmentation_reconstructor,
                                                             augmented_input_reconstructor,
                                                             augmented_normalized_input_reconstructor,
                                                             gt_reconstructor,
                                                             run_config,
                                                             dataset_configs,
                                                             save_folder,
                                                             visdom_logger)

    trainer.train(training_config.nb_epochs)
