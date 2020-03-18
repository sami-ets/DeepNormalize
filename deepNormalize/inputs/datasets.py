from typing import List, Optional, Callable, Union

import numpy as np
import os
import pandas
from math import ceil

import torch
from samitorch.inputs.augmentation.strategies import DataAugmentationStrategy
from samitorch.inputs.datasets import AbstractDatasetFactory, SegmentationDataset
from samitorch.inputs.images import Modality
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor
from samitorch.utils.files import extract_file_paths
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from deepNormalize.utils.utils import natural_sort


class iSEGSegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_paths: Union[List[str], np.ndarray], target_paths: Union[List[str], np.ndarray],
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_paths is not None:
            samples = list(
                map(lambda source, target: Sample(x=source, y=target, is_labeled=True, dataset_id=dataset_id),
                    source_paths, target_paths))

            return SegmentationDataset(list(source_paths), list(target_paths), samples, modalities, dataset_id,
                                       Compose(
                                           [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

        else:
            samples = list(
                map(lambda source: Sample(x=source, y=None, is_labeled=False, dataset_id=dataset_id), source_paths))

            return SegmentationDataset(list(source_paths), None, samples, modalities, dataset_id, Compose(
                [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, max_subject: int = None, max_num_patches=None,
                          augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list) and len(modalities) > 1:
            return iSEGSegmentationFactory._create_multimodal_train_test(source_dir, modalities,
                                                                         dataset_id, test_size, max_subject,
                                                                         max_num_patches,
                                                                         augmentation_strategy)
        else:
            return iSEGSegmentationFactory._create_single_modality_train_test(source_dir, modalities,
                                                                              dataset_id, test_size, max_subject,
                                                                              max_num_patches,
                                                                              augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                                dataset_id: int, test_size: float, max_subjects: int = None, max_num_patches=None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return iSEGSegmentationFactory._create_multimodal_train_valid_test(source_dir, modalities,
                                                                               dataset_id, test_size, max_subjects,
                                                                               max_num_patches,
                                                                               augmentation_strategy)
        else:
            return iSEGSegmentationFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                    dataset_id, test_size, max_subjects,
                                                                                    max_num_patches,
                                                                                    augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           max_subjects: int = None, max_num_patches: int = None,
                                           augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))
        subjects = np.array([dir for dir in sorted(os.listdir(os.path.join(source_dir, str(modality))))]).astype(np.int)

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subjects)), max_subjects, replace=False)
            subjects = subjects[choices]

        train_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)
        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), train_subjects))]
        test_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), test_subjects))]
        reconstruction_csv = csv[
            csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full/", str(modality), str(reconstruction_subject)))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=test_source_paths,
                                                      target_paths=test_target_paths,
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = iSEGSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modality,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                                 max_subjects: int = None, max_num_patches: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        subject_dirs = [dir for dir in sorted(os.listdir(os.path.join(source_dir, str(modality))))]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), train_subjects))]
        valid_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), valid_subjects))]
        test_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), test_subjects))]
        reconstruction_csv = csv[
            csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            valid_csv = valid_csv.sample(n=ceil(max_num_patches * test_size))
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        valid_source_paths, valid_target_paths = shuffle(np.array(natural_sort(list(valid_csv[str(modality)]))),
                                                         np.array(natural_sort(list(valid_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full/", str(modality), str(reconstruction_subject)))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=valid_source_paths,
            target_paths=valid_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=test_source_paths,
                                                      target_paths=test_target_paths,
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = iSEGSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modality,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, modalities: List[Modality], dataset_id: int, test_size: float,
                                      max_subjects: int = None, max_num_patches: int = None,
                                      augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))
        subjects = np.array([dir for dir in sorted(os.listdir(os.path.join(source_dir, str(modalities[0]))))]).astype(
            np.int)

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subjects)), max_subjects, replace=False)
            subjects = subjects[choices]

        train_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)
        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), train_subjects))]
        test_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), test_subjects))]
        reconstruction_csv = csv[
            csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = np.stack(natural_sort(list([extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full/", str(modality), str(reconstruction_subject))) for
                modality in modalities])), axis=1)

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=test_source_paths,
                                                      target_paths=test_target_paths,
                                                      modalities=modalities,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = iSEGSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modalities,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_valid_test(source_dir: str, modalities: List[Modality],
                                            dataset_id: int, test_size: float, max_subjects: int = None,
                                            max_num_patches: int = None,
                                            augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))
        subjects = np.array([dir for dir in sorted(os.listdir(os.path.join(source_dir, str(modalities[0]))))]).astype(
            np.int)

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subjects)), max_subjects, replace=False)
            subjects = subjects[choices]

        train_subjects, valid_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = MRBrainSSegmentationFactory.shuffle_split(valid_subjects, test_size)
        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), train_subjects))]
        valid_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), valid_subjects))]
        test_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), test_subjects))]
        reconstruction_csv = csv[
            csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["labels"]))))
        valid_source_paths, valid_target_paths = shuffle(
            np.stack([natural_sort(list(valid_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(valid_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = np.stack(natural_sort(list([extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full/", str(modality), str(reconstruction_subject))) for
                modality in modalities])), axis=1)

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=valid_source_paths,
            target_paths=valid_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=test_source_paths,
                                                      target_paths=test_target_paths,
                                                      modalities=modalities,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = iSEGSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modalities,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def shuffle_split(subjects: np.ndarray, split_ratio: Union[float, int]):
        shuffle(subjects)
        return subjects[ceil(len(subjects) * split_ratio):], subjects[0:ceil(len(subjects) * split_ratio)]


class MRBrainSSegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_paths: Union[List[str], np.ndarray], target_paths: Union[List[str], np.ndarray],
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_paths is not None:
            samples = list(
                map(lambda source, target: Sample(x=source, y=target, is_labeled=True, dataset_id=dataset_id),
                    source_paths, target_paths))

            return SegmentationDataset(list(source_paths), list(target_paths), samples, modalities, dataset_id,
                                       Compose(
                                           [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

        else:
            samples = list(
                map(lambda source: Sample(x=source, y=None, is_labeled=False, dataset_id=dataset_id), source_paths))

            return SegmentationDataset(list(source_paths), None, samples, modalities, dataset_id, Compose(
                [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, max_subjects: int = None, max_num_patches: int = None,
                          augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return MRBrainSSegmentationFactory._create_multimodal_train_test(source_dir, modalities,
                                                                             dataset_id, test_size, max_subjects,
                                                                             max_num_patches,
                                                                             augmentation_strategy)
        else:
            return MRBrainSSegmentationFactory._create_single_modality_train_test(source_dir, modalities,
                                                                                  dataset_id, test_size, max_subjects,
                                                                                  max_num_patches,
                                                                                  augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]], dataset_id: int,
                                test_size: float, max_subjects: int = None, max_num_patches: int = None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return MRBrainSSegmentationFactory._create_multimodal_train_valid_test(source_dir, modalities,
                                                                                   dataset_id, test_size, max_subjects,
                                                                                   max_num_patches,
                                                                                   augmentation_strategy)
        else:
            return MRBrainSSegmentationFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                        dataset_id, test_size,
                                                                                        max_subjects, max_num_patches,
                                                                                        augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int,
                                           test_size: float, max_subjects: int = None, max_num_patches: int = None,
                                           augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs), replace=False)

        subjects = np.array(sorted(all_dirs))[choices]
        train_subjects, test_subjects = MRBrainSSegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv[csv["center_class"].astype(np.int).isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, train_subjects))]
        test_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, test_subjects))]
        reconstruction_csv = csv[csv[str(modality)].str.match("{}/{}".format(source_dir, reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = shuffle(
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full", str(reconstruction_subject), str(modality)))

        train_dataset = MRBrainSSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = MRBrainSSegmentationFactory.create(source_paths=test_source_paths,
                                                          target_paths=test_target_paths,
                                                          modalities=modality,
                                                          dataset_id=dataset_id,
                                                          transforms=[ToNumpyArray(), ToNDTensor()],
                                                          augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = MRBrainSSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modality,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality, dataset_id: int,
                                                 test_size: float, max_subjects: int = None,
                                                 max_num_patches: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs), replace=False)

        subjects = np.array(sorted(all_dirs))[choices]
        train_subjects, valid_subjects = MRBrainSSegmentationFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = MRBrainSSegmentationFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv[csv["center_class"].astype(np.int).isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, train_subjects))]
        valid_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, valid_subjects))]
        test_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, test_subjects))]
        reconstruction_csv = csv[csv[str(modality)].str.match("{}/{}".format(source_dir, reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            valid_csv = valid_csv.sample(n=ceil(max_num_patches * test_size))
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        valid_source_paths, valid_target_paths = shuffle(np.array(natural_sort(list(valid_csv[str(modality)]))),
                                                         np.array(natural_sort(list(valid_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = shuffle(
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full", str(reconstruction_subject), str(modality)))

        train_dataset = MRBrainSSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = MRBrainSSegmentationFactory.create(
            source_paths=valid_source_paths,
            target_paths=valid_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = MRBrainSSegmentationFactory.create(source_paths=test_source_paths,
                                                          target_paths=test_target_paths,
                                                          modalities=modality,
                                                          dataset_id=dataset_id,
                                                          transforms=[ToNumpyArray(), ToNDTensor()],
                                                          augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = MRBrainSSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modality,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, modalities: List[Modality],
                                      dataset_id: int, test_size: float, max_subjects: int = None,
                                      max_num_patches: int = None,
                                      augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs), replace=False)

        subjects = np.array(sorted(all_dirs))[choices]
        train_subjects, test_subjects = MRBrainSSegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv[csv["center_class"].astype(np.int).isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, train_subjects))]
        test_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, test_subjects))]
        reconstruction_csv = csv[csv[str(modalities[0])].str.match("{}/{}".format(source_dir, reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = np.stack(natural_sort(list([extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full/", str(modality), str(reconstruction_subject))) for
                modality in modalities])), axis=1)

        train_dataset = MRBrainSSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = MRBrainSSegmentationFactory.create(source_paths=test_source_paths,
                                                          target_paths=test_target_paths,
                                                          modalities=modalities,
                                                          dataset_id=dataset_id,
                                                          transforms=[ToNumpyArray(), ToNDTensor()],
                                                          augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = MRBrainSSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modalities,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_valid_test(source_dir: str, modalities: List[Modality],
                                            dataset_id: int, test_size: float, max_subjects: int = None,
                                            max_num_patches: int = None,
                                            augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs), replace=False)

        subjects = np.array(sorted(all_dirs))[choices]
        train_subjects, valid_subjects = MRBrainSSegmentationFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = MRBrainSSegmentationFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv[csv["center_class"].astype(np.int).isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, train_subjects))]
        valid_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, valid_subjects))]
        test_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, test_subjects))]
        reconstruction_csv = csv[csv[str(modalities[0])].str.match("{}/{}".format(source_dir, reconstruction_subject))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        valid_source_paths, valid_target_paths = shuffle(
            np.stack([natural_sort(list(valid_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(valid_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        if augmentation_strategy:
            reconstruction_augmented_paths = np.stack(natural_sort(list([extract_file_paths(
                os.path.join(source_dir, "../../Augmented/Full/", str(modality), str(reconstruction_subject))) for
                modality in modalities])), axis=1)

        train_dataset = MRBrainSSegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = MRBrainSSegmentationFactory.create(source_paths=valid_source_paths,
                                                           target_paths=valid_target_paths,
                                                           modalities=modalities,
                                                           dataset_id=dataset_id,
                                                           transforms=[ToNumpyArray(), ToNDTensor()],
                                                           augmentation_strategy=None)

        test_dataset = MRBrainSSegmentationFactory.create(source_paths=test_source_paths,
                                                          target_paths=test_target_paths,
                                                          modalities=modalities,
                                                          dataset_id=dataset_id,
                                                          transforms=[ToNumpyArray(), ToNDTensor()],
                                                          augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=reconstruction_source_paths,
            target_paths=reconstruction_target_paths,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        if augmentation_strategy:
            reconstruction_augmented_dataset = MRBrainSSegmentationFactory.create(
                source_paths=reconstruction_augmented_paths,
                target_paths=reconstruction_target_paths,
                modalities=modalities,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)
        else:
            reconstruction_augmented_dataset = None

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, reconstruction_augmented_dataset, filtered_csv

    @staticmethod
    def shuffle_split(subjects: np.ndarray, split_ratio: Union[float, int]):
        shuffle(subjects)
        return subjects[ceil(len(subjects) * split_ratio):], subjects[0:ceil(len(subjects) * split_ratio)]


class ABIDESegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_paths: Union[List[str], np.ndarray], target_paths: Union[List[str], np.ndarray],
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_paths is not None:
            samples = list(
                map(lambda source, target: Sample(x=source, y=target, is_labeled=True, dataset_id=dataset_id),
                    source_paths, target_paths))

            return SegmentationDataset(list(source_paths), list(target_paths), samples, modalities, dataset_id,
                                       Compose(
                                           [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

        else:
            samples = list(
                map(lambda source: Sample(x=source, y=None, is_labeled=False, dataset_id=dataset_id), source_paths))

            return SegmentationDataset(list(source_paths), None, samples, modalities, dataset_id, Compose(
                [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, sites: List[str] = None, max_subjects: int = None,
                          max_num_patches: int = None, augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            raise NotImplementedError("ABIDE only contain T1 modality.")
        else:
            return ABIDESegmentationFactory._create_single_modality_train_test(source_dir, modalities,
                                                                               dataset_id, test_size, sites,
                                                                               max_subjects, max_num_patches,
                                                                               augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                                dataset_id: int, test_size: float, sites: List[str] = None, max_subjects: int = None,
                                max_num_patches: int = None, augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            raise NotImplementedError("ABIDE only contain T1 modality.")
        else:
            return ABIDESegmentationFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                     dataset_id, test_size, sites,
                                                                                     max_subjects, max_num_patches,
                                                                                     augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           sites: List[str] = None, max_subjects: int = None,
                                           max_num_patches: int = None, augmentation_strategy=None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))
        subject_dirs = [dir for dir in sorted(os.listdir(source_dir))]

        if sites is not None:
            subject_dirs = ABIDESegmentationFactory.filter_subjects_by_sites(source_dir, sites)
        if max_subjects is not None:
            choices = np.random.choice(len(subject_dirs), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = ABIDESegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv[csv["center_class"].astype(np.int).isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv[str(modality)].str.contains("|".join(train_subjects))]
        test_csv = filtered_csv[filtered_csv[str(modality)].str.contains("|".join(list(test_subjects)))]
        reconstruction_csv = csv[csv[str(modality)].str.contains("|".join(list(reconstruction_subject)))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        train_dataset = ABIDESegmentationFactory.create(
            source_paths=train_source_paths,
            target_paths=train_target_paths,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = ABIDESegmentationFactory.create(source_paths=test_source_paths,
                                                       target_paths=test_target_paths,
                                                       modalities=modality,
                                                       dataset_id=dataset_id,
                                                       transforms=[ToNumpyArray(), ToNDTensor()],
                                                       augmentation_strategy=None)

        reconstruction_dataset = ABIDESegmentationFactory.create(source_paths=reconstruction_source_paths,
                                                                 target_paths=reconstruction_target_paths,
                                                                 modalities=modality,
                                                                 dataset_id=dataset_id,
                                                                 transforms=[ToNumpyArray(), ToNDTensor()],
                                                                 augmentation_strategy=None)

        return train_dataset, test_dataset, reconstruction_dataset, train_csv

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality,
                                                 dataset_id: int, test_size: float, sites: List[str] = None,
                                                 max_subjects: int = None, max_num_patches: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))
        subject_dirs = [dir for dir in sorted(os.listdir(source_dir))]

        if sites is not None:
            subject_dirs = ABIDESegmentationFactory.filter_subjects_by_sites(source_dir, sites)
        if max_subjects is not None:
            choices = np.random.choice(len(subject_dirs), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = ABIDESegmentationFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = ABIDESegmentationFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv[csv["center_class"].astype(np.int).isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv[str(modality)].str.contains("|".join(train_subjects))]
        valid_csv = filtered_csv[filtered_csv[str(modality)].str.contains("|".join(valid_subjects))]
        test_csv = filtered_csv[filtered_csv[str(modality)].str.contains("|".join(list(test_subjects)))]
        reconstruction_csv = csv[csv[str(modality)].str.contains("|".join(list(reconstruction_subject)))]

        if max_num_patches is not None:
            train_csv = train_csv.sample(n=max_num_patches)
            valid_csv = valid_csv.sample(n=ceil(max_num_patches * test_size))
            test_csv = test_csv.sample(n=ceil(max_num_patches * test_size))

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        valid_source_paths, valid_target_paths = shuffle(np.array(natural_sort(list(valid_csv[str(modality)]))),
                                                         np.array(natural_sort(list(valid_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        if sites is not None:
            train_datasets = list()
            valid_datasets = list()
            test_datasets = list()

            for i in range(len(sites)):
                train_dataset = ABIDESegmentationFactory.create(
                    source_paths=np.array([path for path in train_source_paths if sites[i] in path]),
                    target_paths=np.array([path for path in train_target_paths if sites[i] in path]),
                    modalities=modality,
                    dataset_id=dataset_id + i,
                    transforms=[ToNumpyArray(), ToNDTensor()],
                    augmentation_strategy=augmentation_strategy)
                train_datasets.append(train_dataset)

                valid_dataset = ABIDESegmentationFactory.create(
                    source_paths=np.array([path for path in valid_source_paths if sites[i] in path]),
                    target_paths=np.array([path for path in valid_target_paths if sites[i] in path]),
                    modalities=modality,
                    dataset_id=dataset_id + i,
                    transforms=[ToNumpyArray(), ToNDTensor()],
                    augmentation_strategy=None)
                valid_datasets.append(valid_dataset)

                test_dataset = ABIDESegmentationFactory.create(
                    source_paths=np.array([path for path in test_source_paths if sites[i] in path]),
                    target_paths=np.array([path for path in test_target_paths if sites[i] in path]),
                    modalities=modality,
                    dataset_id=dataset_id + i,
                    transforms=[ToNumpyArray(), ToNDTensor()],
                    augmentation_strategy=None)
                test_datasets.append(test_dataset)

            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
            test_dataset = torch.utils.data.ConcatDataset(test_datasets)

        else:
            train_dataset = ABIDESegmentationFactory.create(
                source_paths=train_source_paths,
                target_paths=train_target_paths,
                modalities=modality,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=augmentation_strategy)

            valid_dataset = ABIDESegmentationFactory.create(
                source_paths=valid_source_paths,
                target_paths=valid_target_paths,
                modalities=modality,
                dataset_id=dataset_id,
                transforms=[ToNumpyArray(), ToNDTensor()],
                augmentation_strategy=None)

            test_dataset = ABIDESegmentationFactory.create(source_paths=test_source_paths,
                                                           target_paths=test_target_paths,
                                                           modalities=modality,
                                                           dataset_id=dataset_id,
                                                           transforms=[ToNumpyArray(), ToNDTensor()],
                                                           augmentation_strategy=None)

        reconstruction_dataset = ABIDESegmentationFactory.create(source_paths=reconstruction_source_paths,
                                                                 target_paths=reconstruction_target_paths,
                                                                 modalities=modality,
                                                                 dataset_id=dataset_id,
                                                                 transforms=[ToNumpyArray(), ToNDTensor()],
                                                                 augmentation_strategy=None)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, train_csv

    @staticmethod
    def filter_subjects_by_sites(source_dir: str, sites: List[str]):
        return [dir for dir in sorted(os.listdir(source_dir)) if
                any(substring in dir for substring in sites)]

    @staticmethod
    def shuffle_split(subjects: np.ndarray, split_ratio: Union[float, int]):
        shuffle(subjects)
        return subjects[ceil(len(subjects) * split_ratio):], subjects[0:ceil(len(subjects) * split_ratio)]
