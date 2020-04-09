from typing import List, Callable, Union, Tuple, Optional

import numpy as np
import os
import pandas
import torch
from math import ceil
from samitorch.inputs.augmentation.strategies import DataAugmentationStrategy
from samitorch.inputs.datasets import AbstractDatasetFactory, SegmentationDataset
from samitorch.inputs.images import Modality
from samitorch.inputs.patch import CenterCoordinate, Patch
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor, PadToPatchShape
from samitorch.utils.files import extract_file_paths
from samitorch.utils.slice_builder import SliceBuilder
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose

from deepNormalize.utils.utils import natural_sort


class SliceDataset(Dataset):
    def __init__(self, source_images, target_images, patches: np.ndarray, modalities: Union[Modality, List[Modality]],
                 dataset_id: int = None, transforms: Optional[Callable] = None,
                 augment: DataAugmentationStrategy = None) -> None:
        self._source_images = source_images
        self._target_images = target_images
        self._patches = patches
        self._modalities = modalities
        self._dataset_id = dataset_id
        self._transform = transforms
        self._augment = augment

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, idx):
        patch = self._patches[idx]
        image_id = patch.image_id

        image = self._source_images[image_id]
        target = self._target_images[image_id]

        slice = patch.slice

        slice_x, slice_y = image[tuple(slice)], target[tuple(slice)]

        patch_sample = Sample(x=slice_x, y=slice_y, dataset_id=self._dataset_id, is_labeled=True)

        if self._transform is not None:
            patch_sample = self._transform(patch_sample)

        if self._augment is not None:
            patch_sample.augmented_x = self._augment(patch_sample.x)
        else:
            patch_sample.augmented_x = patch_sample.x

        return patch_sample


class iSEGSliceDatasetFactory(AbstractDatasetFactory):
    @staticmethod
    def create(source_images: np.ndarray, target_images: np.ndarray, patches: np.ndarray,
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_images is not None:

            return SliceDataset(source_images, target_images, patches, modalities, dataset_id,
                                Compose([transform for transform in
                                         transforms]) if transforms is not None else None,
                                augment=augmentation_strategy)

        else:
            raise NotImplementedError

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, max_subject: int = None, max_num_patches=None,
                          augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list) and len(modalities) > 1:
            return iSEGSliceDatasetFactory._create_multimodal_train_test(source_dir, modalities,
                                                                         dataset_id, test_size, max_subject,
                                                                         max_num_patches,
                                                                         augmentation_strategy)
        else:
            return iSEGSliceDatasetFactory._create_single_modality_train_test(source_dir, modalities,
                                                                              dataset_id, test_size, max_subject,
                                                                              max_num_patches,
                                                                              augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                                dataset_id: int, test_size: float, max_subjects: int = None, max_num_patches=None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return iSEGSliceDatasetFactory._create_multimodal_train_valid_test(source_dir, modalities,
                                                                               dataset_id, test_size,
                                                                               max_subjects,
                                                                               max_num_patches,
                                                                               augmentation_strategy)
        else:
            return iSEGSliceDatasetFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                    dataset_id, test_size,
                                                                                    max_subjects,
                                                                                    max_num_patches,
                                                                                    augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           max_subjects: int = None, max_num_patches: int = None,
                                           augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_iseg_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = iSEGSliceDatasetFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(transform(train_source_path))
            train_targets.append(transform(train_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(transform(test_source_path))
            test_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(transform(reconstruction_source_path))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = iSEGSliceDatasetFactory.get_patches(train_images, train_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        test_patches = iSEGSliceDatasetFactory.get_patches(test_images, test_targets,
                                                           (1, 32, 32, 32),
                                                           (1, 4, 4, 4),
                                                           keep_centered_on_foreground=True)
        reconstruction_patches = iSEGSliceDatasetFactory.get_patches(reconstruction_images,
                                                                     reconstruction_targets,
                                                                     (1, 32, 32, 32),
                                                                     (1, 4, 4, 4),
                                                                     keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modality,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, modalities: List[Modality], dataset_id: int, test_size: float,
                                      max_subjects: int = None, max_num_patches: int = None,
                                      augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_iseg_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = iSEGSliceDatasetFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(np.stack([transform(path) for path in train_source_path], axis=0).squeeze(1))
            train_targets.append(transform(train_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(np.stack([transform(path) for path in test_source_path], axis=0).squeeze(1))
            train_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(
                np.stack([transform(path) for path in reconstruction_source_path], axis=0).squeeze(1))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = iSEGSliceDatasetFactory.get_patches(train_images, train_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        test_patches = iSEGSliceDatasetFactory.get_patches(test_images, test_targets,
                                                           (1, 32, 32, 32),
                                                           (1, 4, 4, 4),
                                                           keep_centered_on_foreground=True)
        reconstruction_patches = iSEGSliceDatasetFactory.get_patches(reconstruction_images,
                                                                     reconstruction_targets,
                                                                     (1, 32, 32, 32),
                                                                     (1, 4, 4, 4),
                                                                     keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modalities,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                                 max_subjects: int = None, max_num_patches: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_iseg_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = iSEGSliceDatasetFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = iSEGSliceDatasetFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        valid_csv = csv[csv["subjects"].isin(valid_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        valid_source_paths, valid_target_paths = shuffle(np.array(natural_sort(list(valid_csv[str(modality)]))),
                                                         np.array(natural_sort(list(valid_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        valid_images = list()
        valid_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(transform(train_source_path))
            train_targets.append(transform(train_target_path))

        for valid_source_path, valid_target_path in zip(valid_source_paths, valid_target_paths):
            valid_images.append(transform(valid_source_path))
            valid_targets.append(transform(valid_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(transform(test_source_path))
            test_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(transform(reconstruction_source_path))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = iSEGSliceDatasetFactory.get_patches(train_images, train_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        valid_patches = iSEGSliceDatasetFactory.get_patches(valid_images, valid_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        test_patches = iSEGSliceDatasetFactory.get_patches(test_images, test_targets,
                                                           (1, 32, 32, 32),
                                                           (1, 4, 4, 4),
                                                           keep_centered_on_foreground=True)
        reconstruction_patches = iSEGSliceDatasetFactory.get_patches(reconstruction_images,
                                                                     reconstruction_targets,
                                                                     (1, 32, 32, 32),
                                                                     (1, 4, 4, 4),
                                                                     keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(valid_images),
            target_images=np.array(valid_targets),
            patches=valid_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modality,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_multimodal_train_valid_test(source_dir: str, modalities: List[Modality], dataset_id: int,
                                            test_size: float,
                                            max_subjects: int = None, max_num_patches: int = None,
                                            augmentation_strategy: DataAugmentationStrategy = None):
        csv = pandas.read_csv(os.path.join(source_dir, "output_iseg_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = iSEGSliceDatasetFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = iSEGSliceDatasetFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        valid_csv = csv[csv["subjects"].isin(valid_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        valid_images = list()
        valid_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(np.stack([transform(path) for path in train_source_path], axis=0).squeeze(1))
            train_targets.append(transform(train_target_path))

        for valid_source_path, valid_target_path in zip(valid_source_paths, valid_target_paths):
            valid_images.append(np.stack([transform(path) for path in valid_source_path], axis=0).squeeze(1))
            valid_targets.append(transform(valid_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(np.stack([transform(path) for path in test_source_path], axis=0).squeeze(1))
            train_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(
                np.stack([transform(path) for path in reconstruction_source_path], axis=0).squeeze(1))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = iSEGSliceDatasetFactory.get_patches(train_images, train_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        valid_patches = iSEGSliceDatasetFactory.get_patches(valid_images, valid_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        test_patches = iSEGSliceDatasetFactory.get_patches(test_images, test_targets,
                                                           (1, 32, 32, 32),
                                                           (1, 4, 4, 4),
                                                           keep_centered_on_foreground=True)
        reconstruction_patches = iSEGSliceDatasetFactory.get_patches(reconstruction_images,
                                                                     reconstruction_targets,
                                                                     (1, 32, 32, 32),
                                                                     (1, 4, 4, 4),
                                                                     keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(valid_patches)), int(max_num_patches * test_size),
                                       replace=False)
            valid_patches = valid_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(valid_images),
            target_images=np.array(valid_targets),
            patches=valid_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = iSEGSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modalities,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def get_patches(source_images: list, target_images: list, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], keep_centered_on_foreground: bool = False):

        patches = list()

        for i, (image, target) in enumerate(zip(source_images, target_images)):
            sample = Sample(x=image, y=target, dataset_id=None, is_labeled=True)
            slices = SliceBuilder(sample.x.shape, patch_size=patch_size, step=step).build_slices()
            for slice in slices:
                center_coordinate = CenterCoordinate(sample.x[slice], sample.y[slice])
                patches.append(Patch(slice, i, center_coordinate))

        if keep_centered_on_foreground:
            patches = list(filter(lambda patch: patch.center_coordinate.is_foreground, patches))

        return np.array(patches)

    @staticmethod
    def shuffle_split(subjects: np.ndarray, split_ratio: Union[float, int]):
        shuffle(subjects)
        return subjects[ceil(len(subjects) * split_ratio):], subjects[0:ceil(len(subjects) * split_ratio)]


class MRBrainSSliceDatasetFactory(AbstractDatasetFactory):
    @staticmethod
    def create(source_images: np.ndarray, target_images: np.ndarray, patches: np.ndarray,
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_images is not None:

            return SliceDataset(source_images, target_images, patches, modalities, dataset_id,
                                Compose([transform for transform in
                                         transforms]) if transforms is not None else None,
                                augment=augmentation_strategy)

        else:
            raise NotImplementedError

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, max_subject: int = None, max_num_patches=None,
                          augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list) and len(modalities) > 1:
            return MRBrainSSliceDatasetFactory._create_multimodal_train_test(source_dir, modalities,
                                                                             dataset_id, test_size, max_subject,
                                                                             max_num_patches,
                                                                             augmentation_strategy)
        else:
            return MRBrainSSliceDatasetFactory._create_single_modality_train_test(source_dir, modalities,
                                                                                  dataset_id, test_size, max_subject,
                                                                                  max_num_patches,
                                                                                  augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                                dataset_id: int, test_size: float, max_subjects: int = None, max_num_patches=None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return MRBrainSSliceDatasetFactory._create_multimodal_train_valid_test(source_dir, modalities,
                                                                                   dataset_id, test_size,
                                                                                   max_subjects,
                                                                                   max_num_patches,
                                                                                   augmentation_strategy)
        else:
            return MRBrainSSliceDatasetFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                        dataset_id, test_size,
                                                                                        max_subjects,
                                                                                        max_num_patches,
                                                                                        augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           max_subjects: int = None, max_num_patches: int = None,
                                           augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_mrbrains_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = MRBrainSSliceDatasetFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(transform(train_source_path))
            train_targets.append(transform(train_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(transform(test_source_path))
            test_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(transform(reconstruction_source_path))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = MRBrainSSliceDatasetFactory.get_patches(train_images, train_targets,
                                                                (1, 32, 32, 32),
                                                                (1, 4, 4, 4),
                                                                keep_centered_on_foreground=True)
        test_patches = MRBrainSSliceDatasetFactory.get_patches(test_images, test_targets,
                                                               (1, 32, 32, 32),
                                                               (1, 4, 4, 4),
                                                               keep_centered_on_foreground=True)
        reconstruction_patches = MRBrainSSliceDatasetFactory.get_patches(reconstruction_images,
                                                                         reconstruction_targets,
                                                                         (1, 32, 32, 32),
                                                                         (1, 4, 4, 4),
                                                                         keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modality,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, modalities: List[Modality], dataset_id: int, test_size: float,
                                      max_subjects: int = None, max_num_patches: int = None,
                                      augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_mrbrains_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = MRBrainSSliceDatasetFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(np.stack([transform(path) for path in train_source_path], axis=0).squeeze(1))
            train_targets.append(transform(train_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(np.stack([transform(path) for path in test_source_path], axis=0).squeeze(1))
            train_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(
                np.stack([transform(path) for path in reconstruction_source_path], axis=0).squeeze(1))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = MRBrainSSliceDatasetFactory.get_patches(train_images, train_targets,
                                                                (1, 32, 32, 32),
                                                                (1, 4, 4, 4),
                                                                keep_centered_on_foreground=True)
        test_patches = MRBrainSSliceDatasetFactory.get_patches(test_images, test_targets,
                                                               (1, 32, 32, 32),
                                                               (1, 4, 4, 4),
                                                               keep_centered_on_foreground=True)
        reconstruction_patches = MRBrainSSliceDatasetFactory.get_patches(reconstruction_images,
                                                                         reconstruction_targets,
                                                                         (1, 32, 32, 32),
                                                                         (1, 4, 4, 4),
                                                                         keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modalities,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                                 max_subjects: int = None, max_num_patches: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_mrbrains_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = MRBrainSSliceDatasetFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = MRBrainSSliceDatasetFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        valid_csv = csv[csv["subjects"].isin(valid_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        valid_source_paths, valid_target_paths = shuffle(np.array(natural_sort(list(valid_csv[str(modality)]))),
                                                         np.array(natural_sort(list(valid_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        valid_images = list()
        valid_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(transform(train_source_path))
            train_targets.append(transform(train_target_path))

        for valid_source_path, valid_target_path in zip(valid_source_paths, valid_target_paths):
            valid_images.append(transform(valid_source_path))
            valid_targets.append(transform(valid_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(transform(test_source_path))
            test_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(transform(reconstruction_source_path))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = MRBrainSSliceDatasetFactory.get_patches(train_images, train_targets,
                                                                (1, 32, 32, 32),
                                                                (1, 4, 4, 4),
                                                                keep_centered_on_foreground=True)
        valid_patches = MRBrainSSliceDatasetFactory.get_patches(valid_images, valid_targets,
                                                                (1, 32, 32, 32),
                                                                (1, 4, 4, 4),
                                                                keep_centered_on_foreground=True)
        test_patches = MRBrainSSliceDatasetFactory.get_patches(test_images, test_targets,
                                                               (1, 32, 32, 32),
                                                               (1, 4, 4, 4),
                                                               keep_centered_on_foreground=True)
        reconstruction_patches = MRBrainSSliceDatasetFactory.get_patches(reconstruction_images,
                                                                         reconstruction_targets,
                                                                         (1, 32, 32, 32),
                                                                         (1, 4, 4, 4),
                                                                         keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(valid_images),
            target_images=np.array(valid_targets),
            patches=valid_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modality,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_multimodal_train_valid_test(source_dir: str, modalities: List[Modality], dataset_id: int,
                                            test_size: float,
                                            max_subjects: int = None, max_num_patches: int = None,
                                            augmentation_strategy: DataAugmentationStrategy = None):
        csv = pandas.read_csv(os.path.join(source_dir, "output_mrbrains_images.csv"))

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = MRBrainSSliceDatasetFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = MRBrainSSliceDatasetFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), 1, replace=False)]

        train_csv = csv[csv["subjects"].isin(train_subjects)]
        valid_csv = csv[csv["subjects"].isin(valid_subjects)]
        test_csv = csv[csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        valid_images = list()
        valid_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(np.stack([transform(path) for path in train_source_path], axis=0).squeeze(1))
            train_targets.append(transform(train_target_path))

        for valid_source_path, valid_target_path in zip(valid_source_paths, valid_target_paths):
            valid_images.append(np.stack([transform(path) for path in valid_source_path], axis=0).squeeze(1))
            valid_targets.append(transform(valid_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(np.stack([transform(path) for path in test_source_path], axis=0).squeeze(1))
            train_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(
                np.stack([transform(path) for path in reconstruction_source_path], axis=0).squeeze(1))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = MRBrainSSliceDatasetFactory.get_patches(train_images, train_targets,
                                                                (1, 32, 32, 32),
                                                                (1, 4, 4, 4),
                                                                keep_centered_on_foreground=True)
        valid_patches = MRBrainSSliceDatasetFactory.get_patches(valid_images, valid_targets,
                                                                (1, 32, 32, 32),
                                                                (1, 4, 4, 4),
                                                                keep_centered_on_foreground=True)
        test_patches = MRBrainSSliceDatasetFactory.get_patches(test_images, test_targets,
                                                               (1, 32, 32, 32),
                                                               (1, 4, 4, 4),
                                                               keep_centered_on_foreground=True)
        reconstruction_patches = MRBrainSSliceDatasetFactory.get_patches(reconstruction_images,
                                                                         reconstruction_targets,
                                                                         (1, 32, 32, 32),
                                                                         (1, 4, 4, 4),
                                                                         keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(valid_patches)), int(max_num_patches * test_size),
                                       replace=False)
            valid_patches = valid_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(valid_images),
            target_images=np.array(valid_targets),
            patches=valid_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = MRBrainSSliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modalities,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def get_patches(source_images: list, target_images: list, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], keep_centered_on_foreground: bool = False):

        patches = list()

        for i, (image, target) in enumerate(zip(source_images, target_images)):
            sample = Sample(x=image, y=target, dataset_id=None, is_labeled=True)
            slices = SliceBuilder(sample.x.shape, patch_size=patch_size, step=step).build_slices()
            for slice in slices:
                center_coordinate = CenterCoordinate(sample.x[slice], sample.y[slice])
                patches.append(Patch(slice, i, center_coordinate))

        if keep_centered_on_foreground:
            patches = list(filter(lambda patch: patch.center_coordinate.is_foreground, patches))

        return np.array(patches)

    @staticmethod
    def shuffle_split(subjects: np.ndarray, split_ratio: Union[float, int]):
        shuffle(subjects)
        return subjects[ceil(len(subjects) * split_ratio):], subjects[0:ceil(len(subjects) * split_ratio)]


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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

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

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

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

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

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

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

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

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        subject_dirs = np.array(csv["subjects"].drop_duplicates().tolist())

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

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        filtered_csv = csv.loc[csv["site"].isin(sites)]

        subject_dirs = np.array(filtered_csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            assert max_subjects <= len(subject_dirs), "Too many subjects for the selected site."
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = filtered_csv.loc[filtered_csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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

        filtered_csv = csv.loc[csv["site"].isin(sites)]

        subject_dirs = np.array(filtered_csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            assert max_subjects <= len(subject_dirs), "Too many subjects for the selected site."
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = iSEGSegmentationFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = filtered_csv.loc[filtered_csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = csv[csv["subjects"].isin(reconstruction_subject)]

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


class ABIDESliceDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_images: np.ndarray, target_images: np.ndarray, patches: np.ndarray,
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_images is not None:

            return SliceDataset(source_images, target_images, patches, modalities, dataset_id,
                                Compose([transform for transform in
                                         transforms]) if transforms is not None else None,
                                augment=augmentation_strategy)

        else:
            raise NotImplementedError

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, sites: List[str] = None, max_subjects: int = None,
                          max_num_patches: int = None, augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            raise NotImplementedError("ABIDE only contain T1 modality.")
        else:
            return ABIDESliceDatasetFactory._create_single_modality_train_test(source_dir, modalities,
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
            return ABIDESliceDatasetFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                     dataset_id, test_size, sites,
                                                                                     max_subjects, max_num_patches,
                                                                                     augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           sites: List[str] = None, max_subjects: int = None,
                                           max_num_patches: int = None, augmentation_strategy=None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_abide_images.csv"))

        filtered_csv = csv.loc[csv["site"].isin(sites)]

        subject_dirs = np.array(filtered_csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            assert max_subjects <= len(subject_dirs), "Too many subjects for the selected site."
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, test_subjects = ABIDESliceDatasetFactory.shuffle_split(subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = filtered_csv[filtered_csv["subjects"].isin(reconstruction_subject)]

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

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(transform(train_source_path))
            train_targets.append(transform(train_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(transform(test_source_path))
            test_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(transform(reconstruction_source_path))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = ABIDESliceDatasetFactory.get_patches(train_images, train_targets,
                                                             (1, 32, 32, 32),
                                                             (1, 4, 4, 4),
                                                             keep_centered_on_foreground=True)

        test_patches = ABIDESliceDatasetFactory.get_patches(test_images, test_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        reconstruction_patches = ABIDESliceDatasetFactory.get_patches(reconstruction_images,
                                                                      reconstruction_targets,
                                                                      (1, 32, 32, 32),
                                                                      (1, 4, 4, 4),
                                                                      keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modality,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality,
                                                 dataset_id: int, test_size: float, sites: List[str] = None,
                                                 max_subjects: int = None, max_num_patches: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output_abide_images.csv"))

        filtered_csv = csv.loc[csv["site"].isin(sites)]

        subject_dirs = np.array(filtered_csv["subjects"].drop_duplicates().tolist())

        if max_subjects is not None:
            assert max_subjects <= len(subject_dirs), "Too many subjects for the selected site."
            choices = np.random.choice(np.arange(0, len(subject_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(subject_dirs)), len(subject_dirs), replace=False)

        subjects = np.array(subject_dirs)[choices]
        train_subjects, valid_subjects = ABIDESliceDatasetFactory.shuffle_split(subjects, test_size)
        valid_subjects, test_subjects = ABIDESliceDatasetFactory.shuffle_split(valid_subjects, test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = filtered_csv[filtered_csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = shuffle(np.array(natural_sort(list(train_csv[str(modality)]))),
                                                         np.array(natural_sort(list(train_csv["labels"]))))
        valid_source_paths, valid_target_paths = shuffle(np.array(natural_sort(list(valid_csv[str(modality)]))),
                                                         np.array(natural_sort(list(valid_csv["labels"]))))
        test_source_paths, test_target_paths = shuffle(np.array(natural_sort(list(test_csv[str(modality)]))),
                                                       np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.array(natural_sort(list(reconstruction_csv[str(modality)]))),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        transform = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=(1, 32, 32, 32), step=(1, 4, 4, 4))])

        train_images = list()
        train_targets = list()
        valid_images = list()
        valid_targets = list()
        test_images = list()
        test_targets = list()
        reconstruction_images = list()
        reconstruction_targets = list()

        for train_source_path, train_target_path in zip(train_source_paths, train_target_paths):
            train_images.append(transform(train_source_path))
            train_targets.append(transform(train_target_path))

        for valid_source_path, valid_target_path in zip(valid_source_paths, valid_target_paths):
            test_images.append(transform(valid_source_path))
            test_targets.append(transform(valid_target_path))

        for test_source_path, test_target_path in zip(test_source_paths, test_target_paths):
            test_images.append(transform(test_source_path))
            test_targets.append(transform(test_target_path))

        for reconstruction_source_path, reconstruction_target_path in zip(reconstruction_source_paths,
                                                                          reconstruction_target_paths):
            reconstruction_images.append(transform(reconstruction_source_path))
            reconstruction_targets.append(transform(reconstruction_target_path))

        train_patches = ABIDESliceDatasetFactory.get_patches(train_images, train_targets,
                                                             (1, 32, 32, 32),
                                                             (1, 4, 4, 4),
                                                             keep_centered_on_foreground=True)

        valid_patches = ABIDESliceDatasetFactory.get_patches(valid_images, valid_targets,
                                                             (1, 32, 32, 32),
                                                             (1, 4, 4, 4),
                                                             keep_centered_on_foreground=True)

        test_patches = ABIDESliceDatasetFactory.get_patches(test_images, test_targets,
                                                            (1, 32, 32, 32),
                                                            (1, 4, 4, 4),
                                                            keep_centered_on_foreground=True)
        reconstruction_patches = ABIDESliceDatasetFactory.get_patches(reconstruction_images,
                                                                      reconstruction_targets,
                                                                      (1, 32, 32, 32),
                                                                      (1, 4, 4, 4),
                                                                      keep_centered_on_foreground=False)

        if max_num_patches is not None:
            choices = np.random.choice(np.arange(0, len(train_patches)), max_num_patches, replace=False)
            train_patches = train_patches[choices]
            choices = np.random.choice(np.arange(0, len(valid_patches)), int(max_num_patches * test_size),
                                       replace=False)
            valid_patches = valid_patches[choices]
            choices = np.random.choice(np.arange(0, len(test_patches)), int(max_num_patches * test_size), replace=False)
            test_patches = test_patches[choices]

        train_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(train_images),
            target_images=np.array(train_targets),
            patches=train_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=valid_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        test_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(test_images),
            target_images=np.array(test_targets),
            patches=test_patches,
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        reconstruction_dataset = ABIDESliceDatasetFactory.create(
            source_images=np.array(reconstruction_images),
            target_images=np.array(reconstruction_targets),
            patches=reconstruction_patches,
            dataset_id=dataset_id,
            modalities=modality,
            transforms=[ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset

    @staticmethod
    def filter_subjects_by_sites(source_dir: str, sites: List[str]):
        return [dir for dir in sorted(os.listdir(source_dir)) if
                any(substring in dir for substring in sites)]

    @staticmethod
    def shuffle_split(subjects: np.ndarray, split_ratio: Union[float, int]):
        shuffle(subjects)
        return subjects[ceil(len(subjects) * split_ratio):], subjects[0:ceil(len(subjects) * split_ratio)]

    @staticmethod
    def get_patches(source_images: list, target_images: list, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], keep_centered_on_foreground: bool = False):

        patches = list()

        for i, (image, target) in enumerate(zip(source_images, target_images)):
            sample = Sample(x=image, y=target, dataset_id=None, is_labeled=True)
            slices = SliceBuilder(sample.x.shape, patch_size=patch_size, step=step).build_slices()
            for slice in slices:
                center_coordinate = CenterCoordinate(sample.x[slice], sample.y[slice])
                patches.append(Patch(slice, i, center_coordinate))

        if keep_centered_on_foreground:
            patches = list(filter(lambda patch: patch.center_coordinate.is_foreground, patches))

        return np.array(patches)
