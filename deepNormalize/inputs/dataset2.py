import copy
import os
from typing import Tuple, Callable

import numpy as np
import re
from samitorch.inputs.datasets import AbstractDatasetFactory, SegmentationDataset, MultimodalSegmentationDataset
from samitorch.inputs.images import Modality
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor, PadToPatchShape
from samitorch.utils.files import extract_file_paths
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torchvision.transforms import Compose

from samitorch.inputs.datasets import PatchDataset
from samitorch.utils.slice_builder import SliceBuilder
from samitorch.inputs.patch import Patch, CenterCoordinate


class iSEGPatchDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_test(source_dir: str, target_dir: str, modality: Modality, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], dataset_id: int, keep_centered_on_foreground: bool = True):
        """
        Create a PatchDataset object for both training and validation.

        Args:
            source_dir (str): Path to source directory.
            target_dir (str): Path to target directory.
            modality (:obj:`samitorch.inputs.images.Modalities`): The modality of the data set.
            patch_size (Tuple of int): A tuple representing the desired patch size.
            step (Tuple of int): A tuple representing the desired step between two patches.
            dataset_id (int): An integer representing the ID of the data set.
            test_size (float): The size in percentage of the validation set over total number of samples.
            keep_centered_on_foreground (bool): Keep only patches which center coordinates belongs to a foreground class.

        Returns:
            Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """

        source_dir = os.path.join(source_dir, str(modality))
        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

        patches = iSEGPatchDatasetFactory.get_patches(source_paths, target_paths, patch_size, step, transforms,
                                                  keep_centered_on_foreground)
        label_patches = copy.deepcopy(patches)

        test_samples = list(
            map(lambda source, target: Sample(source, target, dataset_id=dataset_id, is_labeled=True),
                patches, label_patches))

        test_dataset = PatchDataset(list(source_paths), list(target_paths), test_samples, patch_size, step, modality,
                                    dataset_id, Compose([ToNDTensor()]))

        return test_dataset

    @staticmethod
    def get_patches(source_paths: np.ndarray, target_paths: np.ndarray, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], transforms: Callable, keep_centered_on_foreground: bool = False):

        patches = list()

        for idx in range(len(source_paths)):
            source_path, target_path = source_paths[idx], target_paths[idx]
            sample = Sample(x=source_path, y=target_path, dataset_id=None, is_labeled=True)
            transformed_sample = transforms(sample)
            slices = SliceBuilder(transformed_sample.x.shape, patch_size=patch_size, step=step).build_slices()
            for slice in slices:
                if np.count_nonzero(transformed_sample.x[slice]) > 0:
                    center_coordinate = CenterCoordinate(transformed_sample.x[slice], transformed_sample.y[slice])
                    patches.append(
                        Patch(slice, idx, center_coordinate))
                else:
                    pass

        if keep_centered_on_foreground:
            patches = list(filter(lambda patch: patch.center_coordinate.is_foreground, patches))

        return np.array(patches)


class MRBrainSPatchDatasetFactory(AbstractDatasetFactory):

    @staticmethod
    def create_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
                    patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], keep_centered_on_foreground: bool = True):
        """
        Create a SegmentationDataset object for both training and validation.

        Args:
           source_dir (str): Path to source directory.
           target_dir (str): Path to target directory.
           modality (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
           dataset_id (int): An integer representing the ID of the data set.
           test_size (float): The size in percentage of the validation set over total number of samples.

        Returns:
           Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        source_dir = os.path.join(os.path.join(source_dir, "*"))
        target_dir = os.path.join(os.path.join(target_dir, "*"))

        source_paths, target_paths = (extract_file_paths(source_dir)), (extract_file_paths(target_dir))

        images = np.array(list(filter(re.compile(r".*T1_1mm.*\.nii").search, source_paths)))
        targets = np.array(list(filter(re.compile(r".*LabelsForTesting.*\.nii").search, target_paths)))

        transforms = Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

        patches = MRBrainSPatchDatasetFactory.get_patches(images, targets, patch_size, step, transforms,
                                                          keep_centered_on_foreground)
        label_patches = copy.deepcopy(patches)

        test_samples = list(
            map(lambda source, target: Sample(source, target, dataset_id=dataset_id, is_labeled=True),
                patches, label_patches))

        test_dataset = PatchDataset(list(source_paths), list(target_paths), test_samples, patch_size, step, modality,
                                    dataset_id, Compose([ToNDTensor()]))

        return test_dataset

    @staticmethod
    def get_patches(source_paths: np.ndarray, target_paths: np.ndarray, patch_size: Tuple[int, int, int, int],
                    step: Tuple[int, int, int, int], transforms: Callable, keep_centered_on_foreground: bool = False):

        patches = list()

        for idx in range(len(source_paths)):
            source_path, target_path = source_paths[idx], target_paths[idx]
            sample = Sample(x=source_path, y=target_path, dataset_id=None, is_labeled=True)
            transformed_sample = transforms(sample)
            slices = SliceBuilder(transformed_sample.x.shape, patch_size=patch_size, step=step).build_slices()
            for slice in slices:
                if np.count_nonzero(transformed_sample.x[slice]) > 0:
                    center_coordinate = CenterCoordinate(transformed_sample.x[slice], transformed_sample.y[slice])
                    patches.append(
                        Patch(slice, idx, center_coordinate))
                else:
                    pass

        if keep_centered_on_foreground:
            patches = list(filter(lambda patch: patch.center_coordinate.is_foreground, patches))

        return np.array(patches)


class MRBrainSSegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int, test_size: float):
        """
        Create a SegmentationDataset object for both training and validation.

        Args:
           source_dir (str): Path to source directory.
           target_dir (str): Path to target directory.
           modality (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
           dataset_id (int): An integer representing the ID of the data set.
           test_size (float): The size in percentage of the validation set over total number of samples.

        Returns:
           Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        source_dir = os.path.join(os.path.join(source_dir, "*"), "T1_1mm")
        target_dir = os.path.join(os.path.join(target_dir, "*"), "LabelsForTesting")

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        train_ids, test_ids = next(ShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, target_paths))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_ids],
                target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[test_ids],
                target_paths[test_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id, Compose([ToNumpyArray(), ToNDTensor()]))

        test_dataset = SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([ToNumpyArray(), ToNDTensor()]))

        return training_dataset, test_dataset

    @staticmethod
    def create_multimodal_train_test(source_dir: str, target_dir: str, modality_1: Modality, modality_2: Modality,
                                     dataset_id: int, test_size: float):
        """
        Create a MultimodalDataset object for both training and validation.

        Args:
           source_dir (str): Path to source directory.
           target_dir (str): Path to target directory.
           modality_1 (:obj:`samitorch.inputs.images.Modalities`): The first modality of the data set.
           modality_2 (:obj:`samitorch.inputs.images.Modalities`): The second modality of the data set.
           dataset_id (int): An integer representing the ID of the data set.
           test_size (float): The size in percentage of the validation set over total number of samples.

        Returns:
           Tuple of :obj:`torch.utils.data.dataset`: A tuple containing both training and validation dataset.
        """
        source_dir_modality_1 = os.path.join(source_dir, str(modality_1)) + "/*"
        source_dir_modality_2 = os.path.join(source_dir, str(modality_2)) + "/*"

        source_paths_modality_1, target_paths = np.array(extract_file_paths(source_dir_modality_1)), np.array(
            extract_file_paths(target_dir))
        source_paths_modality_2, target_paths = np.array(extract_file_paths(source_dir_modality_2)), np.array(
            extract_file_paths(target_dir))

        source_paths = np.stack((source_paths_modality_1, source_paths_modality_2), axis=1)

        train_ids, test_ids = next(ShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, target_paths))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[train_ids],
                target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True), source_paths[test_ids],
                target_paths[test_ids]))

        training_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), train_samples,
                                                         modality_1.value, modality_2.value, dataset_id,
                                                         Compose([ToNumpyArray(), ToNDTensor()]))

        test_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), test_samples,
                                                     modality_1.value, modality_2.value, dataset_id,
                                                     Compose([ToNumpyArray(), ToNDTensor()]))

        return training_dataset, test_dataset
