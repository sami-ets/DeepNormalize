from typing import List, Optional, Callable

import numpy as np
import os
import pandas
from samitorch.inputs.augmentation.strategies import DataAugmentationStrategy
from samitorch.inputs.datasets import AbstractDatasetFactory, SegmentationDataset, MultimodalSegmentationDataset
from samitorch.inputs.images import Modality
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor
from samitorch.utils.files import extract_file_paths
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose


class TestDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], samples: List[Sample], modality: Modality,
                 dataset_id: int = None, transforms: Optional[Callable] = None) -> None:
        """
        Dataset initializer.

        Args:
            source_paths (List of str): Path to source images.
            target_paths (List of str): Path to target (labels) images.
            samples (list of :obj:`samitorch.inputs.sample.Sample`): A list of Sample objects.
            modality (:obj:`samitorch.inputs.images.Modalities`): The modality of the data set.
            dataset_id (int): An integer representing the ID of the data set.
            transforms (Callable): transform to apply to both source and target images.
        """
        self._source_paths = source_paths
        self._samples = samples
        self._modality = modality
        self._dataset_id = dataset_id
        self._transform = transforms

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        sample = self._samples[idx]

        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class iSEGSegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
               transforms: List[Callable] = None, augmentation_strategy: DataAugmentationStrategy = None):

        if target_dir is not None:
            source_paths, target_paths = extract_file_paths(source_dir), extract_file_paths(target_dir)

            source_paths = np.array(sorted(source_paths,
                                           key=lambda x: int(
                                               os.path.splitext(os.path.splitext(os.path.basename(x))[0])[0])))
            target_paths = np.array(sorted(target_paths,
                                           key=lambda x: int(
                                               os.path.splitext(os.path.splitext(os.path.basename(x))[0])[0])))

            test_samples = list(
                map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id), source_paths,
                    target_paths))

            if transforms is not None:
                return SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([transform for transform in transforms]))
            else:
                return SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id)
        else:
            source_paths = extract_file_paths(source_dir)

            source_paths = np.array(sorted(source_paths,
                                           key=lambda x: int(
                                               os.path.splitext(os.path.splitext(os.path.basename(x))[0])[0])))
            test_samples = list(
                map(lambda source: Sample(source, None, is_labeled=False, dataset_id=dataset_id), source_paths))

            if transforms is not None:
                return SegmentationDataset(list(source_paths), None, test_samples, modality, dataset_id,
                                           Compose([transform for transform in transforms]), augmentation_strategy)
            else:
                return SegmentationDataset(list(source_paths), None, test_samples, modality, dataset_id,
                                           augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int, test_size: float):
        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_dir = os.path.join(source_dir, str(modality)) + "/*"

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(
            extract_file_paths(target_dir + "/*"))

        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(csv["center_class"])))
        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_ids], target_paths[train_ids]))
        valid_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[valid_ids], target_paths[valid_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id, Compose([ToNumpyArray(), ToNDTensor()]))

        valid_dataset = SegmentationDataset(list(source_paths), list(target_paths), valid_samples, modality,
                                            dataset_id, Compose([ToNumpyArray(), ToNDTensor()]))

        return training_dataset, valid_dataset, csv

    @staticmethod
    def create_train_valid_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
                                test_size: float, augmentation_strategy: DataAugmentationStrategy = None):
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
        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_dir = os.path.join(source_dir, str(modality)) + "/*"

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(
            extract_file_paths(target_dir + "/*"))

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(csv["center_class"])))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          np.asarray(csv["center_class"])[
                                                                              train_valid_ids]))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][train_ids], target_paths[train_valid_ids][train_ids]))
        valid_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][valid_ids], target_paths[train_valid_ids][valid_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[test_ids], target_paths[test_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id, Compose([ToNumpyArray(), ToNDTensor()]),
                                               augmentation_strategy)
        valid_dataset = SegmentationDataset(list(source_paths), list(target_paths), valid_samples, modality,
                                            dataset_id, Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        test_dataset = SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        return training_dataset, valid_dataset, test_dataset, csv

    @staticmethod
    def create_multimodal_train_valid_test(source_dir: str, target_dir: str, modality_1: Modality, modality_2: Modality,
                                           dataset_id: int, test_size: float,
                                           augmentation_strategy: DataAugmentationStrategy = None):
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
        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_dir_modality_1 = os.path.join(source_dir, str(modality_1)) + "/*"
        source_dir_modality_2 = os.path.join(source_dir, str(modality_2)) + "/*"

        source_paths_modality_1, target_paths = np.array(extract_file_paths(source_dir_modality_1)), np.array(
            extract_file_paths(target_dir))
        source_paths_modality_2, target_paths = np.array(extract_file_paths(source_dir_modality_2)), np.array(
            extract_file_paths(target_dir))

        source_paths = np.stack((source_paths_modality_1, source_paths_modality_2), axis=1)

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, csv["center_class"]))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          csv["center_class"][train_valid_ids]))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][train_ids], target_paths[train_valid_ids][train_ids]))
        valid_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][valid_ids], target_paths[train_valid_ids][valid_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[test_ids], target_paths[test_ids]))

        training_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), train_samples,
                                                         modality_1.value, modality_2.value, dataset_id,
                                                         Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)
        valid_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), valid_samples,
                                                      modality_1.value, modality_2.value, dataset_id,
                                                      Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)
        test_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), test_samples,
                                                     modality_1.value, modality_2.value, dataset_id,
                                                     Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        return training_dataset, valid_dataset, test_dataset, csv


class MRBrainSSegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create_train_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
                          test_size: float, augmentation_strategy: DataAugmentationStrategy = None):
        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_dir = os.path.join(os.path.join(source_dir, "*"), "T1_1mm")
        target_dir = os.path.join(os.path.join(target_dir, "*"), "LabelsForTesting")

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, csv["center_class"]))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_ids], target_paths[train_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[test_ids], target_paths[test_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id, Compose([ToNumpyArray(), ToNDTensor()]),
                                               augmentation_strategy)

        test_dataset = SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        return training_dataset, test_dataset, csv

    @staticmethod
    def create(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
               transforms: List[Callable] = None):

        source_paths, target_paths = extract_file_paths(source_dir), extract_file_paths(target_dir)

        source_paths = np.array(sorted(source_paths,
                                       key=lambda x: int(
                                           os.path.splitext(os.path.splitext(os.path.basename(x))[0])[0])))
        target_paths = np.array(sorted(target_paths,
                                       key=lambda x: int(
                                           os.path.splitext(os.path.splitext(os.path.basename(x))[0])[0])))

        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id), source_paths,
                target_paths))

        if transforms is not None:
            return SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                       Compose([transform for transform in transforms]))
        else:
            return SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id)

    @staticmethod
    def create_train_valid_test(source_dir: str, target_dir: str, modality: Modality, dataset_id: int,
                                test_size: float, augmentation_strategy: DataAugmentationStrategy = None):
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
        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_dir = os.path.join(os.path.join(source_dir, "*"), "T1_1mm")
        target_dir = os.path.join(os.path.join(target_dir, "*"), "LabelsForTesting")

        source_paths, target_paths = np.array(extract_file_paths(source_dir)), np.array(extract_file_paths(target_dir))

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, csv["center_class"]))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          csv["center_class"][train_valid_ids]))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][train_ids], target_paths[train_valid_ids][train_ids]))
        valid_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][valid_ids], target_paths[train_valid_ids][valid_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[test_ids], target_paths[test_ids]))

        training_dataset = SegmentationDataset(list(source_paths), list(target_paths), train_samples, modality,
                                               dataset_id, Compose([ToNumpyArray(), ToNDTensor()]),
                                               augmentation_strategy)

        valid_dataset = SegmentationDataset(list(source_paths), list(target_paths), valid_samples, modality,
                                            dataset_id, Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        test_dataset = SegmentationDataset(list(source_paths), list(target_paths), test_samples, modality, dataset_id,
                                           Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        return training_dataset, valid_dataset, test_dataset, csv

    @staticmethod
    def create_multimodal_train_valid_test(source_dir: str, target_dir: str, modality_1: Modality, modality_2: Modality,
                                           dataset_id: int, test_size: float,
                                           augmentation_strategy: DataAugmentationStrategy = None):
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
        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_dir_modality_1 = os.path.join(source_dir, str(modality_1)) + "/*"
        source_dir_modality_2 = os.path.join(source_dir, str(modality_2)) + "/*"

        source_paths_modality_1, target_paths = np.array(extract_file_paths(source_dir_modality_1)), np.array(
            extract_file_paths(target_dir))
        source_paths_modality_2, target_paths = np.array(extract_file_paths(source_dir_modality_2)), np.array(
            extract_file_paths(target_dir))

        source_paths = np.stack((source_paths_modality_1, source_paths_modality_2), axis=1)

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, csv["center_class"]))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          csv["center_class"][train_valid_ids]))

        train_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][train_ids], target_paths[train_valid_ids][train_ids]))
        valid_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[train_valid_ids][valid_ids], target_paths[train_valid_ids][valid_ids]))
        test_samples = list(
            map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                source_paths[test_ids], target_paths[test_ids]))

        training_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), train_samples,
                                                         modality_1.value, modality_2.value, dataset_id,
                                                         Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        valid_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), valid_samples,
                                                      modality_1.value, modality_2.value, dataset_id,
                                                      Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        test_dataset = MultimodalSegmentationDataset(list(source_paths), list(target_paths), test_samples,
                                                     modality_1.value, modality_2.value, dataset_id,
                                                     Compose([ToNumpyArray(), ToNDTensor()]), augmentation_strategy)

        return training_dataset, valid_dataset, test_dataset, csv
