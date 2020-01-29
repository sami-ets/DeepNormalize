from typing import List, Optional, Callable, Union

import numpy as np
import os
import pandas
from samitorch.inputs.augmentation.strategies import DataAugmentationStrategy
from samitorch.inputs.datasets import AbstractDatasetFactory, SegmentationDataset
from samitorch.inputs.images import Modality
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor
from samitorch.utils.files import extract_file_paths
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose
from deepNormalize.utils.utils import natural_sort


class TestDataset(Dataset):
    """
    Create a dataset class in PyTorch for reading NIfTI files.
    """

    def __init__(self, source_paths: List[str], samples: List[Sample], modality: Modality,
                 dataset_id: int = None, transforms: Optional[Callable] = None) -> None:
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
    def create(source_paths: Union[List[str], np.ndarray], target_paths: Union[List[str], np.ndarray],
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_paths is not None:
            samples = list(map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                               source_paths, target_paths))

            return SegmentationDataset(list(source_paths), list(target_paths), samples, modalities, dataset_id,
                                       Compose(
                                           [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

        else:
            samples = list(
                map(lambda source: Sample(source, None, is_labeled=False, dataset_id=dataset_id), source_paths))

            return SegmentationDataset(list(source_paths), None, samples, modalities, dataset_id, Compose(
                [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, max_subject: int = None,
                          augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list) and len(modalities) > 1:
            return iSEGSegmentationFactory._create_multimodal_train_test(source_dir, modalities,
                                                                         dataset_id, test_size, max_subject,
                                                                         augmentation_strategy)
        else:
            return iSEGSegmentationFactory._create_single_modality_train_test(source_dir, modalities,
                                                                              dataset_id, test_size, max_subject,
                                                                              augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                                dataset_id: int, test_size: float, max_subjects: int = None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return iSEGSegmentationFactory._create_multimodal_train_valid_test(source_dir, modalities,
                                                                               dataset_id, test_size, max_subjects,
                                                                               augmentation_strategy)
        else:
            return iSEGSegmentationFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                    dataset_id, test_size, max_subjects,
                                                                                    augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           max_subjects: int = None,
                                           augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = [os.path.join(source_dir, str(modality), dir) for dir in
                    sorted(os.listdir(os.path.join(source_dir, str(modality))))]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), str(choices + 1)))]

        source_paths = np.array(sorted(list(filtered_csv[str(modality)])))
        target_paths = np.array(sorted(list(filtered_csv["labels"])))

        reconstruction_source_paths = extract_file_paths(
            os.path.join(source_dir, str(modality), str(reconstruction_choice)))
        reconstruction_target_paths = extract_file_paths(
            os.path.join(source_dir, "label", str(reconstruction_choice))) if os.path.exists(
            os.path.join(source_dir, "label", str(reconstruction_choice))) else None

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))

        train_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[train_ids],
                                                       target_paths=np.array(natural_sort(target_paths))[train_ids],
                                                       modalities=modality,
                                                       dataset_id=dataset_id,
                                                       transforms=[ToNumpyArray(), ToNDTensor()],
                                                       augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[test_ids],
                                                      target_paths=np.array(natural_sort(target_paths))[test_ids],
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(reconstruction_source_paths)),
            target_paths=np.array(natural_sort(reconstruction_target_paths)),
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, test_dataset, reconstruction_dataset, filtered_csv

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                                 max_subjects: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = [os.path.join(source_dir, str(modality), dir) for dir in
                    sorted(os.listdir(os.path.join(source_dir, str(modality))))]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[
            filtered_csv[str(modality)].str.match("{}/{}/{}".format(source_dir, str(modality), str(choices + 1)))]

        source_paths = np.array(sorted(list(filtered_csv[str(modality)])))
        target_paths = np.array(sorted(list(filtered_csv["labels"])))

        reconstruction_source_paths = extract_file_paths(
            os.path.join(source_dir, str(modality), str(reconstruction_choice)))
        reconstruction_target_paths = extract_file_paths(
            os.path.join(source_dir, "label", str(reconstruction_choice))) if os.path.exists(
            os.path.join(source_dir, "label", str(reconstruction_choice))) else None

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          np.asarray(filtered_csv["center_class"])[
                                                                              train_valid_ids]))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(source_paths))[train_valid_ids][train_ids],
            target_paths=np.array(natural_sort(target_paths))[train_valid_ids][train_ids],
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(source_paths))[train_valid_ids][valid_ids],
            target_paths=np.array(natural_sort(target_paths))[train_valid_ids][valid_ids],
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[test_ids],
                                                      target_paths=np.array(natural_sort(target_paths))[test_ids],
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(reconstruction_source_paths)),
            target_paths=np.array(natural_sort(reconstruction_target_paths)),
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, modalities: List[Modality], dataset_id: int, test_size: float,
                                      max_subjects: int = None,
                                      augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = [os.path.join(source_dir, str(modalities[0]), dir) for dir in
                    sorted(os.listdir(os.path.join(source_dir, str(modalities[0]))))]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), str(choices + 1)))]

        source_paths = np.stack([sorted(list(filtered_csv[str(modality)])) for modality in modalities], axis=1)
        target_paths = np.array(sorted(list(filtered_csv["labels"])))

        reconstruction_source_paths = np.array(list(map(lambda modality: extract_file_paths(
            os.path.join(source_dir, str(modality), str(reconstruction_choice))), modalities)))
        reconstruction_source_paths = np.stack(reconstruction_source_paths, axis=1)

        reconstruction_target_paths = np.array(extract_file_paths(
            os.path.join(source_dir, "label", str(reconstruction_choice)))) if os.path.exists(
            os.path.join(source_dir, "label", str(reconstruction_choice))) else None

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))

        train_dataset = iSEGSegmentationFactory.create(source_paths=np.array(source_paths)[train_ids],
                                                       target_paths=np.array(target_paths)[train_ids],
                                                       modalities=modalities,
                                                       dataset_id=dataset_id,
                                                       transforms=[ToNumpyArray(), ToNDTensor()],
                                                       augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(source_paths)[test_ids],
                                                      target_paths=np.array(target_paths)[test_ids],
                                                      modalities=modalities,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(reconstruction_source_paths),
            target_paths=np.array(reconstruction_target_paths),
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, test_dataset, reconstruction_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_valid_test(source_dir: str, modalities: List[Modality],
                                            dataset_id: int, test_size: float, max_subjects: int = None,
                                            augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = [os.path.join(source_dir, str(modalities[0]), dir) for dir in
                    sorted(os.listdir(os.path.join(source_dir, str(modalities[0]))))]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[
            filtered_csv[str(modalities[0])].str.match(
                "{}/{}/{}".format(source_dir, str(modalities[0]), str(choices + 1)))]

        source_paths = np.stack([sorted(list(filtered_csv[str(modality)])) for modality in modalities], axis=1)
        target_paths = np.array(sorted(list(filtered_csv["labels"])))

        reconstruction_source_paths = np.array(list(map(lambda modality: extract_file_paths(
            os.path.join(source_dir, str(modality), str(reconstruction_choice))), modalities)))
        reconstruction_source_paths = np.stack(reconstruction_source_paths, axis=1)

        reconstruction_target_paths = np.array(extract_file_paths(
            os.path.join(source_dir, "label", str(reconstruction_choice)))) if os.path.exists(
            os.path.join(source_dir, "label", str(reconstruction_choice))) else None

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          np.asarray(filtered_csv["center_class"])[
                                                                              train_valid_ids]))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(source_paths)[train_valid_ids][train_ids],
            target_paths=np.array(target_paths)[train_valid_ids][train_ids],
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(source_paths)[train_valid_ids][valid_ids],
            target_paths=np.array(target_paths)[train_valid_ids][valid_ids],
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(source_paths)[test_ids],
                                                      target_paths=np.array(target_paths)[test_ids],
                                                      modalities=modalities,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(reconstruction_source_paths),
            target_paths=np.array(reconstruction_target_paths),
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, filtered_csv


class MRBrainSSegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_paths: Union[List[str], np.ndarray], target_paths: Union[List[str], np.ndarray],
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_paths is not None:
            samples = list(map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                               source_paths, target_paths))

            return SegmentationDataset(list(source_paths), list(target_paths), samples, modalities, dataset_id,
                                       Compose(
                                           [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

        else:
            samples = list(
                map(lambda source: Sample(source, None, is_labeled=False, dataset_id=dataset_id), source_paths))

            return SegmentationDataset(list(source_paths), None, samples, modalities, dataset_id, Compose(
                [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int,
                          test_size: float, augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return MRBrainSSegmentationFactory._create_multimodal_train_test(source_dir, modalities,
                                                                             dataset_id, test_size,
                                                                             augmentation_strategy)
        else:
            return MRBrainSSegmentationFactory._create_single_modality_train_test(source_dir, modalities,
                                                                                  dataset_id, test_size,
                                                                                  augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]], dataset_id: int,
                                test_size: float, max_subjects: int = None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            return MRBrainSSegmentationFactory._create_multimodal_train_valid_test(source_dir, modalities,
                                                                                   dataset_id, test_size, max_subjects,
                                                                                   augmentation_strategy)
        else:
            return MRBrainSSegmentationFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                        dataset_id, test_size,
                                                                                        max_subjects,
                                                                                        augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int,
                                           test_size: float, max_subjects: int = None,
                                           augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        source_dirs = np.array(sorted(all_dirs))[choices]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, source_dirs))]

        source_paths = np.array(sorted(list(filtered_csv[str(modality)])))
        target_paths = np.array(sorted(list(filtered_csv["LabelsForTesting"])))

        reconstruction_source_paths = extract_file_paths(
            os.path.join(source_dir, str(reconstruction_choice), str(modality)))
        reconstruction_target_paths = extract_file_paths(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting")) if os.path.exists(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting")) else None

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))

        train_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[train_ids],
                                                       target_paths=np.array(natural_sort(target_paths))[train_ids],
                                                       modalities=modality,
                                                       dataset_id=dataset_id,
                                                       transforms=[ToNumpyArray(), ToNDTensor()],
                                                       augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[test_ids],
                                                      target_paths=np.array(natural_sort(target_paths))[test_ids],
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(reconstruction_source_paths)),
            target_paths=np.array(natural_sort(reconstruction_target_paths)),
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, test_dataset, reconstruction_dataset, filtered_csv

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality, dataset_id: int,
                                                 test_size: float, max_subjects: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        source_dirs = np.array(sorted(all_dirs))[choices]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[filtered_csv[str(modality)].str.match("{}/{}".format(source_dir, source_dirs))]

        source_paths = np.array(sorted(list(filtered_csv[str(modality)])))
        target_paths = np.array(sorted(list(filtered_csv["LabelsForTesting"])))

        reconstruction_source_paths = extract_file_paths(
            os.path.join(source_dir, str(reconstruction_choice), str(modality)))
        reconstruction_target_paths = extract_file_paths(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting")) if os.path.exists(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting")) else None

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          np.asarray(filtered_csv["center_class"])[
                                                                              train_valid_ids]))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(source_paths))[train_valid_ids][train_ids],
            target_paths=np.array(natural_sort(target_paths))[train_valid_ids][train_ids],
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(source_paths))[train_valid_ids][valid_ids],
            target_paths=np.array(natural_sort(target_paths))[train_valid_ids][valid_ids],
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[test_ids],
                                                      target_paths=np.array(natural_sort(target_paths))[test_ids],
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(reconstruction_source_paths)),
            target_paths=np.array(natural_sort(reconstruction_target_paths)),
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_test(source_dir: str, modalities: List[Modality],
                                      dataset_id: int, test_size: float, max_subjects: int = None,
                                      augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        source_dirs = np.array(sorted(all_dirs))[choices]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, source_dirs))]

        source_paths = np.stack([sorted(list(filtered_csv[str(modality)])) for modality in modalities], axis=1)
        target_paths = np.array(sorted(list(filtered_csv["LabelsForTesting"])))

        reconstruction_source_paths = np.array(list(map(lambda modality: extract_file_paths(
            os.path.join(source_dir, str(modality), str(reconstruction_choice))), modalities)))
        reconstruction_source_paths = np.stack(reconstruction_source_paths, axis=1)

        reconstruction_target_paths = np.array(extract_file_paths(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting"))) if os.path.exists(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting")) else None

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))

        train_dataset = iSEGSegmentationFactory.create(source_paths=np.array(source_paths)[train_ids],
                                                       target_paths=np.array(target_paths)[train_ids],
                                                       modalities=modalities,
                                                       dataset_id=dataset_id,
                                                       transforms=[ToNumpyArray(), ToNDTensor()],
                                                       augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(source_paths)[test_ids],
                                                      target_paths=np.array(target_paths)[test_ids],
                                                      modalities=modalities,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(reconstruction_source_paths),
            target_paths=np.array(reconstruction_target_paths),
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, test_dataset, reconstruction_dataset, filtered_csv

    @staticmethod
    def _create_multimodal_train_valid_test(source_dir: str, modalities: List[Modality],
                                            dataset_id: int, test_size: float, max_subjects: int = None,
                                            augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        all_dirs = next(os.walk(source_dir))[1]

        if max_subjects is not None:
            choices = np.random.choice(np.arange(0, len(all_dirs)), max_subjects, replace=False)
        else:
            choices = np.random.choice(np.arange(0, len(all_dirs)), len(all_dirs) - 1, replace=False)
        reconstruction_choice = np.random.choice(np.setdiff1d(np.arange(1, len(all_dirs) + 1), choices), replace=False)

        source_dirs = np.array(sorted(all_dirs))[choices]

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]
        filtered_csv = filtered_csv[filtered_csv[str(modalities[0])].str.match("{}/{}".format(source_dir, source_dirs))]

        source_paths = np.stack([sorted(list(filtered_csv[str(modality)])) for modality in modalities], axis=1)
        target_paths = np.array(sorted(list(filtered_csv["LabelsForTesting"])))

        reconstruction_source_paths = np.array(list(map(lambda modality: extract_file_paths(
            os.path.join(source_dir, str(modality), str(reconstruction_choice))), modalities)))
        reconstruction_source_paths = np.stack(reconstruction_source_paths, axis=1)

        reconstruction_target_paths = np.array(extract_file_paths(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting"))) if os.path.exists(
            os.path.join(source_dir, str(reconstruction_choice), "LabelsForTesting")) else None

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          np.asarray(filtered_csv["center_class"])[
                                                                              train_valid_ids]))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(source_paths)[train_valid_ids][train_ids],
            target_paths=np.array(target_paths)[train_valid_ids][train_ids],
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(source_paths)[train_valid_ids][valid_ids],
            target_paths=np.array(target_paths)[train_valid_ids][valid_ids],
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(source_paths)[test_ids],
                                                      target_paths=np.array(target_paths)[test_ids],
                                                      modalities=modalities,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        reconstruction_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(reconstruction_source_paths),
            target_paths=np.array(reconstruction_target_paths),
            modalities=modalities,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        return train_dataset, valid_dataset, test_dataset, reconstruction_dataset, filtered_csv


class ABIDESegmentationFactory(AbstractDatasetFactory):

    @staticmethod
    def create(source_paths: Union[List[str], np.ndarray], target_paths: Union[List[str], np.ndarray],
               modalities: Union[Modality, List[Modality]], dataset_id: int, transforms: List[Callable] = None,
               augmentation_strategy: DataAugmentationStrategy = None):

        if target_paths is not None:
            samples = list(map(lambda source, target: Sample(source, target, is_labeled=True, dataset_id=dataset_id),
                               source_paths, target_paths))

            return SegmentationDataset(list(source_paths), list(target_paths), samples, modalities, dataset_id,
                                       Compose(
                                           [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

        else:
            samples = list(
                map(lambda source: Sample(source, None, is_labeled=False, dataset_id=dataset_id), source_paths))

            return SegmentationDataset(list(source_paths), None, samples, modalities, dataset_id, Compose(
                [transform for transform in transforms]) if transforms is not None else None,
                                       augment=augmentation_strategy)

    @staticmethod
    def create_train_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                          dataset_id: int, test_size: float, sites: List[str] = None, max_subjects: int = None,
                          augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            raise NotImplementedError("ABIDE only contain T1 modality.")
        else:
            return ABIDESegmentationFactory._create_single_modality_train_test(source_dir, modalities,
                                                                               dataset_id, test_size, sites,
                                                                               max_subjects,
                                                                               augmentation_strategy)

    @staticmethod
    def create_train_valid_test(source_dir: str, modalities: Union[Modality, List[Modality]],
                                dataset_id: int, test_size: float, sites: List[str] = None, max_subjects: int = None,
                                augmentation_strategy: DataAugmentationStrategy = None):

        if isinstance(modalities, list):
            raise NotImplementedError("ABIDE only contain T1 modality.")
        else:
            return ABIDESegmentationFactory._create_single_modality_train_valid_test(source_dir, modalities,
                                                                                     dataset_id, test_size, sites,
                                                                                     max_subjects,
                                                                                     augmentation_strategy)

    @staticmethod
    def _create_single_modality_train_test(source_dir: str, modality: Modality, dataset_id: int, test_size: float,
                                           sites: List[str] = None, max_subjects: int = None,
                                           augmentation_strategy=None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_paths = list()
        target_paths = list()

        if sites is not None:
            dirs = [dir for dir in sorted(os.listdir(source_dir)) if any(substring in dir for substring in sites)]
            csv = csv[csv["filename"].str.contains("|".join(sites))]
        else:
            dirs = [dir for dir in sorted(os.listdir(source_dir))]

        if max_subjects is not None:
            choices = np.random.choice(len(dirs), max_subjects, replace=False)
            dirs = np.array(dirs)[choices]
            csv = csv[csv["filename"].str.contains("|".join(dirs))]

        for dir in sorted(dirs):
            source_paths.append(
                extract_file_paths(os.path.join(source_dir, dir, "mri/patches/image")))
            target_paths.append(
                extract_file_paths(os.path.join(source_dir, dir, "mri/patches/labels")))

        source_paths = np.array(sorted([item for sublist in source_paths for item in sublist]))
        target_paths = np.array(sorted([item for sublist in target_paths for item in sublist]))

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        if max_subjects is not None:
            choices = np.random.choice(len(source_paths), (50,), replace=False)
            source_paths = source_paths[choices]
            target_paths = target_paths[choices]

        train_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths,
                                                                          np.asarray(filtered_csv["center_class"])))

        train_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[train_ids],
                                                       target_paths=np.array(natural_sort(target_paths))[train_ids],
                                                       modalities=modality,
                                                       dataset_id=dataset_id,
                                                       transforms=[ToNumpyArray(), ToNDTensor()],
                                                       augmentation_strategy=augmentation_strategy)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[test_ids],
                                                      target_paths=np.array(natural_sort(target_paths))[test_ids],
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        return train_dataset, test_dataset, filtered_csv

    @staticmethod
    def _create_single_modality_train_valid_test(source_dir: str, modality: Modality,
                                                 dataset_id: int, test_size: float, sites: List[str] = None,
                                                 max_subjects: int = None,
                                                 augmentation_strategy: DataAugmentationStrategy = None):

        csv = pandas.read_csv(os.path.join(source_dir, "output.csv"))

        source_paths = list()
        target_paths = list()

        if sites is not None:
            dirs = [dir for dir in sorted(os.listdir(source_dir)) if any(substring in dir for substring in sites)]
            csv = csv[csv["filename"].str.contains("|".join(sites))]
        else:
            dirs = [dir for dir in sorted(os.listdir(source_dir))]

        if max_subjects is not None:
            choices = np.random.choice(len(dirs), max_subjects, replace=False)
            dirs = np.array(dirs)[choices]
            csv = csv[csv["filename"].str.contains("|".join(dirs))]

        for dir in sorted(dirs):
            source_paths.append(
                extract_file_paths(os.path.join(source_dir, dir, "mri/patches/image")))
            target_paths.append(
                extract_file_paths(os.path.join(source_dir, dir, "mri/patches/labels")))

        source_paths = np.array(sorted([item for sublist in source_paths for item in sublist]))
        target_paths = np.array(sorted([item for sublist in target_paths for item in sublist]))

        filtered_csv = csv.loc[csv["center_class"].isin([1, 2, 3])]

        train_valid_ids, test_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths, filtered_csv["center_class"]))
        train_ids, valid_ids = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(source_paths[train_valid_ids],
                                                                          np.asarray(filtered_csv["center_class"])[
                                                                              train_valid_ids]))

        train_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(source_paths))[train_valid_ids][train_ids],
            target_paths=np.array(natural_sort(target_paths))[train_valid_ids][train_ids],
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=augmentation_strategy)

        valid_dataset = iSEGSegmentationFactory.create(
            source_paths=np.array(natural_sort(source_paths))[train_valid_ids][valid_ids],
            target_paths=np.array(natural_sort(target_paths))[train_valid_ids][valid_ids],
            modalities=modality,
            dataset_id=dataset_id,
            transforms=[ToNumpyArray(), ToNDTensor()],
            augmentation_strategy=None)

        test_dataset = iSEGSegmentationFactory.create(source_paths=np.array(natural_sort(source_paths))[test_ids],
                                                      target_paths=np.array(natural_sort(target_paths))[test_ids],
                                                      modalities=modality,
                                                      dataset_id=dataset_id,
                                                      transforms=[ToNumpyArray(), ToNDTensor()],
                                                      augmentation_strategy=None)

        return train_dataset, valid_dataset, test_dataset, filtered_csv
