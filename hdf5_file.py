import logging

import h5py
import numpy as np
import pandas
from samitorch.inputs.transformers import ToNumpyArray
from torchvision import transforms

from deepNormalize.inputs.datasets import iSEGSegmentationFactory
from deepNormalize.utils.utils import natural_sort

logging.basicConfig(level=logging.INFO)


class HDF5Writer(object):

    def __init__(self, csv_path, dataset, test_size, modalities):
        LOGGER = logging.getLogger("HD5FWriter")
        self._csv = pandas.read_csv(csv_path)
        self._dataset = dataset
        self._test_size = test_size
        self._modalities = modalities
        self._transform = transforms.Compose([ToNumpyArray()])

    def get_iseg_train_valid_test_paths(self, modalities):
        subjects = np.array(self._csv["subjects"].drop_duplicates().tolist())

        train_subjects, valid_subjects = iSEGSegmentationFactory.shuffle_split(subjects, self._test_size)
        valid_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(valid_subjects, self._test_size)
        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = self._csv.loc[self._csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = self._csv[self._csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = (
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["labels"]))))
        valid_source_paths, valid_target_paths = (
            np.stack([natural_sort(list(valid_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(valid_csv["labels"]))))
        test_source_paths, test_target_paths = (
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["labels"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["labels"]))))

        return (train_source_paths, train_target_paths), \
               (valid_source_paths, valid_target_paths), \
               (test_source_paths, test_target_paths), \
               (reconstruction_source_paths, reconstruction_target_paths)

    def get_mrbrains_train_valid_test_paths(self, modalities):
        subjects = np.array(self._csv["subjects"].drop_duplicates().tolist())

        train_subjects, valid_subjects = iSEGSegmentationFactory.shuffle_split(subjects, self._test_size)
        valid_subjects, test_subjects = iSEGSegmentationFactory.shuffle_split(valid_subjects, self._test_size)

        reconstruction_subject = test_subjects[
            np.random.choice(np.arange(0, len(test_subjects)), len(test_subjects), replace=False)]

        filtered_csv = self._csv.loc[self._csv["center_class"].isin([1, 2, 3])]

        train_csv = filtered_csv[filtered_csv["subjects"].isin(train_subjects)]
        valid_csv = filtered_csv[filtered_csv["subjects"].isin(valid_subjects)]
        test_csv = filtered_csv[filtered_csv["subjects"].isin(test_subjects)]
        reconstruction_csv = self._csv[self._csv["subjects"].isin(reconstruction_subject)]

        train_source_paths, train_target_paths = (
            np.stack([natural_sort(list(train_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(train_csv["LabelsForTesting"]))))
        valid_source_paths, valid_target_paths = (
            np.stack([natural_sort(list(valid_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(valid_csv["LabelsForTesting"]))))
        test_source_paths, test_target_paths = (
            np.stack([natural_sort(list(test_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(test_csv["LabelsForTesting"]))))
        reconstruction_source_paths, reconstruction_target_paths = (
            np.stack([natural_sort(list(reconstruction_csv[str(modality)])) for modality in modalities], axis=1),
            np.array(natural_sort(list(reconstruction_csv["LabelsForTesting"]))))

        return (train_source_paths, train_target_paths), \
               (valid_source_paths, valid_target_paths), \
               (test_source_paths, test_target_paths), \
               (reconstruction_source_paths, reconstruction_target_paths)

    def create_dataset_(self, group, dataset_name, source_paths):
        dataset = group.create_dataset(dataset_name, compression="gzip", compression_opts=9,
                                       shape=(len(source_paths), 1, 32, 32, 32),
                                       chunks=True,
                                       dtype=np.float32)

        temp = np.zeros((len(source_paths), 1, 32, 32, 32), dtype=np.float32)
        for i, image in enumerate(source_paths):
            logging.info("Processing file {}".format(image))
            img = self._transform(image)
            temp[i] = img

        dataset[...] = temp

        return group

    def create_group(self, f, group_name, paths):
        group = f.create_group(group_name)

        t1 = np.array([image[0] for image in paths[0]])
        t2 = np.array([image[1] for image in paths[0]])
        labels = np.array(paths[1])

        group = self.create_dataset_(group, self._modalities[0], t1)
        group = self.create_dataset_(group, self._modalities[1], t2)
        self.create_dataset_(group, "labels", labels)

    def create_file(self, h5py_path):
        f = h5py.File(h5py_path, mode="w", libver="latest")

        if self._dataset == "iSEG":
            training, validation, testing, reconstruction = self.get_iseg_train_valid_test_paths(self._modalities)
        elif self._dataset == "MRBrainS":
            training, validation, testing, reconstruction = self.get_mrbrains_train_valid_test_paths(self._modalities)
        else:
            raise NotImplementedError

        self.create_group(f, "train", training)
        self.create_group(f, "valid", validation)
        self.create_group(f, "test", testing)
        self.create_group(f, "reconstruction", reconstruction)

        f.close()


if __name__ == "__main__":
    # fm = HDF5Writer("/data/users/pldelisle/datasets/Preprocessed_4/iSEG/Training/output.csv",
    #                    "iSEG", 0.3, ["T1", "T2"])
    # fm.create_file("/data/users/pldelisle/datasets/iseg.hdf5")

    fm = HDF5Writer("/data/users/pldelisle/datasets/Preprocessed_4/MRBrainS/DataNii/TrainingData/output.csv",
                    "MRBrainS", 0.3, ["T1", "T2_FLAIR"])
    fm.create_file("/home/AM54900/mrbrains.hd5f")

    # fm = HDF5Writer("/mnt/md0/Data/Preprocessed_4/iSEG/Training/output.csv",
    #                 "iSEG", 0.3, ["T1", "T2"])
    # fm.create_file("/mnt/home/ETS/iseg.hd5f")
    #
    # fm = HDF5Writer("/mnt/md0/Data/Preprocessed_4/MRBrainS/DataNii/TrainingData/output.csv",
    #                 "MRBrainS", 0.3, ["T1", "T2_FLAIR"])
    # fm.create_file("/mnt/home/ETS/mrbrains.hd5f")

    # fm = HDF5Writer("/mnt/md0/Data/Preprocessed_4/iSEG/Training/output.csv",
    #                 "iSEG", 0.3, ["T1", "T2"])
    # fm.create_file("/mnt/md0/iseg.hdf5")

    # fm = HDF5Writer("/mnt/md0/Data/Preprocessed_4/MRBrainS/DataNii/TrainingData/output.csv",
    #                 "MRBrainS", 0.3, ["T1", "T2_FLAIR"])
    # fm.create_file("/mnt/md0/mrbrains.hdf5")
