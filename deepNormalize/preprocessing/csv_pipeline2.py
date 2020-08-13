import csv
import logging

import numpy as np
import os
import re
from samitorch.inputs.transformers import ToNumpyArray
from samitorch.utils.files import extract_file_paths
from torchvision.transforms import Compose

from deepNormalize.utils.utils import natural_sort

logging.basicConfig(level=logging.INFO)
LABELSFORTESTING = 0
LABELSFORTRAINNG = 1
ROIT1 = 2
T1 = 3
T1_1MM = 4
T1_IR = 5
T2_FLAIR = 6


class ToCSViSEGPipeline(object):
    LOGGER = logging.getLogger("iSEGPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self, output_filename: str):

        source_paths_t2 = list()
        source_paths_t1 = list()
        target_paths = list()

        for subject in sorted(os.listdir(os.path.join(self._source_dir))):
            source_paths_t1.append(extract_file_paths(os.path.join(self._source_dir, subject, "T1")))
            source_paths_t2.append(extract_file_paths(os.path.join(self._source_dir, subject, "T2")))
            target_paths.append(extract_file_paths(os.path.join(self._source_dir, subject, "Labels")))

        subjects = np.arange(1, 11)
        source_paths_t1 = natural_sort([item for sublist in source_paths_t1 for item in sublist])
        source_paths_t2 = natural_sort([item for sublist in source_paths_t2 for item in sublist])
        target_paths = natural_sort([item for sublist in target_paths for item in sublist])

        with open(os.path.join(self._output_dir, output_filename), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["T1", "T2", "labels", "subject", "T1_min", "T1_max", "T1_mean", "T1_std", "T2_min", "T2_max",
                 "T2_mean", "T2_std"])

            for source_path, source_path_t2, target_path, subject in zip(source_paths_t1, source_paths_t2,
                                                                         target_paths, subjects):
                self.LOGGER.info("Processing file {}".format(source_path))

                t1 = ToNumpyArray()(source_path)
                t2 = ToNumpyArray()(source_path_t2)

                csv_data = np.vstack((source_path, source_path_t2, target_path, subject, str(t1.min()), str(t1.max()),
                                      str(t1.mean()), str(t1.std()), str(t2.min()), str(t2.max()), str(t2.mean()),
                                      str(t2.std())))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item],
                         csv_data[5][item], csv_data[6][item], csv_data[7][item], csv_data[8][item], csv_data[9][item],
                         csv_data[10][item], csv_data[11][item]])
            output_file.close()


class ToCSVMRBrainSPipeline(object):
    LOGGER = logging.getLogger("MRBrainSPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self, output_filename: str):
        source_paths_t1_1mm = list()
        source_paths_t2 = list()
        source_paths_t1 = list()
        source_paths_t1_ir = list()
        target_paths = list()
        target_paths_training = list()

        for subject in sorted(os.listdir(os.path.join(self._source_dir))):
            source_paths_t2.append(extract_file_paths(os.path.join(self._source_dir, subject, "T2_FLAIR")))
            source_paths_t1_ir.append(extract_file_paths(os.path.join(self._source_dir, subject, "T1_IR")))
            source_paths_t1_1mm.append(extract_file_paths(os.path.join(self._source_dir, subject, "T1_1mm")))
            source_paths_t1.append(extract_file_paths(os.path.join(self._source_dir, subject, "T1")))
            target_paths.append(extract_file_paths(os.path.join(self._source_dir, subject, "LabelsForTesting")))
            target_paths_training.append(
                extract_file_paths(os.path.join(self._source_dir, subject, "LabelsForTraining")))

        subjects = np.arange(1, 6)

        with open(os.path.join(self._output_dir, output_filename), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["T1_1mm", "T1", "T1_IR", "T2_FLAIR", "LabelsForTesting", "LabelsForTraining", "subject", "T1_min",
                 "T1_max", "T1_mean", "T1_std", "T2_min", "T2_max", "T2_mean", "T2_std"])

            for source_path_t2, source_path_t1_ir, source_path_t1_1mm, source_path_t1, target_path, target_path_training, subject in zip(
                    source_paths_t2, source_paths_t1_ir, source_paths_t1_1mm, source_paths_t1, target_paths,
                    target_paths_training, subjects):
                self.LOGGER.info("Processing file {}".format(source_path_t1))

                t1 = ToNumpyArray()(source_path_t1[0])
                t2 = ToNumpyArray()(source_path_t2[0])
                csv_data = np.vstack((
                    source_path_t1_1mm, source_path_t1, source_path_t1_ir, source_path_t2, target_path,
                    target_path_training, subject, str(t1.min()), str(t1.max()), str(t1.mean()), str(t1.std()),
                    str(t2.min()), str(t2.max()), str(t2.mean()), str(t2.std())))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item],
                         csv_data[5][item], csv_data[6][item], csv_data[7][item], csv_data[8][item], csv_data[9][item],
                         csv_data[10][item], csv_data[11][item], csv_data[12][item], csv_data[13][item],
                         csv_data[14][item]])
            output_file.close()


class ToCSVABIDEPipeline(object):
    LOGGER = logging.getLogger("ABIDEPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self, output_filename: str):
        source_paths = list()
        target_paths = list()
        subjects = list()
        sites = list()
        for dir in sorted(os.listdir(self._source_dir)):
            source_paths_ = extract_file_paths(os.path.join(self._source_dir, dir, "mri", "T1"), "T1.nii.gz")
            target_paths_ = extract_file_paths(os.path.join(self._source_dir, dir, "mri", "Labels"), "Labels.nii.gz")
            subject_ = dir
            source_paths.append(source_paths_)
            target_paths.append(target_paths_)
            if len(source_paths_) is not 0:
                match = re.search('(?P<site>.*)_(?P<patient_id>[0-9]*)', str(dir))
                site_ = match.group("site")
                sites.append(site_)
                subjects.append(subject_)

        source_paths = list(filter(None, source_paths))
        target_paths = list(filter(None, target_paths))

        with open(os.path.join(self._output_dir, output_filename), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "labels", "subject", "site", "min", "max", "mean", "std"])

            for source_path, target_path, subject, site in zip(source_paths, target_paths, subjects, sites):
                self.LOGGER.info("Processing file {}".format(source_path))

                image = ToNumpyArray()(source_path[0])
                csv_data = np.vstack((source_path, target_path, subject, site, (image.min()), (image.max()),
                                      (image.mean()), (image.std())))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item],
                         csv_data[5][item], csv_data[6][item], csv_data[7][item]])
            output_file.close()


if __name__ == "__main__":
    ToCSVABIDEPipeline("/mnt/md0/Data/ABIDE/",
                       output_dir="/mnt/md0/Data/ABIDE/").run('output_abide_images.csv')
    ToCSVABIDEPipeline("/mnt/md0/Data/ABIDE_scaled/",
                       output_dir="/mnt/md0/Data/ABIDE_scaled/").run('output_abide_images.csv')
    ToCSVABIDEPipeline("/mnt/md0/Data/ABIDE_scaled_augmented/",
                       output_dir="/mnt/md0/Data/ABIDE_scaled_augmented/").run('output_abide_augmented_images.csv')
    ToCSVABIDEPipeline("/mnt/md0/Data/ABIDE_augmented/",
                       output_dir="/mnt/md0/Data/ABIDE_augmented/").run('output_abide_augmented_images.csv')
    ToCSVABIDEPipeline("/mnt/md0/Data/ABIDE_standardized_triple/",
                       output_dir="/mnt/md0/Data/ABIDE_standardized_triple/").run('output_abide_standardized_images.csv')
    ToCSViSEGPipeline("/mnt/md0/Data/iSEG/Training/",
                      output_dir="/mnt/md0/Data/iSEG/Training").run("output_iseg_images.csv")
    ToCSViSEGPipeline("/mnt/md0/Data/iSEG_scaled/Training/",
                      output_dir="/mnt/md0/Data/iSEG_scaled/Training").run("output_iseg_images.csv")
    ToCSViSEGPipeline("/mnt/md0/Data/iSEG_scaled_augmented/Training/",
                      output_dir="/mnt/md0/Data/iSEG_scaled_augmented/Training/").run(
        "output_iseg_augmented_images.csv")
    ToCSViSEGPipeline("/mnt/md0/Data/iSEG_augmented/Training/",
                      output_dir="/mnt/md0/Data/iSEG_augmented/Training/").run(
        "output_iseg_augmented_images.csv")
    ToCSViSEGPipeline("/mnt/md0/Data/iSEG_standardized_dual/Training/",
                      output_dir="/mnt/md0/Data/iSEG_standardized_dual/Training/").run(
        "output_iseg_standardized_dual_images.csv")
    ToCSViSEGPipeline("/mnt/md0/Data/iSEG_standardized_triple/Training/",
                      output_dir="/mnt/md0/Data/iSEG_standardized_triple/Training/").run(
        "output_iseg_standardized_triple_images.csv")
    ToCSVMRBrainSPipeline("/mnt/md0/Data/MRBrainS/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/MRBrainS/DataNii/TrainingData").run(
        "output_mrbrains_images.csv")
    ToCSVMRBrainSPipeline("/mnt/md0/Data//MRBrainS_scaled/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData").run(
        "output_mrbrains_images.csv")
    ToCSVMRBrainSPipeline("/mnt/md0/Data/MRBrainS_scaled_augmented/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/MRBrainS_scaled_augmented/DataNii/TrainingData").run(
        "output_mrbrains_augmented_images.csv")
    ToCSVMRBrainSPipeline("/mnt/md0/Data/MRBrainS_augmented/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/MRBrainS_augmented/DataNii/TrainingData").run(
        "output_mrbrains_augmented_images.csv")
    ToCSVMRBrainSPipeline("/mnt/md0/Data/MRBrainS_standardized_dual/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/MRBrainS_standardized_dual/DataNii/TrainingData").run(
        "output_mrbrains_standardized_dual_images.csv")
    ToCSVMRBrainSPipeline("/mnt/md0/Data/MRBrainS_standardized_triple/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/MRBrainS_standardized_triple/DataNii/TrainingData").run(
        "output_mrbrains_standardized_triple_images.csv")
