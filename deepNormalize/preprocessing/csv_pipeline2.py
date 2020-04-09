import csv
import logging
import re

import numpy as np
import os
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

    def run(self):

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

        with open(os.path.join(self._output_dir, 'output_iseg_images.csv'), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "T2", "labels", "subjects"])

            for source_path, source_path_t2, target_path, subject in zip(source_paths_t1, source_paths_t2,
                                                                         target_paths, subjects):
                self.LOGGER.info("Processing file {}".format(source_path))

                csv_data = np.vstack((source_path, source_path_t2, target_path, subject))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item]])
            output_file.close()


class ToCSVMRBrainSPipeline(object):
    LOGGER = logging.getLogger("MRBrainSPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
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

        with open(os.path.join(self._output_dir, 'output_mrbrains_images.csv'), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["T1_1mm", "T1", "T1_IR", "T2_FLAIR", "LabelsForTesting", "LabelsForTraining", "subjects"])

            for source_path_t2, source_path_t1_ir, source_path_t1_1mm, source_path_t1, target_path, target_path_training, subject in zip(
                    source_paths_t2, source_paths_t1_ir, source_paths_t1_1mm, source_paths_t1, target_paths,
                    target_paths_training, subjects):
                self.LOGGER.info("Processing file {}".format(source_path_t1))

                csv_data = np.vstack((
                    source_path_t1_1mm, source_path_t1, source_path_t1_ir, source_path_t2, target_path,
                    target_path_training, subject))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item],
                         csv_data[5][item], csv_data[6][item]])
            output_file.close()


class ToCSVABIDEPipeline(object):
    LOGGER = logging.getLogger("ABIDEPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        source_paths = list()
        target_paths = list()
        subjects = list()
        sites = list()
        for dir in sorted(os.listdir(self._source_dir)):
            source_paths_ = extract_file_paths(os.path.join(self._source_dir, dir, "mri/"), "real_brainmask.nii.gz")
            target_paths_ = extract_file_paths(os.path.join(self._source_dir, dir, "mri/"), "aligned_labels.nii.gz")
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

        with open(os.path.join(self._output_dir, 'output_abide_images.csv'), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "labels", "subjects", "site"])

            for source_path, target_path, subject, site in zip(source_paths, target_paths, subjects, sites):
                self.LOGGER.info("Processing file {}".format(source_path))

                csv_data = np.vstack((source_path, target_path, subject, site))

                for item in range(csv_data.shape[1]):
                    writer.writerow([csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item]])
            output_file.close()


if __name__ == "__main__":
    ToCSViSEGPipeline("/mnt/md0/Data/Preprocessed/iSEG/Training/",
                      output_dir="/mnt/md0/Data/Preprocessed/iSEG/Training").run()
    ToCSVMRBrainSPipeline("/mnt/md0/Data/Preprocessed/MRBrainS/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/DataNii/TrainingData").run()
    ToCSVABIDEPipeline("/home/pierre-luc-delisle/ABIDE/5.1/", output_dir="/mnt/md0/Data/").run()
