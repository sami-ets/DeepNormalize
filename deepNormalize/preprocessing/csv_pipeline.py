import csv
import logging

import numpy as np
import os
from samitorch.inputs.patch import CenterCoordinate
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray
from samitorch.utils.files import extract_file_paths
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)


class ToCSViSEGPipeline(object):
    LOGGER = logging.getLogger("iSEGPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        t1_file_names = list()
        t2_file_names = list()
        target_file_names = list()
        subjects = list()

        for subject in sorted(os.listdir(self._source_dir)):
            source_paths_T1_ = extract_file_paths(os.path.join(self._source_dir, subject, "T1"))
            source_paths_T2_ = extract_file_paths(os.path.join(self._source_dir, subject, "T2"))
            target_paths_ = extract_file_paths(os.path.join(self._source_dir, subject, "Labels"))
            subjects_ = [subject] * len(source_paths_T1_)
            t1_file_names.append(np.array(source_paths_T1_))
            t2_file_names.append(np.array(source_paths_T2_))
            target_file_names.append(np.array(target_paths_))
            subjects.append(np.array(subjects_))

        t1_file_names = sorted([item for sublist in t1_file_names for item in sublist])
        t2_file_names = sorted([item for sublist in t2_file_names for item in sublist])
        target_file_names = sorted([item for sublist in target_file_names for item in sublist])
        subjects = sorted([item for sublist in subjects for item in sublist])

        with open(os.path.join(self._output_dir, 'output.csv'), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "T2", "labels", "center_class", "subjects"])

            for source_path, source_path_t2, target_path, subject in zip(t1_file_names, t2_file_names,
                                                                         target_file_names, subjects):
                self.LOGGER.info("Processing file {}".format(source_path))
                transformed_image = self._transforms(source_path)
                transformed_labels = self._transforms(target_path)

                sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

                center_coordinate = CenterCoordinate(sample.x, sample.y)

                center_class = (int(center_coordinate.class_id))

                csv_data = np.vstack((source_path, source_path_t2, target_path, center_class, subject))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item]])
            output_file.close()


class ToCSVMRBrainSPipeline(object):
    LOGGER = logging.getLogger("MRBrainSPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        center_class = list()
        source_paths_t1_1mm = list()
        source_paths_t2 = list()
        source_paths_t1 = list()
        source_paths_t1_ir = list()
        target_paths = list()
        target_paths_training = list()
        subjects = list()

        for subject in sorted(os.listdir(os.path.join(self._source_dir))):
            source_paths_t2_ = extract_file_paths(os.path.join(self._source_dir, subject, "T2_FLAIR"))
            source_paths_t1_ir_ = extract_file_paths(os.path.join(self._source_dir, subject, "T1_IR"))
            source_paths_t1_1mm_ = extract_file_paths(os.path.join(self._source_dir, subject, "T1_1mm"))
            source_paths_t1_ = extract_file_paths(os.path.join(self._source_dir, subject, "T1"))
            target_paths_ = extract_file_paths(os.path.join(self._source_dir, subject, "LabelsForTesting"))
            target_paths_training_ = extract_file_paths(os.path.join(self._source_dir, subject, "LabelsForTraining"))

            subjects_ = [subject] * len(source_paths_t1_1mm_)
            source_paths_t2.append(np.array(source_paths_t2_))
            source_paths_t1_ir.append(np.array(source_paths_t1_ir_))
            source_paths_t1_1mm.append(np.array(source_paths_t1_1mm_))
            source_paths_t1.append(np.array(source_paths_t1_))
            target_paths.append(np.array(target_paths_))
            target_paths_training.append(np.array(target_paths_training_))
            subjects.append(np.array(subjects_))

        source_paths_t2 = sorted([item for sublist in source_paths_t2 for item in sublist])
        source_paths_t1_ir = sorted([item for sublist in source_paths_t1_ir for item in sublist])
        source_paths_t1_1mm = sorted([item for sublist in source_paths_t1_1mm for item in sublist])
        source_paths_t1 = sorted([item for sublist in source_paths_t1 for item in sublist])
        target_paths = sorted([item for sublist in target_paths for item in sublist])
        target_paths_training = sorted([item for sublist in target_paths_training for item in sublist])
        subjects = sorted([item for sublist in subjects for item in sublist])

        with open(os.path.join(self._output_dir, 'output.csv'), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["T1_1mm", "T1", "T1_IR", "T2_FLAIR", "LabelsForTesting", "LabelsForTraining", "center_class",
                 "subjects"])

            for source_path_t2, source_path_t1_ir, source_path_t1_1mm, source_path_t1, target_path, target_path_training, subject in zip(
                    source_paths_t2, source_paths_t1_ir, source_paths_t1_1mm, source_paths_t1, target_paths,
                    target_paths_training, subjects):
                self.LOGGER.info("Processing file {}".format(source_path_t1))
                transformed_image = self._transforms(source_path_t1)
                transformed_labels = self._transforms(target_path)

                sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

                center_coordinate = CenterCoordinate(sample.x, sample.y)

                center_class = (int(center_coordinate.class_id))

                csv_data = np.vstack((
                    source_path_t1_1mm, source_path_t1, source_path_t1_ir, source_path_t2, target_path,
                    target_path_training, center_class, subject))

                for item in range(csv_data.shape[1]):
                    writer.writerow(
                        [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item],
                         csv_data[5][item], csv_data[6][item], csv_data[7][item]])
            output_file.close()


class ToCSVABIDEipeline(object):
    LOGGER = logging.getLogger("ABIDEPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        source_paths = list()
        target_paths = list()
        subjects = list()
        for dir in sorted(os.listdir(self._source_dir)):
            source_paths_ = extract_file_paths(os.path.join(self._source_dir, dir, "mri/patches/image"))
            target_paths_ = extract_file_paths(os.path.join(self._source_dir, dir, "mri/patches/labels"))
            subjects_ = [dir] * len(source_paths_)
            source_paths.append(np.array(source_paths_))
            target_paths.append(np.array(target_paths_))
            subjects.append(np.array(subjects_))

        source_paths = sorted([item for sublist in source_paths for item in sublist])
        target_paths = sorted([item for sublist in target_paths for item in sublist])
        subjects = sorted([item for sublist in subjects for item in sublist])

        with open(os.path.join(self._output_dir, 'output.csv'), mode='a+') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "labels", "center_class", "subjects"])

            for source_path, target_path, subject in zip(source_paths, target_paths, subjects):
                self.LOGGER.info("Processing file {}".format(source_path))
                transformed_image = self._transforms(source_path)
                transformed_labels = self._transforms(target_path)

                sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

                center_coordinate = CenterCoordinate(sample.x, sample.y)

                center_class = (int(center_coordinate.class_id))

                csv_data = np.vstack((source_path, target_path, center_class, subject))

                for item in range(csv_data.shape[1]):
                    writer.writerow([csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item]])
            output_file.close()


if __name__ == "__main__":
    # ToCSViSEGPipeline("/mnt/md0/Data/Preprocessed_4/iSEG/Training/",
    #                   output_dir="/mnt/md0/Data/Preprocessed_4/iSEG/Training/").run()
    # ToCSVMRBrainSPipeline("/mnt/md0/Data/Preprocessed_4/MRBrainS/DataNii/TrainingData/",
    #                       output_dir="/mnt/md0/Data/Preprocessed_4/MRBrainS/DataNii/TrainingData/").run()
    ToCSViSEGPipeline("/mnt/md0/Data/Preprocessed_8/iSEG/Training/",
                      output_dir="/mnt/md0/Data/Preprocessed_8/iSEG/Training").run()
    ToCSVMRBrainSPipeline("/mnt/md0/Data/Preprocessed_8/MRBrainS/DataNii/TrainingData/",
                          output_dir="/mnt/md0/Data/Preprocessed_8/MRBrainS/DataNii/TrainingData").run()

    # ToCSVABIDEipeline(args.path_abide, output_dir="/home/AM54900").run()
