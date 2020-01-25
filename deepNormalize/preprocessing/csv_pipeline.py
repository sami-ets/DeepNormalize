import argparse
import csv
import os

import numpy as np
from samitorch.inputs.patch import CenterCoordinate
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray
from samitorch.utils.files import extract_file_paths
from torchvision.transforms import Compose


class ToCSViSEGPipeline(object):

    def __init__(self, root_dir: str, target_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._target_dir = target_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        t1_file_names = list()
        t2_file_names = list()
        target_file_names = list()
        center_class = list()

        for (root, dirs, files), (target_root, target_dirs, target_files) in zip(
                os.walk(os.path.join(self._source_dir, "T1")),
                os.walk(self._target_dir)):
            for file in files:
                transformed_image = self._transforms(os.path.join(root, file))
                transformed_labels = self._transforms(os.path.join(target_root, file))

                sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

                center_coordinate = CenterCoordinate(sample.x, sample.y)

                t1_file_names.append(os.path.join(root, file))
                target_file_names.append(os.path.join(target_root, file))
                center_class.append(int(center_coordinate.class_id))

        for root, dirs, files in os.walk(os.path.join(self._source_dir, "T2")):
            for file in files:
                t2_file_names.append(os.path.join(root, file))

        t1_file_names = np.array(t1_file_names)
        t2_file_names = np.array(t2_file_names)
        target_file_names = np.array(target_file_names)
        center_classes = np.array(center_class)

        csv_data = np.vstack((t1_file_names, t2_file_names, target_file_names, center_classes))

        with open(os.path.join(self._output_dir, 'output.csv'), mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "T2", "labels", "center_class"])
            for item in range(csv_data.shape[1]):
                writer.writerow([csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item]])
        output_file.close()


class ToCSVMRBrainSPipeline(object):

    def __init__(self, root_dir: str, target_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._target_dir = target_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        center_class = list()

        source_dir_t1_1mm = os.path.join(os.path.join(self._source_dir, "*"), "T1_1mm")
        source_dir_t2 = os.path.join(os.path.join(self._source_dir, "*"), "T2_FLAIR")
        source_dir_t1 = os.path.join(os.path.join(self._source_dir, "*"), "T1")
        source_dir_t1_ir = os.path.join(os.path.join(self._source_dir, "*"), "T1_IR")
        target_dir = os.path.join(os.path.join(self._target_dir, "*"), "LabelsForTesting")
        target_dir_full = os.path.join(os.path.join(self._target_dir, "*"), "LabelsForTraining")

        source_paths_t1_1mm = extract_file_paths(source_dir_t1_1mm)
        source_paths_t2 = extract_file_paths(source_dir_t2)
        source_paths_t1 = extract_file_paths(source_dir_t1)
        source_paths_t1_ir = extract_file_paths(source_dir_t1_ir)
        target_paths = extract_file_paths(target_dir)
        target_paths_training = extract_file_paths(target_dir_full)

        for file, target_file in zip(source_paths_t1_1mm, target_paths):
            transformed_image = self._transforms(file)
            transformed_labels = np.ceil(self._transforms(target_file))

            sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

            center_coordinate = CenterCoordinate(sample.x, sample.y)
            center_class.append(int(center_coordinate.class_id))

        t1_1mm_file_names = np.array(source_paths_t1_1mm)
        t2_file_names = np.array(source_paths_t2)
        t1_file_names = np.array(source_paths_t1)
        t1_ir_file_names = np.array(source_paths_t1_ir)
        target_file_names = np.array(target_paths)
        target_paths_training = np.array(target_paths_training)
        center_classes = np.array(center_class)

        csv_data = np.vstack(
            (t1_1mm_file_names, t1_file_names, t1_ir_file_names, t2_file_names, target_file_names,
             target_paths_training, center_classes))

        with open(os.path.join(self._output_dir, 'output.csv'), mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["T1_1mm", "T1", "T1_IR", "T2", "LabelsForTesting", "LabelsForTraining", "center_class"])
            for item in range(csv_data.shape[1]):
                writer.writerow(
                    [csv_data[0][item], csv_data[1][item], csv_data[2][item], csv_data[3][item], csv_data[4][item],
                     csv_data[5][item], csv_data[6][item]])
        output_file.close()


class ToCSVABIDEipeline(object):

    def __init__(self, root_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        file_names = list()
        center_class = list()
        source_paths = list()
        target_paths = list()
        for dir in sorted(os.listdir(self._source_dir)):
            source_paths.append(
                np.array(extract_file_paths(os.path.join(self._source_dir, dir, "mri/patches/image"))))
            target_paths.append(
                np.array(extract_file_paths(os.path.join(self._source_dir, dir, "mri/patches/labels"))))

        source_paths = sorted([item for sublist in source_paths for item in sublist])
        target_paths = sorted([item for sublist in target_paths for item in sublist])

        for source_path, target_path in zip(source_paths, target_paths):
            transformed_image = self._transforms(source_path)
            transformed_labels = self._transforms(target_path)

            sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

            center_coordinate = CenterCoordinate(sample.x, sample.y)

            center_class.append(int(center_coordinate.class_id))

        file_names = np.array(source_paths)
        labels_file_names = np.array(target_paths)
        center_classes = np.array(center_class)

        csv_data = np.vstack((file_names, labels_file_names, center_classes))

        with open(os.path.join(self._output_dir, 'output.csv'), mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["T1", "labels", "center_class"])
            for item in range(csv_data.shape[1]):
                writer.writerow([csv_data[0][item], csv_data[1][item], csv_data[2][item]])
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    #parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)
    parser.add_argument('--path-abide', type=str, help='Path to the preprocessed directory.', required=True)
    args = parser.parse_args()
    #ToCSViSEGPipeline(os.path.join(args.path_iseg), output_dir=args.path_iseg,
    #                 target_dir=os.path.join(args.path_iseg, "label")).run()
    #ToCSVMRBrainSPipeline(args.path_mrbrains, output_dir=args.path_mrbrains,
    #                      target_dir=args.path_mrbrains).run()
    ToCSVABIDEipeline(args.path_abide, output_dir=args.path_abide).run()
