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
        file_names = list()
        center_class = list()

        for (root, dirs, files), (target_root, target_dirs, target_files) in zip(os.walk(self._source_dir),
                                                                                 os.walk(self._target_dir)):

            for file in files:
                transformed_image = self._transforms(os.path.join(root, file))
                transformed_labels = self._transforms(os.path.join(target_root, file))

                sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

                center_coordinate = CenterCoordinate(sample.x, sample.y)

                file_names.append(os.path.join(root, file))
                center_class.append(int(center_coordinate.class_id))

        file_names = np.array(file_names)
        center_classes = np.array(center_class)

        csv_data = np.vstack((file_names, center_classes))

        with open(os.path.join(self._output_dir, 'output.csv'), mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filename", "center_class"])
            for item in range(csv_data.shape[1]):
                writer.writerow([csv_data[0][item], csv_data[1][item]])
        output_file.close()


class ToCSVMRBrainSPipeline(object):

    def __init__(self, root_dir: str, target_dir: str, output_dir: str):
        self._source_dir = root_dir
        self._target_dir = target_dir
        self._output_dir = output_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        file_names = list()
        center_class = list()
        self._source_dir = os.path.join(os.path.join(self._source_dir, "*"), "T1_1mm")
        self._target_dir = os.path.join(os.path.join(self._target_dir, "*"), "LabelsForTesting")

        source_paths, target_paths = extract_file_paths(self._source_dir), extract_file_paths(self._target_dir)

        for file, target_file in zip(source_paths, target_paths):
            transformed_image = self._transforms(file)
            transformed_labels = np.ceil(self._transforms(target_file))

            sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=False)

            center_coordinate = CenterCoordinate(sample.x, sample.y)

            file_names.append(file)
            center_class.append(int(center_coordinate.class_id))

        file_names = np.array(file_names)
        center_classes = np.array(center_class)

        csv_data = np.vstack((file_names, center_classes))

        with open(os.path.join(self._output_dir, 'output.csv'), mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filename", "center_class"])
            for item in range(csv_data.shape[1]):
                writer.writerow([csv_data[0][item], csv_data[1][item]])
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

            file_names.append(source_path)
            center_class.append(int(center_coordinate.class_id))

        file_names = np.array(file_names)
        center_classes = np.array(center_class)

        csv_data = np.vstack((file_names, center_classes))

        with open(os.path.join(self._output_dir, 'output.csv'), mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["filename", "center_class"])
            for item in range(csv_data.shape[1]):
                writer.writerow([csv_data[0][item], csv_data[1][item]])
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)
    parser.add_argument('--path-abide', type=str, help='Path to the preprocessed directory.', required=True)
    args = parser.parse_args()
    # ToCSViSEGPipeline(os.path.join(args.path_iseg, "T1"), output_dir=args.path_iseg,
    #                   target_dir=os.path.join(args.path_iseg, "label")).run()
    # ToCSVMRBrainSPipeline(args.path_mrbrains, output_dir=args.path_mrbrains,
    #                       target_dir=args.path_mrbrains).run()
    ToCSVABIDEipeline(args.path_abide, output_dir=args.path_abide).run()
