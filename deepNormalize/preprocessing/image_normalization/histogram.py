# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
import argparse
import logging
import os
import re
import shutil

import nibabel as nib
import numpy as np

from typing import Callable
from samitorch.inputs.transformers import ToNumpyArray, ToNifti1Image, NiftiToDisk
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class AbstractPreProcessingPipeline(metaclass=abc.ABCMeta):
    """
    Define a preprocessing pipeline.
    """

    @staticmethod
    def _get_image_affine(file):
        return nib.load(file).affine

    @staticmethod
    def _get_image_header(file):
        return nib.load(file).header

    @abc.abstractmethod
    def run(self, **kwargs):
        """
        Run the preprocessing pipeline.
        Args:
            **kwargs: Optional keyword arguments.
        """
        raise NotImplementedError


class Histogram(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, output_dir: str, scaler: Callable = None, params: dict = None):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._transforms = transforms.Compose([ToNumpyArray()])
        self._scaler = scaler
        self._params = params

    def run(self, prefix="scaled_"):
        density_histograms = list()
        histograms = list()
        bins = list()
        headers = list()
        file_names = list()
        root_dirs = list()
        EXCLUDED = ["ROI", "label", "Normalized"]

        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            if os.path.basename(os.path.normpath(root)) in EXCLUDED:
                continue

            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r".*T.*\.nii").search, files))
            for file in images:
                if not os.path.exists(os.path.join(self._output_dir, root_dir_number)):
                    os.makedirs(os.path.join(self._output_dir, root_dir_number))

                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root_dir_number)
                    image = self._transforms(os.path.join(root, file)).flatten()
                    density_histogram, bin = np.histogram(image[np.nonzero(image)], bins=1024, density=True)
                    histogram, bin = np.histogram(image[np.nonzero(image)], bins=1024)
                    density_histograms.append(density_histogram)
                    histograms.append(histogram)
                    bins.append(bin)
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        density_histograms, histograms, bins = np.array(density_histograms), np.array(histograms), np.array(bins)

        for histogram, density_histogram, bin in zip(histograms, density_histograms, bins):
            median_x = np.searchsorted(density_histogram.cumsum(), 0.50)
            mean_x = bin[np.searchsorted(histogram, np.mean(histogram))]

        images_shape = images.shape
        images = images.reshape(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4])
        histogram, bins = np.histogram(images.flatten(), 256)
        cdf = histogram.cumsum()

        transform_ = transforms.Compose([self._scaler(self._params)])

        transformed_images = transform_(images).reshape(images_shape)

        for i, (image, header) in enumerate(zip(range(images.shape[0]), headers)):
            transforms_ = transforms.Compose([ToNifti1Image(header),
                                              NiftiToDisk(
                                                  os.path.join(
                                                      os.path.join(self._output_dir, root_dirs[i]),
                                                      prefix + file_names[i]))])

            transforms_(transformed_images[i])

        for root, dirs, files in os.walk(self._root_dir):
            root_dir_end = os.path.basename(os.path.normpath(root))
            if "ROI" in root_dir_end or "label" in root_dir_end:
                for file in files:
                    if not os.path.exists(os.path.join(self._output_dir, root_dir_end)):
                        os.makedirs(os.path.join(self._output_dir, root_dir_end))
                    shutil.copy(os.path.join(root, file), os.path.join(self._output_dir, root_dir_end))


class ScalerMRBrainSPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, output_dir: str, scaler: Callable = None, params: dict = None):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._transforms = transforms.Compose([ToNumpyArray()])
        self._scaler = scaler
        self._params = params

    def run(self, prefix="scaled_"):
        images_np = list()
        headers = list()
        file_names = list()
        root_dirs = list()

        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r"^T.*\.nii").search, files))
            for file in images:
                if not os.path.exists(os.path.join(self._output_dir, root_dir_number)):
                    os.makedirs(os.path.join(self._output_dir, root_dir_number))

                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root_dir_number)
                    images_np.append(self._transforms(os.path.join(root, file)))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        images = np.array(images_np).astype(np.float32)
        images_shape = images.shape
        images = images.reshape(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4])
        transform_ = transforms.Compose([self._scaler(self._params)])

        transformed_images = transform_(images).reshape(images_shape)

        for i, (image, header) in enumerate(zip(range(images.shape[0]), headers)):
            transforms_ = transforms.Compose([ToNifti1Image(header),
                                              NiftiToDisk(
                                                  os.path.join(
                                                      os.path.join(self._output_dir, root_dirs[i]),
                                                      prefix + file_names[i]))])

            transforms_(transformed_images[i])

        for root, dirs, files in os.walk(self._root_dir):
            root_dir_end = os.path.basename(os.path.normpath(root))

            images = list(filter(re.compile(r"^LabelsFor.*\.nii").search, files))

            for file in images:
                if not os.path.exists(os.path.join(self._output_dir, root_dir_end)):
                    os.makedirs(os.path.join(self._output_dir, root_dir_end))
                shutil.copy(os.path.join(root, file), os.path.join(self._output_dir, root_dir_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)

    args = parser.parse_args()
    #
    # ScalerPipeline(root_dir=args.path_iseg,
    #                output_dir="/mnt/md0/Data/Preprocessed/iSEG/Scaled",
    #                scaler=MinMaxScalerTransformer).run()
    # ScalerMRBrainSPipeline(root_dir=args.path_mrbrains,
    #                        output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Scaled",
    #                        scaler=MinMaxScalerTransformer
    #                        ).run()
    #
    # ScalerPipeline(root_dir=args.path_iseg,
    #                output_dir="/mnt/md0/Data/Preprocessed/iSEG/Standardized",
    #                scaler=StandardScalerTransformer).run(prefix="standardized_")
    # ScalerMRBrainSPipeline(root_dir=args.path_mrbrains,
    #                        output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Standardized",
    #                        scaler=StandardScalerTransformer
    #                        ).run(prefix="standardized_")
    #
    Histogram(root_dir=args.path_iseg,
                   output_dir="/mnt/md0/Data/Preprocessed/iSEG/Histogram",
                   params={}).run(prefix="historgram_")
    # ScalerMRBrainSPipeline(root_dir=args.path_mrbrains,
    #                        output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Quantile",
    #                        scaler=QuantileScalerTransformer, params={}).run(prefix="quantile_")

    print("Preprocessing pipeline completed successfully.")
