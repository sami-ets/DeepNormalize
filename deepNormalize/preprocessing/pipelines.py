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
import nibabel as nib
import os
import argparse
import re
import logging

from functools import reduce

from torchvision.transforms import transforms

from samitorch.inputs.transformers import ToNumpyArray, RemapClassIDs, ToNifti1Image, NiftiToDisk, ApplyMask, \
    ResampleNiftiImageToTemplate, CropToContent, PadToShape, LoadNifti
from samitorch.inputs.images import Modalities, Image


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


class iSEGPreProcessingPipeline(AbstractPreProcessingPipeline):
    """
    A iSEG data pre-processing pipeline. Remap classes to a [0, 3] ordering.
    """
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, input_modality: str, output_dir: str, output_modality: str = None):
        """
        Pre-processing pipeline constructor.

        Args:
            root_dir: Root directory where all files are located.
        """
        self._root_dir = root_dir
        self._transforms = None
        self._input_modality = input_modality
        self._output_modality = output_modality if output_modality is not None else input_modality
        self._output_dir = output_dir

        self.LOGGER.info("Changing class IDs of {} images in {}".format(input_modality, root_dir))

    def run(self, prefix: str = "Preprocessed_"):
        """
        Apply piepline's transformations.

        Args:
            prefix (str): Prefix to transformed file.
        """
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            for file in files:
                self._transforms = transforms.Compose([ToNumpyArray(),
                                                       RemapClassIDs([10, 150, 250], [1, 2, 3]),
                                                       ToNifti1Image(),
                                                       NiftiToDisk(os.path.join(root, prefix + file))])
                self._transforms(os.path.join(root, file))


class MRBrainsImagePreProcessingPipeline(AbstractPreProcessingPipeline):
    """
       A MRBrainS data pre-processing pipeline. Resample images to a Template size.
    """

    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, input_modality: str, output_dir: str, output_modality: str = None):
        """
        Pre-processing pipeline constructor.

        Args:
            root_dir: Root directory where all files are located.
        """
        self._root_dir = root_dir
        self._transforms = None
        self._input_modality = input_modality
        self._output_modality = output_modality if output_modality is not None else input_modality
        self._output_dir = output_dir

        self.LOGGER.info("Resampling of {} images in {}".format(input_modality, root_dir))

    def _run_images_transforms(self, prefix: str = "Preprocessed_"):
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            images = list(filter(re.compile(r"^T.*\.nii").search, files))
            for file in images:
                if not "_1mm" in file:
                    self._transforms = transforms.Compose([LoadNifti(),
                                                           ResampleNiftiImageToTemplate(clip=False,
                                                                                        template=root + "/T1_1mm.nii",
                                                                                        interpolation="continuous"),
                                                           ApplyMask(root + "/Preprocessed_LabelsForTesting.nii"),
                                                           NiftiToDisk(os.path.join(root, prefix + file))])
                else:
                    self._transforms = transforms.Compose([LoadNifti(),
                                                           ApplyMask(root + "/Preprocessed_LabelsForTesting.nii"),
                                                           NiftiToDisk(os.path.join(root, prefix + file))])

                self._transforms(os.path.join(root, file))

    def _run_label_transforms(self, prefix: str = "Preprocessed_"):
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            labels = list(filter(re.compile(r"^Labels.*\.nii").search, files))
            for file in labels:
                self._transforms = transforms.Compose([LoadNifti(),
                                                       ResampleNiftiImageToTemplate(clip=True,
                                                                                    template=root + "/T1_1mm.nii",
                                                                                    interpolation="linear"),
                                                       NiftiToDisk(os.path.join(root, prefix + file))])
                self._transforms(os.path.join(root, file))

    def run(self, prefix: str = "Preprocessed_"):
        """
        Apply piepline's transformations.

        Args:
            prefix (str): Prefix to transformed file.
        """
        self._run_label_transforms(prefix=prefix)
        self._run_images_transforms(prefix=prefix)


class AnatomicalPreProcessingPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, input_modality: str, output_dir: str, output_modality: str = None):
        self._root_dir = root_dir
        self._normalized_shape = self.compute_normalized_shape_from_images_in(root_dir)
        self._transforms = None
        self._input_modality = input_modality
        self._output_modality = output_modality if output_modality is not None else input_modality
        self._output_dir = output_dir

        self.LOGGER.info("Computing the normalized shape of {} images in {}".format(input_modality, root_dir))
        self._normalized_shape = self.compute_normalized_shape_from_images_in(root_dir)
        self._transforms = transforms.Compose([ToNumpyArray(),
                                               CropToContent(),
                                               PadToShape(self._normalized_shape)])

    def run(self, prefix="Normalized_"):
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            for file in list(filter(lambda path: Image.is_(self._input_modality, path), files)):
                try:
                    match = re.search(
                        self._root_dir + '/(?P<file_name>.*)',
                        os.path.join(root, file))

                    self.LOGGER.info("Processing: {}".format(file))

                    transformed_image = self._transforms(os.path.join(root, file))
                    header = self._get_image_header(os.path.join(root, file))
                    transforms_ = transforms.Compose([ToNifti1Image(header),
                                                      NiftiToDisk(os.path.join(root, prefix + file))])
                    transforms_(transformed_image)

                except Exception as e:
                    self.LOGGER.warning(e)

    def compute_normalized_shape_from_images_in(self, root_dir):
        image_shapes = []

        for root, dirs, files in os.walk(root_dir):
            for file in list(filter(lambda path: Image.is_(self._input_modality, path), files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} wiith error {}".format(file, e))

        return reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), image_shapes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)

    args = parser.parse_args()
    iSEGPreProcessingPipeline(root_dir=args.path_iseg + "/label").run()
    MRBrainsImagePreProcessingPipeline(root_dir=args.path_mrbrains).run()
    T1AnatomicalPreProcessingPipeline(root_dir=args.path_mrbrains).run([r"^Preprocessed_.*\.nii"])
    T1AnatomicalPreProcessingPipeline(root_dir=args.path_iseg).run([r".*T.*\.nii", r"^Preprocessed_.*label.*\.nii"])

    print("Preprocessing pipeline completed successfully.")
