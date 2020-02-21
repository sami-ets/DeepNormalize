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

import argparse
import logging
from typing import Callable

import abc
import nibabel as nib
import numpy as np
import os
import re
import shutil
from functools import reduce
from samitorch.inputs.transformers import ToNumpyArray, NiftiToDisk, CropToContent, PadToShape, Image, ToNifti1Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torchvision.transforms import transforms

from deepNormalize.preprocessing.pipelines import MRBrainSPatchPreProcessingPipeline, iSEGPatchPreProcessingPipeline


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


class QuantileScalerTransformer(object):
    def __init__(self):
        self._scaler = RobustScaler()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        if isinstance(input, np.ndarray):
            return self._scaler.fit_transform(input)
        else:
            raise NotImplementedError("Type {} is not supported.".format(type(input)))


class StandardScalerTransformer(object):
    def __init__(self):
        self._scaler = StandardScaler()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        if isinstance(input, np.ndarray):
            return self._scaler.fit_transform(input)
        else:
            raise NotImplementedError("Type {} is not supported.".format(type(input)))


class MinMaxScalerTransformer(object):
    def __init__(self, min=0.0, max=1.0):
        self._min = min
        self._max = max
        self._scaler = MinMaxScaler(copy=False, feature_range=(self._min, self._max))

    def __call__(self, input: np.ndarray) -> np.ndarray:
        if isinstance(input, np.ndarray):
            return self._scaler.fit_transform(input)
        else:
            raise NotImplementedError("Type {} is not supported.".format(type(input)))


class iSEGScalerPipeline(AbstractPreProcessingPipeline):
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
                    images_np.append(self._transforms(os.path.join(root, file)))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        images = np.array(images_np).astype(np.float32)
        images_shape = images.shape
        images = images.reshape(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4])
        transform_ = transforms.Compose([self._scaler])

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


class MRBrainSScalerPipeline(AbstractPreProcessingPipeline):
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
        transform_ = transforms.Compose([self._scaler])

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


class Transpose(object):
    def __init__(self, axis):
        self._axis = axis

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.transpose(input, axes=self._axis)


class Rotate(object):
    def __init__(self, k):
        self._k = k

    def __call__(self, input: np.ndarray):
        return np.expand_dims(np.rot90(input.squeeze(0), k=self._k), axis=0)


class FlipLR(object):
    def __init__(self):
        pass

    def __call__(self, input: np.ndarray):
        return np.fliplr(input)


class AlignPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, output_dir: str, transforms, params: dict = None):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._transforms = transforms
        self._params = params

    def run(self, prefix=""):
        images_np = list()
        headers = list()
        file_names = list()
        root_dirs = list()

        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            root_dir_number = os.path.basename(os.path.normpath(root))
            for file in files:
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

        for i, header in enumerate(headers):
            transforms_ = transforms.Compose([ToNifti1Image(),
                                              NiftiToDisk(
                                                  os.path.join(
                                                      os.path.join(
                                                          self._output_dir,
                                                          root_dirs[i]),
                                                      prefix + file_names[i]))])

            transforms_(images_np[i])


class DualStandardScaler(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir_iseg: str, root_dir_mrbrains: str, output_dir: str, params: dict = None):
        self._root_dir_iseg = root_dir_iseg
        self._root_dir_mrbrains = root_dir_mrbrains
        self._output_dir = output_dir
        self._normalized_shape = self.compute_normalized_shape_from_images_in(self._root_dir_iseg,
                                                                              self._root_dir_mrbrains)
        self._transforms = transforms.Compose([ToNumpyArray(),
                                               CropToContent(),
                                               PadToShape(self._normalized_shape)])
        self._params = params

    def run(self, prefix="standardize_"):
        images_np = list()
        headers = list()
        file_names = list()
        root_dirs = list()
        root_dirs_number = list()
        EXCLUDED = ["ROI", "label", "Normalized"]

        for root, dirs, files in os.walk(os.path.join(self._root_dir_mrbrains)):
            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r"^T.*\.nii").search, files))
            for file in images:
                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root)
                    root_dirs_number.append(root_dir_number)
                    images_np.append(self._transforms(os.path.join(root, file)))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        for root, dirs, files in os.walk(os.path.join(self._root_dir_iseg)):
            if os.path.basename(os.path.normpath(root)) in EXCLUDED:
                continue

            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r".*T.*\.nii").search, files))

            for file in images:
                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root)
                    root_dirs_number.append(root_dir_number)
                    images_np.append(self._transforms(os.path.join(root, file)))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        images = np.array(images_np).astype(np.float32)
        transformed_images = np.subtract(images, np.mean(images)) / np.std(images)

        for i in range(transformed_images.shape[0]):
            if "MRBrainS" in root_dirs[i]:
                root_dir_number = os.path.basename(os.path.normpath(root_dirs[i]))
                if not os.path.exists(
                        os.path.join(self._output_dir, "MRBrainS/Dual_Standardized/{}".format(root_dir_number))):
                    os.makedirs(os.path.join(self._output_dir, "MRBrainS/Dual_Standardized/{}".format(root_dir_number)))
                transforms_ = transforms.Compose([
                    ToNifti1Image(),
                    NiftiToDisk(
                        os.path.join(
                            os.path.join(self._output_dir,
                                         os.path.join("MRBrainS/Dual_Standardized",
                                                      root_dir_number)),
                            prefix + file_names[i]))])
                transforms_(transformed_images[i])
            elif "iSEG" in root_dirs[i]:
                root_dir_number = os.path.basename(os.path.normpath(root_dirs[i]))
                if not os.path.exists(
                        os.path.join(self._output_dir, "iSEG/Dual_Standardized/{}".format(root_dir_number))):
                    os.makedirs(os.path.join(self._output_dir, "iSEG/Dual_Standardized/{}".format(root_dir_number)))
                transforms_ = transforms.Compose([
                    ToNifti1Image(),
                    NiftiToDisk(
                        os.path.join(
                            os.path.join(self._output_dir,
                                         os.path.join("iSEG/Dual_Standardized",
                                                      root_dir_number)),
                            prefix + file_names[i]))])

                transforms_(transformed_images[i])

        for root, dirs, files in os.walk(self._root_dir_mrbrains):
            root_dir_end = os.path.basename(os.path.normpath(root))

            images = list(filter(re.compile(r"^LabelsFor.*\.nii").search, files))

            for file in images:
                if not os.path.exists(os.path.join(self._output_dir, os.path.join("MRBrainS/Dual_Standardized",
                                                                                  root_dir_end))):
                    os.makedirs(os.path.join(self._output_dir, os.path.join("MRBrainS/Dual_Standardized",
                                                                            root_dir_end)))

                transforms_ = transforms.Compose([ToNumpyArray(),
                                                  CropToContent(),
                                                  PadToShape(self._normalized_shape),
                                                  ToNifti1Image(),
                                                  NiftiToDisk(
                                                      os.path.join(
                                                          os.path.join(self._output_dir,
                                                                       os.path.join("MRBrainS/Dual_Standardized",
                                                                                    root_dir_end)),
                                                          file))])

                transforms_(os.path.join(root, file))

        for root, dirs, files in os.walk(self._root_dir_iseg):
            root_dir_end = os.path.basename(os.path.normpath(root))
            if "ROI" in root_dir_end or "label" in root_dir_end:
                for file in files:
                    if not os.path.exists(os.path.join(self._output_dir, os.path.join("iSEG/Dual_Standardized",
                                                                                      root_dir_end))):
                        os.makedirs(os.path.join(self._output_dir, os.path.join("iSEG/Dual_Standardized",
                                                                                root_dir_end)))
                    transforms_ = transforms.Compose([ToNumpyArray(),
                                                      CropToContent(),
                                                      PadToShape(self._normalized_shape),
                                                      ToNifti1Image(),
                                                      NiftiToDisk(
                                                          os.path.join(
                                                              os.path.join(self._output_dir,
                                                                           os.path.join("iSEG/Dual_Standardized",
                                                                                        root_dir_end)),
                                                              file))])

                    transforms_(os.path.join(root, file))

    def compute_normalized_shape_from_images_in(self, root_dir_1, root_dir_2):
        image_shapes_iSEG = []
        image_shapes_MRBrainS = []

        for root, dirs, files in os.walk(root_dir_1):
            for file in list(filter(lambda path: Image.is_nifti(path), files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes_iSEG.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        c, h, w, d = reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                            image_shapes_iSEG)

        for root, dirs, files in os.walk(root_dir_2):
            for file in list(filter(lambda path: Image.is_nifti(path), files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes_MRBrainS.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        c_2, h_2, w_2, d_2 = reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                                    image_shapes_MRBrainS)
        return max(c, c_2), max(h, h_2), max(w, w_2), max(d, d_2)


class TripleStandardScaler(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir_iseg: str, root_dir_mrbrains: str, root_dir_abide: str, output_dir: str,
                 params: dict = None):
        self._root_dir_iseg = root_dir_iseg
        self._root_dir_mrbrains = root_dir_mrbrains
        self._root_dir_abide = root_dir_abide
        self._output_dir = output_dir
        self._normalized_shape = self.compute_normalized_shape_from_images_in(self._root_dir_iseg,
                                                                              self._root_dir_mrbrains,
                                                                              self._root_dir_abide)
        self._transforms = transforms.Compose([ToNumpyArray(),
                                               CropToContent(),
                                               PadToShape(self._normalized_shape)])
        self._params = params

    def run(self, prefix="triple_standardized"):
        images = list()
        images_means = list()
        images_stds = list()
        headers = list()
        file_names = list()
        root_dirs = list()
        root_dirs_number = list()
        EXCLUDED = ["ROI", "label", "Normalized"]

        for root, dirs, files in os.walk(os.path.join(self._root_dir_mrbrains)):
            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r"^T.*\.nii").search, files))
            for file in images:
                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root)
                    root_dirs_number.append(root_dir_number)
                    image = self._transforms(os.path.join(root, file))
                    images.append(image)
                    images_means.append(np.mean(image))
                    images_stds.append(np.std(image))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        for root, dirs, files in os.walk(os.path.join(self._root_dir_iseg)):
            if os.path.basename(os.path.normpath(root)) in EXCLUDED:
                continue

            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r".*T.*\.nii").search, files))

            for file in images:
                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root)
                    root_dirs_number.append(root_dir_number)
                    image = self._transforms(os.path.join(root, file))
                    images.append(image)
                    images_means.append(np.mean(image))
                    images_stds.append(np.std(image))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        for root, dirs, files in os.walk(self._root_dir_abide):
            for file in list(filter(lambda path: Image.is_mgz(path) and "brainmask.mgz" in path, files)):
                try:
                    self.LOGGER.info("Processing: {}".format(file))
                    file_names.append(file)
                    root_dirs.append(root)
                    image = self._transforms(os.path.join(root, file))
                    images.append(image)
                    images_means.append(np.mean(image))
                    images_stds.append(np.std(image))
                    headers.append(self._get_image_header(os.path.join(root, file)))

                except Exception as e:
                    self.LOGGER.warning(e)

        images_np = np.array(images).astype(np.float32)
        transformed_images = np.subtract(images_np, np.mean(images_np)) / np.std(images_np)

        for i in range(transformed_images.shape[0]):
            if "MRBrainS" in root_dirs[i]:
                root_dir_number = os.path.basename(os.path.normpath(root_dirs[i]))
                if not os.path.exists(
                        os.path.join(self._output_dir, "MRBrainS/Triple_Standardized/{}".format(root_dir_number))):
                    os.makedirs(os.path.join(self._output_dir, "MRBrainS/Triple_Standardized/{}".format(root_dir_number)))
                transforms_ = transforms.Compose([
                    ToNifti1Image(),
                    NiftiToDisk(
                        os.path.join(
                            os.path.join(self._output_dir,
                                         os.path.join("MRBrainS/Triple_Standardized",
                                                      root_dir_number)),
                            prefix + file_names[i]))])
                transforms_(transformed_images[i])
            elif "iSEG" in root_dirs[i]:
                root_dir_number = os.path.basename(os.path.normpath(root_dirs[i]))
                if not os.path.exists(
                        os.path.join(self._output_dir, "iSEG/Triple_Standardized/{}".format(root_dir_number))):
                    os.makedirs(os.path.join(self._output_dir, "iSEG/Triple_Standardized/{}".format(root_dir_number)))
                transforms_ = transforms.Compose([
                    ToNifti1Image(),
                    NiftiToDisk(
                        os.path.join(
                            os.path.join(self._output_dir,
                                         os.path.join("iSEG/Triple_Standardized",
                                                      root_dir_number)),
                            prefix + file_names[i]))])

                transforms_(transformed_images[i])
            elif "ABIDE" in root_dirs[i]:
                root_dir_number = os.path.basename(os.path.normpath(root_dirs[i]))
                transforms_ = transforms.Compose([
                    ToNifti1Image(),
                    NiftiToDisk(os.path.join("/data/users/pldelisle/datasets/ABIDE/freesurfer/{}/mri".format(root_dir_number),
                                             prefix + file_names[i]))])

                transforms_(transformed_images[i])

        for root, dirs, files in os.walk(self._root_dir_mrbrains):
            root_dir_end = os.path.basename(os.path.normpath(root))

            images = list(filter(re.compile(r"^LabelsFor.*\.nii").search, files))

            for file in images:
                if not os.path.exists(os.path.join(self._output_dir, os.path.join("MRBrainS/Triple_Standardized",
                                                                                  root_dir_end))):
                    os.makedirs(os.path.join(self._output_dir, os.path.join("MRBrainS/Triple_Standardized",
                                                                            root_dir_end)))

                transforms_ = transforms.Compose([ToNumpyArray(),
                                                  CropToContent(),
                                                  PadToShape(self._normalized_shape),
                                                  ToNifti1Image(),
                                                  NiftiToDisk(
                                                      os.path.join(
                                                          os.path.join(self._output_dir,
                                                                       os.path.join("MRBrainS/Triple_Standardized",
                                                                                    root_dir_end)),
                                                          file))])

                transforms_(os.path.join(root, file))

        for root, dirs, files in os.walk(self._root_dir_iseg):
            root_dir_end = os.path.basename(os.path.normpath(root))
            if "ROI" in root_dir_end or "label" in root_dir_end:
                for file in files:
                    if not os.path.exists(os.path.join(self._output_dir, os.path.join("iSEG/Triple_Standardized",
                                                                                      root_dir_end))):
                        os.makedirs(os.path.join(self._output_dir, os.path.join("iSEG/Triple_Standardized",
                                                                                root_dir_end)))
                    transforms_ = transforms.Compose([ToNumpyArray(),
                                                      CropToContent(),
                                                      PadToShape(self._normalized_shape),
                                                      ToNifti1Image(),
                                                      NiftiToDisk(
                                                          os.path.join(
                                                              os.path.join(self._output_dir,
                                                                           os.path.join("iSEG/Dual_Standardized",
                                                                                        root_dir_end)),
                                                              file))])

                    transforms_(os.path.join(root, file))

    def compute_normalized_shape_from_images_in(self, root_dir_1, root_dir_2, root_dir_3):
        image_shapes_iSEG = []
        image_shapes_MRBrainS = []
        image_shapes_ABIDE = []

        for root, dirs, files in os.walk(root_dir_1):
            for file in list(filter(lambda path: Image.is_nifti(path), files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes_iSEG.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        c, h, w, d = reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                            image_shapes_iSEG)

        for root, dirs, files in os.walk(root_dir_2):
            for file in list(filter(lambda path: Image.is_nifti(path), files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes_MRBrainS.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        c_2, h_2, w_2, d_2 = reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                                    image_shapes_MRBrainS)

        for root, dirs, files in os.walk(root_dir_3):
            for file in list(filter(lambda path: Image.is_mgz(path) and "brainmask.mgz" in path, files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes_ABIDE.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        c_3, h_3, w_3, d_3 = reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                                    image_shapes_ABIDE)

        return max(c, c_2, c_3), max(h, h_2, h_3), max(w, w_2, w_3), max(d, d_2, d_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)

    args = parser.parse_args()

    iSEGScalerPipeline(root_dir=args.path_iseg,
                       output_dir="/mnt/md0/Data/Preprocessed/iSEG/Scaled",
                       scaler=MinMaxScalerTransformer()).run()
    MRBrainSScalerPipeline(root_dir=args.path_mrbrains,
                           output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Scaled",
                           scaler=MinMaxScalerTransformer()).run()
    MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Scaled",
                                       output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Patches/Scaled",
                                       patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()
    iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Scaled",
                                   output_dir="/mnt/md0/Data/Preprocessed/iSEG/Patches/Scaled",
                                   patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()

    iSEGScalerPipeline(root_dir=args.path_iseg,
                       output_dir="/mnt/md0/Data/Preprocessed/iSEG/Standardized",
                       scaler=StandardScalerTransformer()).run()
    MRBrainSScalerPipeline(root_dir=args.path_mrbrains,
                           output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Standardized",
                           scaler=StandardScalerTransformer()).run()
    iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Standardized",
                                   output_dir="/mnt/md0/Data/Preprocessed/iSEG/Patches/Standardized",
                                   patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()
    MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Standardized",
                                       output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Patches/Standardized",
                                       patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()

    # iSEGScalerPipeline(root_dir=args.path_iseg,
    #                    output_dir="/mnt/md0/Data/Preprocessed/iSEG/Quantile",
    #                    scaler=QuantileScalerTransformer(),
    #                    params={}).run()
    # MRBrainSScalerPipeline(root_dir=args.path_mrbrains,
    #                        output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Quantile",
    #                        scaler=QuantileScalerTransformer(), params={}).run()
    # iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Quantile",
    #                                output_dir="/mnt/md0/Data/Preprocessed/iSEG/Patches/Quantile",
    #                                patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()
    # MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Quantile",
    #                                    output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Patches/Quantile",
    #                                    patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()

    DualStandardScaler(root_dir_iseg="/mnt/md0/Data/Preprocessed/iSEG/Aligned",
                       root_dir_mrbrains="/mnt/md0/Data/Preprocessed/MRBrainS/Aligned",
                       output_dir="/mnt/md0/Data/Preprocessed/").run()
    MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Dual_Standardized",
                                       output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Patches/Dual_Standardized",
                                       patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()
    iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Dual_Standardized",
                                   output_dir="/mnt/md0/Data/Preprocessed/iSEG/Patches/Dual_Standardized",
                                   patch_size=[1, 32, 32, 32], step=[1, 8, 8, 8]).run()

    print("Preprocessing pipeline completed successfully.")
