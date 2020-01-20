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
from typing import Tuple, Union

import abc
import nibabel as nib
import numpy as np
import os
import re
import shutil
from functools import reduce
from nipype.interfaces import freesurfer
from samitorch.inputs.images import Image
from samitorch.inputs.patch import Patch, CenterCoordinate
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, RemapClassIDs, ToNifti1Image, NiftiToDisk, ApplyMask, \
    ResampleNiftiImageToTemplate, CropToContent, PadToShape, LoadNifti, PadToPatchShape, Squeeze
from samitorch.utils.slice_builder import SliceBuilder
from torchvision.transforms import transforms

print(os.getenv("FREESURFER_HOME"))


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

    def __init__(self, root_dir: str, output_dir: str):
        """
        Pre-processing pipeline constructor.

        Args:
            root_dir: Root directory where all files are located.
        """
        self._root_dir = root_dir
        self._transforms = None
        self._output_dir = output_dir

        self.LOGGER.info("Changing class IDs of images in {}".format(root_dir))

    def run(self, prefix: str = ""):
        """
        Apply piepline's transformations.

        Args:
            prefix (str): Prefix to transformed file.
        """
        if not os.path.exists(os.path.join(self._output_dir, "label")):
            os.makedirs(os.path.join(self._output_dir, "label"))

        for root, dirs, files in os.walk(os.path.join(self._root_dir, "label")):
            for file in files:
                self._transforms = transforms.Compose([ToNumpyArray(),
                                                       RemapClassIDs([10, 150, 250], [1, 2, 3]),
                                                       ToNifti1Image(),
                                                       NiftiToDisk(os.path.join(os.path.join(self._output_dir, "label"),
                                                                                prefix + file))])
                self._transforms(os.path.join(root, file))

        for root, dirs, files in os.walk(self._root_dir):
            root_dir_end = os.path.basename(os.path.normpath(root))

            if "label" in root_dir_end:
                pass
            for file in files:
                if not os.path.exists(os.path.join(self._output_dir, root_dir_end)):
                    os.makedirs(os.path.join(self._output_dir, root_dir_end))
                shutil.copy(os.path.join(root, file), os.path.join(self._output_dir, root_dir_end))


class MRBrainsPreProcessingPipeline(AbstractPreProcessingPipeline):
    """
       A MRBrainS data pre-processing pipeline. Resample images to a Template size.
    """

    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        """
        Pre-processing pipeline constructor.

        Args:
            root_dir: Root directory where all files are located.
        """
        self._root_dir = root_dir
        self._transforms = None
        self._output_dir = output_dir

    def _run_images_transforms(self, prefix: str = ""):
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            root_dir_number = os.path.basename(os.path.normpath(root))
            images = list(filter(re.compile(r"^T.*\.nii").search, files))
            for file in images:
                if not "_1mm" in file:
                    self._transforms = transforms.Compose([LoadNifti(),
                                                           ResampleNiftiImageToTemplate(clip=False,
                                                                                        template=os.path.join(root,
                                                                                                              "T1_1mm.nii"),
                                                                                        interpolation="continuous"),
                                                           ApplyMask(os.path.join(os.path.join(self._output_dir,
                                                                                               root_dir_number),
                                                                                  "LabelsForTesting.nii")),
                                                           NiftiToDisk(os.path.join(
                                                               os.path.join(self._output_dir, root_dir_number),
                                                               prefix + file))])
                else:
                    self._transforms = transforms.Compose([LoadNifti(),
                                                           ApplyMask(os.path.join(os.path.join(self._output_dir,
                                                                                               root_dir_number),
                                                                                  "LabelsForTesting.nii")),
                                                           NiftiToDisk(os.path.join(
                                                               os.path.join(self._output_dir, root_dir_number),
                                                               prefix + file))])

                self._transforms(os.path.join(root, file))

    def _run_label_transforms(self, prefix: str = ""):
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            root_dir_number = os.path.basename(os.path.normpath(root))
            labels = list(filter(re.compile(r"^Labels.*\.nii").search, files))

            for file in labels:
                if not os.path.exists(os.path.join(self._output_dir, root_dir_number)):
                    os.makedirs(os.path.join(self._output_dir, root_dir_number))
                self._transforms = transforms.Compose([LoadNifti(),
                                                       ResampleNiftiImageToTemplate(clip=True,
                                                                                    template=root + "/T1_1mm.nii",
                                                                                    interpolation="linear"),
                                                       NiftiToDisk(
                                                           os.path.join(os.path.join(self._output_dir, root_dir_number),
                                                                        prefix + file))])
                self._transforms(os.path.join(root, file))

    def run(self, prefix: str = ""):
        """
        Apply piepline's transformations.

        Args:
            prefix (str): Prefix to transformed file.
        """
        self._run_label_transforms(prefix=prefix)
        self._run_images_transforms(prefix=prefix)


class AnatomicalPreProcessingPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, output_dir: str):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._normalized_shape = self.compute_normalized_shape_from_images_in(root_dir)
        self._transforms = transforms.Compose([ToNumpyArray(),
                                               CropToContent(),
                                               PadToShape(self._normalized_shape)])

    def run(self, prefix=""):
        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            root_dir_number = os.path.basename(os.path.normpath(root))

            for file in files:
                if not os.path.exists(os.path.join(self._output_dir, root_dir_number)):
                    os.makedirs(os.path.join(self._output_dir, root_dir_number))
                try:
                    self.LOGGER.info("Processing: {}".format(file))

                    transformed_image = self._transforms(os.path.join(root, file))
                    header = self._get_image_header(os.path.join(root, file))
                    transforms_ = transforms.Compose([ToNifti1Image(header),
                                                      NiftiToDisk(
                                                          os.path.join(
                                                              os.path.join(self._output_dir, root_dir_number),
                                                              prefix + file))])
                    transforms_(transformed_image)

                except Exception as e:
                    self.LOGGER.warning(e)

    def compute_normalized_shape_from_images_in(self, root_dir):
        image_shapes = []

        for root, dirs, files in os.walk(root_dir):
            for file in list(filter(lambda path: Image.is_nifti(path), files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        return reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), image_shapes)


class MRBrainSPatchPreProcessingPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PatchExtraction")

    def __init__(self, root_dir: str, output_dir: str, patch_size: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int]):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._patch_size = patch_size
        self._step = step
        self._transforms = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

    def run(self, prefix="", keep_foreground_only=True, keep_labels=True):

        for root, dirs, files in os.walk(self._source_dir):
            root_dir_number = os.path.basename(os.path.normpath(root))

            if not os.path.exists(os.path.join(self._output_dir)):
                os.makedirs(os.path.join(self._output_dir))

            for file in files:
                modality = file.split(".")[0]
                modality_path = os.path.join(os.path.join(self._output_dir, root_dir_number), modality)

                if not os.path.exists(modality_path):
                    os.makedirs(modality_path)

                transformed_image = self._transforms(os.path.join(root, file))

                if keep_labels:
                    transformed_labels = (np.ceil(self._transforms(os.path.join(root, "LabelsForTesting.nii")))).astype(
                        np.int8)
                    sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=True)
                else:
                    sample = Sample(x=transformed_image, dataset_id=None, is_labeled=False)

                patches = MRBrainSPatchPreProcessingPipeline.get_patches(sample, self._patch_size, self._step,
                                                                         keep_foreground_only=keep_foreground_only,
                                                                         keep_labels=keep_labels)

                for i, patch in enumerate(patches):
                    x = transformed_image[tuple(patch.slice)]
                    transform_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                        os.path.join(self._output_dir, root_dir_number, modality, str(i) + ".nii.gz"))])
                    transform_(x)

                    if keep_labels:
                        y = transformed_labels[tuple(patch.slice)]
                        transform_ = transforms.Compose(
                            [ToNifti1Image(), NiftiToDisk(os.path.join(root, "LabelsForTesting", str(i) + ".nii.gz"))])
                        transform_(y)

    @staticmethod
    def get_patches(sample, patch_size, step, keep_foreground_only: bool = True, keep_labels: bool = True):
        slices = SliceBuilder(sample.x.shape, patch_size=patch_size, step=step).build_slices()

        patches = list()

        for slice in slices:
            if keep_labels:
                center_coordinate = CenterCoordinate(sample.x[tuple(slice)], sample.y[tuple(slice)])
                patches.append(Patch(slice, 0, center_coordinate))
            else:
                patches.append(Patch(slice, 0, None))

        if keep_foreground_only:
            return np.array(list(filter(lambda patch: patch.center_coordinate.is_foreground, patches)))
        else:
            return patches


class iSEGPatchPreProcessingPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("PatchExtraction")

    def __init__(self, root_dir: str, output_dir: str, patch_size: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int]):
        self._source_dir = root_dir
        self._output_dir = output_dir
        self._patch_size = patch_size
        self._step = step
        self._transforms = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

    def run(self, prefix="", keep_foreground_only=True, keep_labels=True):

        for root, dirs, files in os.walk(self._source_dir):
            root_dir_number = os.path.basename(os.path.normpath(root))

            if not os.path.exists(os.path.join(self._output_dir)):
                os.makedirs(os.path.join(self._output_dir))

            if root_dir_number == "label":
                pass
            for file in files:
                modality_path = os.path.join(self._output_dir, root_dir_number)
                subject = re.search(r"-(?P<subject_no>.*)-", file).group("subject_no")
                if keep_labels:
                    label_path = os.path.join(os.path.join(self._output_dir, "label"), subject)

                    if not os.path.exists(label_path):
                        os.makedirs(label_path)

                if not os.path.exists(os.path.join(modality_path, subject)):
                    os.makedirs(os.path.join(modality_path, subject))

                current_path = os.path.join(modality_path, subject)
                transformed_image = self._transforms(os.path.join(root, file))

                if keep_labels:
                    transformed_labels = self._transforms(os.path.join(os.path.join(self._source_dir, "label"),
                                                                       "subject-" + subject + "-label.nii")).astype(
                        np.int8)
                    sample = Sample(x=transformed_image, y=transformed_labels, dataset_id=None, is_labeled=True)
                else:
                    sample = Sample(x=transformed_image, dataset_id=None, is_labeled=False)
                patches = iSEGPatchPreProcessingPipeline.get_patches(sample, self._patch_size, self._step,
                                                                     keep_foreground_only=keep_foreground_only,
                                                                     keep_labels=keep_labels)

                for i, patch in enumerate(patches):
                    x = transformed_image[tuple(patch.slice)]
                    transform_ = transforms.Compose(
                        [ToNifti1Image(), NiftiToDisk(os.path.join(current_path, str(i) + ".nii.gz"))])
                    transform_(x)

                    if keep_labels:
                        y = transformed_labels[tuple(patch.slice)]
                        transform_ = transforms.Compose(
                            [ToNifti1Image(), NiftiToDisk(os.path.join(label_path, str(i) + ".nii.gz"))])
                        transform_(y)

    @staticmethod
    def get_patches(sample, patch_size, step, keep_foreground_only: bool = True, keep_labels: bool = True):
        slices = SliceBuilder(sample.x.shape, patch_size=patch_size, step=step).build_slices()

        patches = list()

        for slice in slices:
            if keep_labels:
                center_coordinate = CenterCoordinate(sample.x[tuple(slice)], sample.y[tuple(slice)])
                patches.append(Patch(slice, 0, center_coordinate))
            else:
                patches.append(Patch(slice, 0, None))

        if keep_foreground_only:
            return np.array(list(filter(lambda patch: patch.center_coordinate.is_foreground, patches)))
        else:
            return patches


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


class ABIDEPreprocessingPipeline(AbstractPreProcessingPipeline):
    """
    An ABIDE data pre-processing pipeline. Extract necessary tissues for brain segmentation among other transformations.
    """
    LOGGER = logging.getLogger("PreProcessingPipeline")

    def __init__(self, root_dir: str, patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)):
        """
        Pre-processing pipeline constructor.

        Args:
            root_dir: Root directory where all files are located.
        """
        self._root_dir = root_dir
        self._transforms = None
        self._patch_size = patch_size
        self._step = step
        self._normalized_shape = self._compute_normalized_shape_from_images_in(root_dir)
        self._transform_crop = transforms.Compose([ToNumpyArray(),
                                                   CropToContent(),
                                                   PadToShape(self._normalized_shape)])
        self._transform_align = transforms.Compose([ToNumpyArray(),
                                                    Transpose((0, 2, 3, 1)),
                                                    Rotate(k=1, axes=(2, 3))
                                                    ])
        self._transform_patch = transforms.Compose([ToNumpyArray(), PadToPatchShape(patch_size=patch_size, step=step)])

    def run(self, prefix: str = ""):
        for root, dirs, files in os.walk(self._root_dir):
            for dir in dirs:
                if os.path.exists(os.path.join(self._root_dir, dir, "mri", "aparc+aseg.mgz")):
                    self._extract_labels(os.path.join(self._root_dir, dir, "mri"))
                    self._crop_to_content(os.path.join(self._root_dir, dir, "mri"))
                    self._align(os.path.join(self._root_dir, dir, "mri"))
                    self._apply_mask(os.path.join(self._root_dir, dir, "mri"))
                    self._extract_patches(os.path.join(self._root_dir, dir, "mri"),
                                          os.path.join(self._root_dir, dir, "mri/patches"))

    def _extract_patches(self, root, output_dir, keep_foreground_only=True, keep_labels=True):
        if not os.path.exists(os.path.join(output_dir, "image")):
            os.makedirs(os.path.join(output_dir, "image"))
        if not os.path.exists(os.path.join(output_dir, "labels")):
            os.makedirs(os.path.join(output_dir, "labels"))

        transformed_image = Sample(x=os.path.join(root, "real_brainmask.nii.gz"),
                                   y=os.path.join(root, "aligned_labels.nii.gz"),
                                   is_labeled=True)
        self._transform_patch(transformed_image)

        patches = ABIDEPreprocessingPipeline.get_patches(transformed_image, self._patch_size, self._step,
                                                         keep_foreground_only)

        for i, patch in enumerate(patches):
            x = transformed_image.x[tuple(patch.slice)]
            transform_ = transforms.Compose(
                [ToNifti1Image(), NiftiToDisk(os.path.join(os.path.join(output_dir, "image"), str(i) + ".nii.gz"))])
            transform_(x)

            if keep_labels:
                y = transformed_image.y[tuple(patch.slice)]
                transform_ = transforms.Compose(
                    [ToNifti1Image(),
                     NiftiToDisk(os.path.join(os.path.join(output_dir, "labels"), str(i) + ".nii.gz"))])
                transform_(y)

    @staticmethod
    def get_patches(sample, patch_size, step, keep_foreground_only: bool = True, keep_labels=True):
        slices = SliceBuilder(sample.x.shape, patch_size=patch_size, step=step).build_slices()

        patches = list()

        for slice in slices:
            if keep_labels:
                center_coordinate = CenterCoordinate(sample.x[tuple(slice)], sample.y[tuple(slice)])
                patches.append(Patch(slice, 0, center_coordinate))
            else:
                patches.append(Patch(slice, 0, None))

        if keep_foreground_only:
            return np.array(list(filter(lambda patch: patch.center_coordinate.is_foreground, patches)))
        else:
            return patches

    def _apply_mask(self, root):
        try:
            self._transform_mask = transforms.Compose(
                [LoadNifti(),
                 ApplyMask(os.path.join(root, "aligned_labels.nii.gz")),
                 NiftiToDisk(os.path.join(root, "real_brainmask.nii.gz"))])
            self._transform_mask(os.path.join(root, "aligned_brainmask.nii.gz"))

        except Exception as e:
            self.LOGGER.warning(e)

    def _align(self, root):
        try:
            sample = Sample(x=os.path.join(root, "cropped_brainmask.nii.gz"),
                            y=os.path.join(root, "cropped_labels.nii.gz"),
                            is_labeled=True)
            transformed_sample = self._transform_align(sample)

            transforms_ = transforms.Compose([Squeeze(0),
                                              ToNifti1Image(),
                                              NiftiToDisk([
                                                  os.path.join(root, "aligned_brainmask.nii.gz"),
                                                  os.path.join(root, "aligned_labels.nii.gz")])])
            transforms_(transformed_sample)

        except Exception as e:
            self.LOGGER.warning(e)

    def _extract_labels(self, root):
        self._mri_binarize(os.path.join(root, "aparc+aseg.mgz"),
                           os.path.join(root, "wm_mask.mgz"),
                           "wm")

        self._mri_binarize(os.path.join(root, "aparc+aseg.mgz"),
                           os.path.join(root, "gm_mask.mgz"),
                           "gm")
        self._mri_binarize(os.path.join(root, "aparc+aseg.mgz"),
                           os.path.join(root, "csf_mask.mgz"),
                           "csf")

        self._remap_labels(os.path.join(root, "gm_mask.mgz"),
                           os.path.join(root, "remapped_gm_mask.nii.gz"),
                           1, 2)
        self._remap_labels(os.path.join(root, "wm_mask.mgz"),
                           os.path.join(root, "remapped_wm_mask.nii.gz"),
                           1, 3)

        transform_ = transforms.Compose([ToNumpyArray()])
        wm_vol = transform_(os.path.join(root, "remapped_wm_mask.nii.gz"))
        gm_vol = transform_(os.path.join(root, "remapped_gm_mask.nii.gz"))
        csf_vol = transform_(os.path.join(root, "csf_mask.mgz"))

        merged = self._merge_volumes(wm_vol, gm_vol, csf_vol)

        transform_ = transforms.Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(root, "labels.nii.gz"))])
        transform_(merged)

    def _crop_to_content(self, root):
        try:
            sample = Sample(x=os.path.join(root, "brainmask.mgz"),
                            y=os.path.join(root, "labels.nii.gz"),
                            is_labeled=True)
            transformed_image = self._transform_crop(sample)
            sample = Sample.from_sample(transformed_image)
            transforms_ = transforms.Compose([Squeeze(0),
                                              ToNifti1Image(),
                                              NiftiToDisk([
                                                  os.path.join(root, "cropped_brainmask.nii.gz"),
                                                  os.path.join(root, "cropped_labels.nii.gz")])])
            transforms_(sample)

        except Exception as e:
            self.LOGGER.warning(e)

    def _compute_normalized_shape_from_images_in(self, root_dir):
        image_shapes = []

        for root, dirs, files in os.walk(root_dir):
            for file in list(filter(lambda path: Image.is_mgz(path) and "brainmask.mgz" in path, files)):
                try:
                    self.LOGGER.debug("Computing the bounding box of {}".format(file))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        return reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), image_shapes)

    @staticmethod
    def _mri_binarize(input, output, tissue):
        if tissue == "wm":
            freesurfer.Binarize(in_file=input, wm=True, binary_file=output).run()
        elif tissue == "gm":
            freesurfer.Binarize(in_file=input, args="--gm", binary_file=output).run()
        elif tissue == "csf":
            freesurfer.Binarize(in_file=input, match=[4, 5, 24, 31, 43, 44, 63, 122, 221], binary_file=output).run()
        else:
            raise NotImplementedError("Invalid tissue to binarize.")

    @staticmethod
    def _remap_labels(input, output, old_value, remapped_value):
        remap_transform = transforms.Compose([ToNumpyArray(),
                                              RemapClassIDs([old_value], [remapped_value]),
                                              ToNifti1Image(),
                                              NiftiToDisk(output)])
        return remap_transform(input)

    @staticmethod
    def _merge_volumes(volume_1, volume_2, volume_3):
        return volume_1 + volume_2 + volume_3


class Transpose(object):
    def __init__(self, axis):
        self._axis = axis

    def __call__(self, input: Union[Sample, np.ndarray]) -> Union[Sample, np.ndarray]:
        if isinstance(input, np.ndarray):
            return np.transpose(input, axes=self._axis)
        elif isinstance(input, Sample):
            sample = input

            if not isinstance(sample.x, np.ndarray):
                raise TypeError("Only Numpy arrays are supported.")
            transformed_sample = Sample.from_sample(sample)
            transformed_sample.x = np.transpose(sample.x, axes=self._axis)
            transformed_sample.y = np.transpose(sample.y, axes=self._axis)

            return sample.update(transformed_sample)


class Rotate(object):
    def __init__(self, k, axes):
        self._k = k
        self._axes = axes

    def __call__(self, input: np.ndarray):
        if isinstance(input, np.ndarray):
            return np.expand_dims(np.rot90(input.squeeze(0), k=self._k), axis=0)
        elif isinstance(input, Sample):
            sample = input

            if not isinstance(sample.x, np.ndarray):
                raise TypeError("Only Numpy arrays are supported.")
            transformed_sample = Sample.from_sample(sample)
            transformed_sample.x = np.rot90(sample.x, self._k, self._axes)
            transformed_sample.y = np.rot90(sample.y, self._k, self._axes)

            return sample.update(transformed_sample)


class FlipLR(object):
    def __init__(self):
        pass

    def __call__(self, input: Union[Sample, np.ndarray]) -> Union[Sample, np.ndarray]:
        if isinstance(input, np.ndarray):
            return np.fliplr(input)
        elif isinstance(input, Sample):
            sample = input

            if not isinstance(sample.x, np.ndarray):
                raise TypeError("Only Numpy arrays are supported.")
            transformed_sample = Sample.from_sample(sample)
            transformed_sample.x = np.fliplr(sample.x)
            transformed_sample.y = np.fliplr(sample.y)

            return sample.update(transformed_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)
    parser.add_argument('--path-abide', type=str, help='Path to the preprocessed directory.', required=True)
    args = parser.parse_args()

    # iSEGPreProcessingPipeline(root_dir=os.path.join(args.path_iseg, "Training"),
    #                           output_dir="/mnt/md0/Data/Preprocessed/iSEG/Preprocessed").run()
    # MRBrainsPreProcessingPipeline(root_dir=os.path.join(args.path_mrbrains, "TrainingData"),
    #                               output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Preprocessed").run()
    #
    # AnatomicalPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Preprocessed",
    #                                 output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/SizeNormalized").run()
    # AnatomicalPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Preprocessed",
    #                                 output_dir="/mnt/md0/Data/Preprocessed/iSEG/SizeNormalized").run()
    #
    # AlignPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/SizeNormalized",
    #               transforms=transforms.Compose([ToNumpyArray(),
    #                                              FlipLR()]),
    #               output_dir="/mnt/md0/Data/Preprocessed/iSEG/Aligned"
    #               ).run()
    # AlignPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/SizeNormalized",
    #               transforms=transforms.Compose([ToNumpyArray(),
    #                                              Transpose((0, 2, 3, 1))]),
    #               output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Aligned"
    #               ).run()
    #
    # MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Aligned",
    #                                    output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/Patches/Aligned",
    #                                    patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)).run()
    # iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Aligned",
    #                                output_dir="/mnt/md0/Data/Preprocessed/iSEG/Patches/Aligned",
    #                                patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)).run()

    AnatomicalPreProcessingPipeline(root_dir=os.path.join(args.path_mrbrains, "TestData"),
                                    output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/SizeNormalized").run()
    AnatomicalPreProcessingPipeline(root_dir=os.path.join(args.path_iseg, "Testing"),
                                    output_dir="/mnt/md0/Data/Preprocessed/iSEG/Testing/SizeNormalized").run()

    AlignPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Testing/SizeNormalized",
                  transforms=transforms.Compose([ToNumpyArray(),
                                                 FlipLR()]),
                  output_dir="/mnt/md0/Data/Preprocessed/iSEG/Testing/Aligned"
                  ).run()
    AlignPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/SizeNormalized",
                  transforms=transforms.Compose([ToNumpyArray(),
                                                 Transpose((0, 2, 3, 1))]),
                  output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Aligned"
                  ).run()

    MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Aligned",
                                       output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Patches/Aligned",
                                       patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)).run(keep_foreground_only=False,
                                                                                          keep_labels=False)
    iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/Testing/Aligned",
                                   output_dir="/mnt/md0/Data/Preprocessed/iSEG/Testing/Patches/Aligned",
                                   patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)).run(keep_foreground_only=False,
                                                                                      keep_labels=False)
    # ABIDEPreprocessingPipeline(root_dir=args.path_abide).run()

    print("Preprocessing pipeline completed successfully.")
