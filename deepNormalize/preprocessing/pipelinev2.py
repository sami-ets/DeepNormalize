import argparse
import logging
import multiprocessing
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
from samitorch.utils.files import extract_file_paths

from deepNormalize.utils.utils import natural_sort

logging.basicConfig(level=logging.INFO)
print(os.getenv("FREESURFER_HOME"))
LABELSFORTESTING = 0
LABELSFORTRAINNG = 1
ROIT1 = 2
T1 = 3
T1_1MM = 4
T1_IR = 5
T2_FLAIR = 6


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

    @staticmethod
    def chunks(l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _do_job(self, *args):
        raise NotImplementedError

    def _dispatch_jobs(self, files, job_number):
        total = len(files)
        chunk_size = int(total / job_number)
        slices = iSEGPipeline.chunks(files, chunk_size)
        jobs = []

        for slice in slices:
            j = multiprocessing.Process(target=self._do_job, args=(slice,))
            jobs.append(j)
        for j in jobs:
            j.start()

    def _compute_normalized_shape_from_images_in(self, root_dir):
        image_shapes = []

        for root, dirs, files in os.walk(root_dir):
            for file in list(filter(lambda path: Image.is_nifti(path), files)):
                try:
                    self.LOGGER.info("Computing the bounding box of {}".format(os.path.join(root, file)))
                    c, d_min, d_max, h_min, h_max, w_min, w_max = CropToContent.extract_content_bounding_box_from(
                        ToNumpyArray()(os.path.join(root, file)))
                    image_shapes.append((c, d_max - d_min, h_max - h_min, w_max - w_min))
                except Exception as e:
                    self.LOGGER.warning(
                        "Error while computing the content bounding box for {} with error {}".format(file, e))

        return reduce(lambda a, b: (a[0], max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])), image_shapes)

    @staticmethod
    def get_patches_from_sample(sample, patch_size, step, keep_foreground_only: bool = True, keep_labels: bool = True):
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

    @staticmethod
    def get_patches(image, patch_size, step):
        slices = SliceBuilder(image.shape, patch_size=patch_size, step=step).build_slices()

        patches = list()

        for slice in slices:
            patches.append(Patch(slice, 0, None))

        return patches


class iSEGPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("iSEGPipeline")
    PATCH_SIZE = (1, 32, 32, 32)
    STEP = (1, 8, 8, 8)

    def __init__(self, root_dir, output_dir):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._normalized_shape = self._compute_normalized_shape_from_images_in(root_dir)

    def run(self):
        images_T1 = natural_sort(extract_file_paths(os.path.join(self._root_dir, "T1")))
        images_T2 = natural_sort(extract_file_paths(os.path.join(self._root_dir, "T2")))
        labels = natural_sort(extract_file_paths(os.path.join(self._root_dir, "label")))
        files = np.stack((np.array(images_T1), np.array(images_T2), np.array(labels)), axis=1)
        self._dispatch_jobs(files, 5)

    def _do_job(self, files):
        for file in files:
            subject = re.search(r"-(?P<subject_no>.*)-", file[0]).group("subject_no")
            self.LOGGER.info("Processing file {}".format(file[2]))
            label = self._crop_to_content(file[2])
            label = self._align(label)
            label = self._remap_class_ids(label)
            self._extract_patches(label, subject, "Labels", self.PATCH_SIZE, self.STEP)
            self.LOGGER.info("Processing file {}".format(file[0]))
            image = self._crop_to_content(file[0])
            image = self._align(image)
            self._extract_patches(image, subject, "T1", self.PATCH_SIZE, self.STEP)
            self.LOGGER.info("Processing file {}".format(file[1]))
            image = self._crop_to_content(file[1])
            image = self._align(image)
            self._extract_patches(image, subject, "T2", self.PATCH_SIZE, self.STEP)

    def _crop_to_content(self, file):
        if isinstance(file, np.ndarray):
            transforms_ = transforms.Compose([CropToContent(),
                                              PadToShape(self._normalized_shape)])
        else:
            transforms_ = transforms.Compose([ToNumpyArray(),
                                              CropToContent(),
                                              PadToShape(self._normalized_shape)])
        return transforms_(file)

    def _align(self, file):
        transform_ = transforms.Compose([FlipLR()])
        return transform_(file)

    def _remap_class_ids(self, file):
        transform_ = transforms.Compose([RemapClassIDs([10, 150, 250], [1, 2, 3])])
        return transform_(file)

    def _extract_patches(self, image, subject, modality, patch_size, step):
        transforms_ = transforms.Compose([PadToPatchShape(patch_size=patch_size, step=step)])
        transformed_image = transforms_(image)

        patches = iSEGPipeline.get_patches(transformed_image, patch_size, step)

        if not os.path.exists(os.path.join(self._output_dir, subject, modality, "patches")):
            os.makedirs(os.path.join(self._output_dir, subject, modality, "patches"))

        for i, patch in enumerate(patches):
            x = transformed_image[tuple(patch.slice)]
            transforms_ = transforms.Compose(
                [ToNifti1Image(),
                 NiftiToDisk(os.path.join(self._output_dir, subject, modality, "patches", str(i) + ".nii.gz"))])
            transforms_(x)


class MRBrainSPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("MRBrainSPipeline")
    PATCH_SIZE = (1, 32, 32, 32)
    STEP = (1, 8, 8, 8)

    def __init__(self, root_dir, output_dir):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._normalized_shape = self._compute_normalized_shape_from_images_in(root_dir)

    def run(self):
        source_paths = list()

        for subject in sorted(os.listdir(os.path.join(self._root_dir))):
            source_paths.append(extract_file_paths(os.path.join(self._root_dir, subject)))

        self._dispatch_jobs(source_paths, 5)
        # self._do_job(source_paths)

    def _do_job(self, files):
        for file in files:
            subject = os.path.dirname(file[LABELSFORTESTING])[-1]

            self.LOGGER.info("Processing file {}".format(file[LABELSFORTESTING]))
            label_for_testing = self._resample_to_template(file[LABELSFORTESTING], file[T1_1MM], interpolation="linear")
            label_for_testing_full = np.expand_dims(label_for_testing.get_fdata(), 0)
            label_for_testing = self._crop_to_content(label_for_testing_full)
            label_for_testing = self._align(label_for_testing)
            self._extract_patches(label_for_testing, subject, "LabelsForTesting", self.PATCH_SIZE, self.STEP)

            self.LOGGER.info("Processing file {}".format(file[LABELSFORTRAINNG]))
            label_for_training = self._resample_to_template(file[LABELSFORTRAINNG], file[T1_1MM],
                                                            interpolation="linear")
            label_for_training = np.expand_dims(label_for_training.get_fdata(), 0)
            label_for_training = self._crop_to_content(label_for_training)
            label_for_training = self._align(label_for_training)
            self._extract_patches(label_for_training, subject, "LabelsForTraining", self.PATCH_SIZE, self.STEP)

            self.LOGGER.info("Processing file {}".format(file[T1]))
            t1 = self._resample_to_template(file[T1], file[T1_1MM], interpolation="continuous")
            t1 = np.expand_dims(t1.get_fdata(), 0)
            t1 = self._apply_mask(t1, label_for_testing_full)
            t1 = self._crop_to_content(t1)
            t1 = self._align(t1)
            self._extract_patches(t1, subject, "T1", self.PATCH_SIZE, self.STEP)

            self.LOGGER.info("Processing file {}".format(file[T1_IR]))
            t1_ir = self._resample_to_template(file[T1_IR], file[T1_1MM], interpolation="continuous")
            t1_ir = np.expand_dims(t1_ir.get_fdata(), 0)
            t1_ir = self._apply_mask(t1_ir, label_for_testing_full)
            t1_ir = self._crop_to_content(t1_ir)
            t1_ir = self._align(t1_ir)
            self._extract_patches(t1_ir, subject, "T1_IR", self.PATCH_SIZE, self.STEP)

            self.LOGGER.info("Processing file {}".format(file[T2_FLAIR]))
            t2 = self._resample_to_template(file[T2_FLAIR], file[T1_1MM], interpolation="continuous")
            t2 = np.expand_dims(t2.get_fdata(), 0)
            t2 = self._apply_mask(t2, label_for_testing_full)
            t2 = self._crop_to_content(t2)
            t2 = self._align(t2)
            self._extract_patches(t2, subject, "T2_FLAIR", self.PATCH_SIZE, self.STEP)

            self.LOGGER.info("Processing file {}".format(file[T1_1MM]))
            t1_1mm = self._to_numpy_array(file[T1_1MM])
            t1_1mm = self._apply_mask(t1_1mm, label_for_testing_full.transpose((0, 3, 1, 2)))
            t1_1mm = self._crop_to_content(t1_1mm)
            t1_1mm = self._align(t1_1mm)
            self._extract_patches(t1_1mm, subject, "T1_1mm", self.PATCH_SIZE, self.STEP)

    def _crop_to_content(self, file):
        if isinstance(file, np.ndarray):
            transforms_ = transforms.Compose([CropToContent(),
                                              PadToShape(self._normalized_shape)])
        else:
            transforms_ = transforms.Compose([ToNumpyArray(),
                                              CropToContent(),
                                              PadToShape(self._normalized_shape)])
        return transforms_(file)

    def _to_numpy_array(self, file):
        transform_ = transforms.Compose([ToNumpyArray()])

        return transform_(file)

    def _align(self, file):
        transform_ = transforms.Compose([Transpose((0, 2, 3, 1))])

        return transform_(file)

    def _resample_to_template(self, file, nifti_template, interpolation):
        transforms_ = transforms.Compose([LoadNifti(),
                                          ResampleNiftiImageToTemplate(clip=False,
                                                                       template=nifti_template,
                                                                       interpolation=interpolation)])
        return transforms_(file)

    def _apply_mask(self, file, mask):
        transform_ = transforms.Compose([ApplyMask(mask)])

        return transform_(file)

    def _extract_patches(self, image, subject, modality, patch_size, step):
        transforms_ = transforms.Compose([PadToPatchShape(patch_size=patch_size, step=step)])
        transformed_image = transforms_(image)

        patches = MRBrainSPipeline.get_patches(transformed_image, patch_size, step)

        if not os.path.exists(os.path.join(self._output_dir, subject, modality, "patches")):
            os.makedirs(os.path.join(self._output_dir, subject, modality, "patches"))

        for i, patch in enumerate(patches):
            x = transformed_image[tuple(patch.slice)]
            transforms_ = transforms.Compose(
                [ToNifti1Image(),
                 NiftiToDisk(os.path.join(self._output_dir, subject, modality, "patches", str(i) + ".nii.gz"))])
            transforms_(x)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)
    args = parser.parse_args()

    iSEGPipeline(args.path_iseg, "/mnt/md0/Data/Preprocessed_8/iSEG/Training").run()
    MRBrainSPipeline(args.path_mrbrains, "/mnt/md0/Data/Preprocessed_8/MRBrainS/DataNii/TrainingData").run()
