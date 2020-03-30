import argparse
import logging
import multiprocessing
from typing import Union

import abc
import nibabel as nib
import numpy as np
import os
import re
from samitorch.inputs.patch import Patch, CenterCoordinate
from samitorch.inputs.sample import Sample
from samitorch.inputs.transformers import ToNumpyArray, RemapClassIDs, ToNifti1Image, NiftiToDisk, ApplyMask, \
    ResampleNiftiImageToTemplate, LoadNifti, PadToPatchShape
from samitorch.utils.files import extract_file_paths
from samitorch.utils.slice_builder import SliceBuilder
from torchvision.transforms import transforms

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

    def __init__(self, root_dir, output_dir, step):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._step = step

    def run(self):
        images_T1 = natural_sort(extract_file_paths(os.path.join(self._root_dir, "T1")))
        images_T2 = natural_sort(extract_file_paths(os.path.join(self._root_dir, "T2")))
        labels = natural_sort(extract_file_paths(os.path.join(self._root_dir, "label")))
        files = np.stack((np.array(images_T1), np.array(images_T2), np.array(labels)), axis=1)

        self._dispatch_jobs(files, 5)
        # self._do_job(files)

    def _do_job(self, files):
        for file in files:
            subject = re.search(r"-(?P<subject_no>.*)-", file[0]).group("subject_no")
            self.LOGGER.info("Processing file {}".format(file[2]))
            label = self._to_numpy_array(file[2])
            label = self._remap_class_ids(label)
            self._extract_patches(label, subject, "Labels", self.PATCH_SIZE, self._step)
            self.LOGGER.info("Processing file {}".format(file[0]))
            t1 = self._to_numpy_array(file[0])
            self._extract_patches(t1, subject, "T1", self.PATCH_SIZE, self._step)
            self.LOGGER.info("Processing file {}".format(file[1]))
            t2 = self._to_numpy_array(file[1])
            self._extract_patches(t2, subject, "T2", self.PATCH_SIZE, self._step)

    def _to_numpy_array(self, file):
        transform_ = transforms.Compose([ToNumpyArray()])

        return transform_(file)

    def _remap_class_ids(self, file):
        transform_ = transforms.Compose([RemapClassIDs([10, 150, 250], [1, 2, 3])])
        return transform_(file)

    def _extract_patches(self, image, subject, modality, patch_size, step):
        transforms_ = transforms.Compose([PadToPatchShape(patch_size=patch_size, step=step)])
        transformed_image = transforms_(image)

        patches = iSEGPipeline.get_patches(transformed_image, patch_size, step)

        if not os.path.exists(os.path.join(self._output_dir, subject, modality)):
            os.makedirs(os.path.join(self._output_dir, subject, modality))

        for i, patch in enumerate(patches):
            x = transformed_image[tuple(patch.slice)]
            transforms_ = transforms.Compose(
                [ToNifti1Image(),
                 NiftiToDisk(os.path.join(self._output_dir, subject, modality, str(i) + ".nii.gz"))])
            transforms_(x)


class MRBrainSPipeline(AbstractPreProcessingPipeline):
    LOGGER = logging.getLogger("MRBrainSPipeline")
    PATCH_SIZE = (1, 32, 32, 32)

    def __init__(self, root_dir, output_dir, step):
        self._root_dir = root_dir
        self._output_dir = output_dir
        self._step = step

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
            label_for_testing = self._resample_to_template(file[LABELSFORTESTING], file[T1_1MM],
                                                           interpolation="linear")
            label_for_testing = self._to_numpy_array(label_for_testing)
            label_for_testing = label_for_testing.transpose((3, 0, 1, 2))
            label_for_testing = np.rot90(label_for_testing, axes=(1, -2))
            self._extract_patches(label_for_testing, subject, "LabelsForTesting", self.PATCH_SIZE, self._step)

            self.LOGGER.info("Processing file {}".format(file[LABELSFORTRAINNG]))
            label_for_training = self._resample_to_template(file[LABELSFORTRAINNG], file[T1_1MM],
                                                            interpolation="linear")
            label_for_training = self._to_numpy_array(label_for_training)
            label_for_training = label_for_training.transpose((3, 0, 1, 2))
            label_for_training = np.rot90(label_for_training, axes=(1, -2))
            self._extract_patches(label_for_training, subject, "LabelsForTraining", self.PATCH_SIZE, self._step)

            self.LOGGER.info("Processing file {}".format(file[T1]))
            t1 = self._resample_to_template(file[T1], file[T1_1MM], interpolation="continuous")
            t1 = self._to_numpy_array(t1)
            t1 = t1.transpose((3, 0, 1, 2))
            t1 = np.rot90(t1, axes=(1, -2))
            t1 = self._apply_mask(t1, label_for_testing)
            self._extract_patches(t1, subject, "T1", self.PATCH_SIZE, self._step)

            self.LOGGER.info("Processing file {}".format(file[T1_IR]))
            t1_ir = self._resample_to_template(file[T1_IR], file[T1_1MM], interpolation="continuous")
            t1_ir = self._to_numpy_array(t1_ir)
            t1_ir = t1_ir.transpose((3, 0, 1, 2))
            t1_ir = np.rot90(t1_ir, axes=(1, -2))
            t1_ir = self._apply_mask(t1_ir, label_for_testing)
            self._extract_patches(t1_ir, subject, "T1_IR", self.PATCH_SIZE, self._step)

            self.LOGGER.info("Processing file {}".format(file[T2_FLAIR]))
            t2 = self._resample_to_template(file[T2_FLAIR], file[T1_1MM], interpolation="continuous")
            t2 = self._to_numpy_array(t2)
            t2 = t2.transpose((3, 0, 1, 2))
            t2 = np.rot90(t2, axes=(1, -2))
            t2 = self._apply_mask(t2, label_for_testing)
            self._extract_patches(t2, subject, "T2_FLAIR", self.PATCH_SIZE, self._step)

            self.LOGGER.info("Processing file {}".format(file[T1_1MM]))
            t1_1mm = self._to_numpy_array(file[T1_1MM])
            t1_1mm = np.rot90(np.rot90(t1_1mm, axes=(-1, 1)), axes=(1, -2))
            t1_1mm = np.flip(t1_1mm, axis=-1)
            t1_1mm = self._apply_mask(t1_1mm, label_for_testing)
            self._extract_patches(t1_1mm, subject, "T1_1mm", self.PATCH_SIZE, self._step)

    def _to_numpy_array(self, file):
        transform_ = transforms.Compose([ToNumpyArray()])

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

        if not os.path.exists(os.path.join(self._output_dir, subject, modality)):
            os.makedirs(os.path.join(self._output_dir, subject, modality))

        for i, patch in enumerate(patches):
            x = transformed_image[tuple(patch.slice)]
            transforms_ = transforms.Compose(
                [ToNifti1Image(),
                 NiftiToDisk(os.path.join(self._output_dir, subject, modality, str(i) + ".nii.gz"))])
            transforms_(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the MRBrainS directory.', required=True)
    args = parser.parse_args()

    iSEGPipeline(args.path_iseg, "/mnt/md0/Data/Preprocessed_4/iSEG/Training",
                 step=(1, 4, 4, 4)).run()
    MRBrainSPipeline(args.path_mrbrains, "/mnt/md0/Data/Preprocessed_4/MRBrainS/DataNii/TrainingData",
                     step=(1, 4, 4, 4)).run()

    iSEGPipeline(args.path_iseg, "/mnt/md0/Data/Preprocessed_8/iSEG/Training",
                 step=(1, 8, 8, 8)).run()
    MRBrainSPipeline(args.path_mrbrains, "/mnt/md0/Data/Preprocessed_8/MRBrainS/DataNii/TrainingData",
                     step=(1, 8, 8, 8)).run()
