import unittest

import matplotlib.pyplot as plt
import numpy as np
from samitorch.inputs.images import Modality
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape, ToNDTensor
from samitorch.utils.files import extract_file_paths
from torchvision.transforms import Compose

from deepNormalize.inputs.datasets import iSEGSegmentationFactory, iSEGSliceDatasetFactory, MRBrainSSegmentationFactory, \
    ABIDESegmentationFactory
from deepNormalize.utils.image_slicer import ImageReconstructor
from deepNormalize.utils.utils import natural_sort


class ImageReconstructorTest(unittest.TestCase):
    PATH = "/mnt/md0/Data/Preprocessed/iSEG/Training/Patches/Aligned/Full/T1/10"
    # TARGET_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Patches/Aligned/label/6"
    FULL_IMAGE_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Training/Aligned/T1/subject-10-T1.nii"

    def setUp(self) -> None:
        paths = extract_file_paths(self.PATH)
        self._dataset = iSEGSegmentationFactory.create(natural_sort(paths), None, modalities=Modality.T1,
                                                       dataset_id=0)
        self._reconstructor = ImageReconstructor([128, 160, 128], [32, 32, 32], [8, 8, 8])
        transforms = Compose([ToNumpyArray(), PadToPatchShape([1, 32, 32, 32], [1, 8, 8, 8])])
        self._full_image = transforms(self.FULL_IMAGE_PATH)

    def test_should_output_reconstructed_image(self):
        all_patches = []
        all_labels = []
        for current_batch, input in enumerate(self._dataset):
            all_patches.append(input.x)
            all_labels.append(input.y)
        img = self._reconstructor.reconstruct_from_patches_3d(all_patches)
        plt.imshow(img[64, :, :], cmap="gray")
        plt.show()
        np.testing.assert_array_almost_equal(img, self._full_image.squeeze(0), 6)


class ImageReconstructorMRBrainSTest(unittest.TestCase):
    PATH = "/mnt/md0/Data/Preprocessed_4/MRBrainS/DataNii/TrainingData/1/T1"
    # TARGET_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Patches/Aligned/label/6"
    FULL_IMAGE_PATH = "/mnt/md0/Data/Preprocessed/MRBrainS/DataNii/TrainingData/1/T1/T1.nii.gz"

    def setUp(self) -> None:
        paths = extract_file_paths(self.PATH)
        self._dataset = MRBrainSSegmentationFactory.create(natural_sort(paths), None, modalities=Modality.T1,
                                                           dataset_id=0)
        self._reconstructor = ImageReconstructor([256, 256, 192], [1, 32, 32, 32], [1, 8, 8, 8])
        transforms = Compose([ToNumpyArray(), PadToPatchShape([1, 32, 32, 32], [1, 8, 8, 8])])
        self._full_image = transforms(self.FULL_IMAGE_PATH)

    def test_should_output_reconstructed_image(self):
        all_patches = []
        all_labels = []
        for current_batch, input in enumerate(self._dataset):
            all_patches.append(input.x)
            all_labels.append(input.y)
        img = self._reconstructor.reconstruct_from_patches_3d(all_patches)
        plt.imshow(img[64, :, :], cmap="gray")
        plt.show()
        np.testing.assert_array_almost_equal(img, self._full_image.squeeze(0), 6)


class ImageReconstructorABIDETest(unittest.TestCase):
    PATH = "/home/pierre-luc-delisle/ABIDE/5.1/Stanford_0051160/mri/patches/image"
    # TARGET_PATH = "/home/pierre-luc-delisle/ABIDE/5.1/Stanford_0051160/mri/patches/labels"
    FULL_IMAGE_PATH = "/home/pierre-luc-delisle/ABIDE/5.1/Stanford_0051160/mri/real_brainmask.nii.gz"

    def setUp(self) -> None:
        paths = extract_file_paths(self.PATH)
        self._dataset = ABIDESegmentationFactory.create(natural_sort(paths), None, modalities=Modality.T1,
                                                        dataset_id=0)
        self._reconstructor = ImageReconstructor([224, 224, 192], [1, 32, 32, 32], [1, 8, 8, 8])
        transforms = Compose([ToNumpyArray(), PadToPatchShape([1, 32, 32, 32], [1, 8, 8, 8])])
        self._full_image = transforms(self.FULL_IMAGE_PATH)

    def test_should_output_reconstructed_image(self):
        all_patches = []
        all_labels = []
        for current_batch, input in enumerate(self._dataset):
            all_patches.append(input.x)
            all_labels.append(input.y)
        img = self._reconstructor.reconstruct_from_patches_3d(all_patches)
        plt.imshow(img[112, :, :], cmap="gray")
        plt.show()
        np.testing.assert_array_almost_equal(img, self._full_image.squeeze(0), 6)
        plt.imshow(self._full_image.squeeze(0)[112, :, :], cmap="gray")
        plt.show()


class SlicedImageReconstructorTest(unittest.TestCase):
    FULL_IMAGE_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Training/1/T1/T1.nii.gz"
    TARGET_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Training/1/Labels/Labels.nii.gz"

    def setUp(self) -> None:
        transforms = Compose([ToNumpyArray(), PadToPatchShape((1, 32, 32, 32), (1, 8, 8, 8))])
        self._image = transforms(self.FULL_IMAGE_PATH)
        self._target = transforms(self.TARGET_PATH)
        patches = iSEGSliceDatasetFactory.get_patches([self._image], [self._target], (1, 32, 32, 32), (1, 16, 16, 16))
        self._dataset = iSEGSliceDatasetFactory.create([self._image], [self._target], patches, Modality.T1, 0,
                                                       transforms=[ToNDTensor()])
        self._reconstructor = ImageReconstructor([256, 192, 160], [1, 32, 32, 32], [1, 16, 16, 16], models=None,
                                                 test_image=self._image)

    def test_should_output_reconstructed_image(self):
        all_patches = list(map(lambda dataset: [patch.slice for patch in dataset._patches],
                               [self._dataset]))
        img = self._reconstructor.reconstruct_from_patches_3d(all_patches[0])
        plt.imshow(img[150, :, :], cmap="gray")
        plt.show()
        np.testing.assert_array_almost_equal(img, self._image.squeeze(0), 6)
