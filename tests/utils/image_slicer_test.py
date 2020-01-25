import unittest

import nibabel as nib
import numpy as np
from samitorch.inputs.images import Modality
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape

from deepNormalize.inputs.datasets import iSEGSegmentationFactory, ABIDESegmentationFactory
from deepNormalize.utils.image_slicer import ImageReconstructor
from deepNormalize.utils.utils import natural_sort
from samitorch.utils.files import extract_file_paths
from torchvision.transforms import Compose
import matplotlib.pyplot as plt


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
    PATH = "/mnt/md0/Data/Preprocessed/MRBrainS/TrainingData/Patches/Aligned/Full/5/T1_1mm"
    # TARGET_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Patches/Aligned/label/6"
    FULL_IMAGE_PATH = "/mnt/md0/Data/Preprocessed/MRBrainS/TrainingData/Aligned/5/T1_1mm.nii"

    def setUp(self) -> None:
        paths = extract_file_paths(self.PATH)
        self._dataset = iSEGSegmentationFactory.create(natural_sort(paths), None, modalities=Modality.T1,
                                                       dataset_id=0)
        self._reconstructor = ImageReconstructor([160, 192, 160], [32, 32, 32], [8, 8, 8])
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