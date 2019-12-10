import unittest

import nibabel as nib
import numpy as np
from samitorch.inputs.images import Modality
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape

from deepNormalize.inputs.datasets import iSEGSegmentationFactory
from deepNormalize.utils.image_slicer import ImageReconstructor
from torchvision.transforms import Compose
import matplotlib.pyplot as plt


class ImageReconstructorTest(unittest.TestCase):
    PATH = "/mnt/md0/Data/Preprocessed/iSEG/TestingData/Patches/Aligned/T1/11"
    # TARGET_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Patches/Aligned/label/6"
    FULL_IMAGE_PATH = "/mnt/md0/Data/Preprocessed/iSEG/TestingData/Aligned/T1/subject-11-T1.nii"

    def setUp(self) -> None:
        self._dataset = iSEGSegmentationFactory.create(self.PATH, None, modality=Modality.T1,
                                                       dataset_id=0)
        self._reconstructor = ImageReconstructor([128, 160, 160], [32, 32, 32], [8, 8, 8])
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
