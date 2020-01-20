import unittest

import nibabel as nib
import numpy as np
from samitorch.inputs.images import Modality
from samitorch.inputs.transformers import ToNumpyArray, PadToPatchShape

from deepNormalize.inputs.datasets import iSEGSegmentationFactory, ABIDESegmentationFactory
from deepNormalize.utils.image_slicer import ImageReconstructor
from torchvision.transforms import Compose
import matplotlib.pyplot as plt


class ImageReconstructorTest(unittest.TestCase):
    PATH = "/home/pierre-luc-delisle/ABIDE/5.1/Stanford_0051160"
    # TARGET_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Patches/Aligned/label/6"
    FULL_IMAGE_PATH = "/home/pierre-luc-delisle/ABIDE/5.1/Stanford_0051160/mri/aligned_brainmask.nii.gz"

    def setUp(self) -> None:
        self._dataset = ABIDESegmentationFactory.create(self.PATH, Modality.T1,
                                                       dataset_id=0)
        self._reconstructor = ImageReconstructor([224, 224, 192], [32, 32, 32], [8, 8, 8])
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
