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

import math
import nibabel as nib
import nrrd
import numpy as np

from nilearn.image.resampling import resample_to_img

from deepNormalize.input.images import Image, ImageType


class ToNumpyArray(object):
    """
    Creates a numpy ndarray from a given Nifti or NRRD image file path.

    The numpy arrays are transposed to respect the standard dimensions (DxHxW) for 3D or (CxDxHxW) for 4D arrays.
    """

    def __call__(self, image_path: str):
        if Image.is_nifti(image_path):
            nd_array = nib.load(image_path).get_fdata().__array__()
        elif Image.is_nrrd(image_path):
            nd_array, header = nrrd.read(image_path)
        else:
            raise NotImplementedError("Only {} files are supported but got {} !".format(ImageType.ALL, image_path))

        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 3).transpose((3, 2, 1, 0))
        elif nd_array.ndim == 4:
            nd_array = nd_array.transpose((3, 2, 1, 0))

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNifti1Image(object):

    def __call__(self, image_path: str):
        if Image.is_nifti(image_path):
            return nib.load(image_path)
        else:
            raise NotImplementedError("Only {} files are supported !".format(ImageType.NIFTI))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NiftiImageToNumpy(object):

    def __call__(self, image: nib.Nifti1Image):
        if Image.is_nifti(image):
            nd_array = image.get_fdata().__array__()

        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 3).transpose((3, 2, 1, 0))
        elif nd_array.ndim == 4:
            nd_array = nd_array.transpose((3, 2, 1, 0))

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ResampleToImg(object):

    def __init__(self, clip: bool, reference_nii_file: nib.nifti1, interpolation: str):
        self._clip = clip
        self._reference_nii_file = reference_nii_file
        self._interpolation = interpolation

    def __call__(self, nifti_image: nib.Nifti1Image):
        return resample_to_img(nifti_image, nib.load(self._reference_nii_file), clip=self._clip,
                               interpolation=self._interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ExtractBrainNifti(object):

    def __init__(self, mask: str):
        nd_array = nib.load(mask).get_fdata().__array__()
        nd_array[nd_array >= 1] = 1
        self._mask = nd_array

    def __call__(self, image: nib.Nifti1Image):
        header = image.header
        brain = np.multiply(image.get_fdata(), self._mask)
        return nib.Nifti1Image(brain, None, header)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ExtractBrain(object):

    def __init__(self, mask: str):
        nd_array = nib.load(mask).get_fdata().__array__()
        if nd_array.ndim == 3:
            nd_array = np.expand_dims(nd_array, 3).transpose((3, 2, 1, 0))
        elif nd_array.ndim == 4:
            nd_array = nd_array.transpose((3, 2, 1, 0))
        nd_array[nd_array >= 1] = 1
        self._mask = nd_array

    def __call__(self, nd_array: np.ndarray):
        return np.multiply(nd_array, self._mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RemapClassIDs(object):

    def __init__(self, initial_ids: list, final_ids: list):
        if not isinstance(initial_ids, list) and isinstance(final_ids, list):
            raise TypeError(
                "Initial and final IDs must be a list of integers, but got {} and {}".format(type(initial_ids),
                                                                                             type(final_ids)))
        self._initial_ids = initial_ids
        self._final_ids = final_ids

    def __call__(self, nd_array: np.ndarray):
        for i, id in enumerate(self._initial_ids):
            nd_array[nd_array == id] = self._final_ids[i]

        return nd_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Nifti1ToDisk(object):
    def __init__(self, file_path: str):
        self._file_path = file_path

    def __call__(self, image: nib.Nifti1Image):
        nib.save(image, self._file_path)


class ToNiftiFile(object):
    """
    Creates a Nifti1Image from a given numpy ndarray

    The numpy arrays are transposed to respect the standard Nifti dimensions (WxHxDxC)
    """

    def __init__(self, file_path: str, header: nib.Nifti1Header):
        self._file_path = file_path
        self._header = header

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        nifti1_file = nib.Nifti1Image(nd_array.transpose((3, 2, 1, 0)), None, self._header)
        nib.save(nifti1_file, self._file_path)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropToContent(object):
    """
    Crops the image to its content.

    The content's bounding box is defined by the first non-zero slice in each direction (D, H, W)
    """

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        d_min, d_max, h_min, h_max, w_min, w_max = self.extract_content_bounding_box_from(nd_array)

        return nd_array[:, d_min:d_max, h_min:h_max, w_min:w_max] if nd_array.ndim is 4 else \
            nd_array[d_min:d_max, h_min:h_max, w_min:w_max]

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def extract_content_bounding_box_from(nd_array):
        """
        Computes the D, H, W min and max values defining the content bounding box.

        :param nd_array: The input image as a numpy ndarray
        :return: The D, H, W min and max values of the bounding box.
        """

        depth_slices = np.any(nd_array, axis=(2, 3))
        height_slices = np.any(nd_array, axis=(1, 3))
        width_slices = np.any(nd_array, axis=(1, 2))

        d_min, d_max = np.where(depth_slices)[1][[0, -1]]
        h_min, h_max = np.where(height_slices)[1][[0, -1]]
        w_min, w_max = np.where(width_slices)[1][[0, -1]]

        return d_min, d_max, h_min, h_max, w_min, w_max


class PadToShape(object):

    def __init__(self, target_shape, padding_value=0):
        self._target_shape = target_shape
        self._padding_value = padding_value

    def __call__(self, nd_array):
        if not isinstance(nd_array, np.ndarray) or (nd_array.ndim not in [3, 4]):
            raise TypeError("Only 3D (DxHxW) or 4D (CxDxHxW) ndarrays are supported")

        return self.apply(nd_array, self._target_shape, self._padding_value)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def apply(nd_array, target_shape, padding_value):
        deltas = tuple(max(0, target - current) for target, current in zip(target_shape, nd_array.shape))

        if nd_array.ndim == 3:
            nd_array = np.pad(nd_array, ((math.floor(deltas[0] / 2), math.ceil(deltas[0] / 2)),
                                         (math.floor(deltas[1] / 2), math.ceil(deltas[1] / 2)),
                                         (math.floor(deltas[2] / 2), math.ceil(deltas[2] / 2))),
                              'constant', constant_values=padding_value)
        elif nd_array.ndim == 4:
            nd_array = np.pad(nd_array, ((0, 0),
                                         (math.floor(deltas[1] / 2), math.ceil(deltas[1] / 2)),
                                         (math.floor(deltas[2] / 2), math.ceil(deltas[2] / 2)),
                                         (math.floor(deltas[3] / 2), math.ceil(deltas[3] / 2))),
                              'constant', constant_values=padding_value)
        return nd_array

    @staticmethod
    def undo(nd_array, original_shape):
        deltas = tuple(max(0, current - target) for target, current in zip(original_shape, nd_array.shape))

        if nd_array.ndim == 3:
            nd_array = nd_array[
                       math.floor(deltas[0] / 2):-math.ceil(deltas[0] / 2),
                       math.floor(deltas[1] / 2):-math.ceil(deltas[1] / 2),
                       math.floor(deltas[2] / 2):-math.ceil(deltas[2] / 2)]
        elif nd_array.ndim == 4:
            nd_array = nd_array[
                       :,
                       math.floor(deltas[1] / 2):-math.ceil(deltas[1] / 2),
                       math.floor(deltas[2] / 2):-math.ceil(deltas[2] / 2),
                       math.floor(deltas[3] / 2):-math.ceil(deltas[3] / 2)]
        return nd_array
