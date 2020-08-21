from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from kerosene.utils.constants import EPSILON

from deepNormalize.inputs.images import SliceType


class SliceBuilder(object):
    """
    Build slices for a data set.
    """

    def __init__(self, image_shape: Tuple[int, int, int, int], patch_size: Tuple[int, int, int, int],
                 step: Tuple[int, int, int, int]):
        """
        Args:
            image_shape (tuple of int): The shape of a dataset image.
            patch_size(tuple of int): The size of the patch to produce.
            step (tuple of int): The size of the stride between patches.
        """

        self._image_shape = image_shape
        self._patch_size = patch_size
        self._step = step
        self._slices = None

    @property
    def image_shape(self) -> Tuple:
        """
        Input image's shape
        Returns:
            Tuple of int: The image's shape.
        """
        return self._image_shape

    @property
    def patch_size(self) -> Tuple:
        """
        Patch size.
        Returns:
            Tuple of int: The patch size.
        """
        return self._patch_size

    @property
    def step(self) -> Tuple:
        """
        Step between two patches.
        Returns:
            Tuple of int: The step in each direction.
        """
        return self._step

    @property
    def slices(self) -> list:
        """
        Image's slices.
        Returns:
            list: List of image's slices.
        """
        return self._slices

    def build_slices(self) -> list:
        """
        Iterates over a given n-dim dataset patch-by-patch with a given step and builds an array of slice positions.
        Returns:
            list: list of slices.
        """
        slices = []
        channels, i_z, i_y, i_x = self._image_shape
        k_c, k_z, k_y, k_x = self._patch_size
        s_c, s_z, s_y, s_x = self._step
        z_steps = SliceBuilder.gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder.gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder.gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if len(self._image_shape) == 4:
                        slice_idx = (slice(0, channels),) + slice_idx
                    slices.append(slice_idx)

        self._slices = slices

        return slices

    def build_overlap_map(self):
        map = np.zeros(self.image_shape)
        patch = np.ones(self.patch_size)
        slices = self.build_slices()

        for slice in slices:
            map[slice] += patch

        return 1.0 / map

    @staticmethod
    def gen_indices(i: int, k: int, s: int):
        """
        Generate slice indices.
        Args:
            i (int): image's coordinate.
            k (int): patch size.
            s (int): step size.
        Returns:
            generator: A generator of indices.
        """
        assert i >= k, 'Sample size has to be bigger than the patch size.'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


class PadToPatchShape(object):
    def __init__(self, patch_size: Union[int, Tuple[int, int, int, int]], step: Union[int, Tuple[int, int, int, int]]):
        """
        Transformer initializer.
        Args:
            patch_size (int or Tuple of int):  The size of the patch to produce.
            step (int or Tuple of int):  The size of the stride between patches.
        """
        self._patch_size = patch_size
        self._step = step

    def __call__(self, input: Union[np.ndarray]) -> Union[np.ndarray]:
        if isinstance(input, np.ndarray):

            for i in range(1, input.ndim):
                if not input.shape[i] >= self._patch_size[i]:
                    raise ValueError("Shape incompatible with patch_size parameter.")

            c, d, h, w, = input.shape

            pad_d, pad_h, pad_w = 0, 0, 0

            if d % self._patch_size[1] != 0:
                pad_d = ((self._patch_size[1] - d % self._patch_size[1]) / 2)
            if h % self._patch_size[2] != 0:
                pad_h = ((self._patch_size[2] - h % self._patch_size[2]) / 2)
            if w % self._patch_size[3] != 0:
                pad_w = ((self._patch_size[3] - w % self._patch_size[3]) / 2)

            if pad_d != 0 or pad_h != 0 or pad_w != 0:
                input = np.pad(input, ((0, 0),
                                       (int(np.ceil(pad_d)), int(np.floor(pad_d))),
                                       (int(np.ceil(pad_h)), int(np.floor(pad_h))),
                                       (int(np.ceil(pad_w)), int(np.floor(pad_w)))),
                               mode="constant", constant_values=0)

            return input


class ImageSlicer(object):

    def __init__(self):
        pass

    @staticmethod
    def get_slice(slice_type, image):
        if slice_type == SliceType.SAGITAL:
            slice = np.rot90(image[:, :, int(image.shape[2] / 2), :, :], 2)
        elif slice_type == SliceType.CORONAL:
            slice = np.rot90(image[:, :, :, int(image.shape[3] / 2), :], 2)
        elif slice_type == SliceType.AXIAL:
            slice = image[:, :, :, :, int(image.shape[4] / 2)]
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return ImageSlicer.normalize(slice)

    @staticmethod
    def get_dti_slice(slice_type, image, log=True):
        b, c, d, h, w = image.size()
        image = image.reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)
        U, S, V = torch.svd(image.cpu())

        if log:
            S = torch.exp(S)

        num = torch.pow((S[:, 0] - S[:, 1]), 2) + torch.pow((S[:, 1] - S[:, 2]), 2) + torch.pow(
            (S[:, 0] - S[:, 2]), 2)
        denom = 2 * (torch.pow(S[:, 0], 2) + torch.pow(S[:, 1], 2) + torch.pow(S[:, 2], 2))

        fa = (torch.abs(U[:, :, 0]) * torch.clamp(torch.pow(num / denom, 0.5), 0, 1).unsqueeze(1)).reshape(b, d, h, w,
                                                                                                           3)
        image = F.interpolate(fa.permute(0, 4, 1, 2, 3), scale_factor=int(160 / d), mode="trilinear",
                              align_corners=True).numpy()

        if slice_type == SliceType.SAGITAL:
            slice = np.rot90(image[:, :, int(image.shape[2] / 2), :, :], 2)
        elif slice_type == SliceType.CORONAL:
            slice = np.rot90(image[:, :, :, int(image.shape[3] / 2), :], 2)
        elif slice_type == SliceType.AXIAL:
            slice = image[:, :, :, :, int(image.shape[4] / 2)]
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return slice

    @staticmethod
    def normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)
