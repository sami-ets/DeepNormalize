#  -*- coding: utf-8 -*-
#  Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from samitorch.inputs.sample import Sample
from samitorch.utils.slice_builder import SliceBuilder
from samitorch.inputs.transformers import ToNumpyArray, ToNDTensor
from torchvision.transforms import Compose

from deepNormalize.inputs.images import SliceType


class SegmentationSlicer(object):
    DEFAULT_COLOR_MAP = plt.get_cmap('viridis')

    def __init__(self, colormap=None):
        self._colormap = colormap if colormap is not None else self.DEFAULT_COLOR_MAP

    def get_colored_slice(self, slice_type, seg_map):
        if slice_type == SliceType.SAGITAL:
            colored_slice = self._colormap(
                np.rot90(seg_map[:, :, int(seg_map.shape[2] / 2), :, :].squeeze(1), 2))
        elif slice_type == SliceType.CORONAL:
            colored_slice = self._colormap(
                np.rot90(seg_map[:, :, : int(seg_map.shape[3] / 2), :].squeeze(1), 2))
        elif slice_type == SliceType.AXIAL:
            colored_slice = self._colormap(seg_map[:, :, :, :, int(seg_map.shape[2] / 2)]).squeeze(1)
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return np.transpose(np.uint8(colored_slice[:, :, :, :3] * 255.0), axes=[0, 3, 1, 2])


class AdaptedImageSlicer(object):

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

        return slice


class ImageReconstructor(object):

    def __init__(self, image_size: List[int], patch_size: List[int], step: List[int], model: torch.nn.Module = None):
        self._patch_size = patch_size
        self._image_size = image_size
        self._step = step
        self._model = model
        self._transform = Compose([ToNumpyArray()])

    def reconstruct_from_patches_3d(self, patches: List[np.ndarray]):
        img = np.zeros(self._image_size)
        divisor = np.zeros(self._image_size)

        n_d = self._image_size[0] - self._patch_size[0] + 1
        n_h = self._image_size[1] - self._patch_size[1] + 1
        n_w = self._image_size[2] - self._patch_size[2] + 1

        for p, (z, y, x) in zip(patches, product(range(0, n_d, self._step[0]),
                                                 range(0, n_h, self._step[1]),
                                                 range(0, n_w, self._step[2]))):
            p = self._transform(p)
            p = np.expand_dims(p, 0)

            if self._model is not None:
                p = torch.Tensor().new_tensor(p, device="cuda:0")
                p = self._model.forward(p).cpu().detach().numpy()

            img[z:z + self._patch_size[0], y:y + self._patch_size[1], x:x + self._patch_size[2]] += p[0][0]
            divisor[z:z + self._patch_size[0], y:y + self._patch_size[1], x:x + self._patch_size[2]] += 1

        return img / divisor
