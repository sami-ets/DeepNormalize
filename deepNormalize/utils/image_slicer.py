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
from samitorch.inputs.transformers import ToNumpyArray
from torchvision.transforms import Compose

from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import EPSILON


class SegmentationSlicer(object):
    DEFAULT_COLOR_MAP = plt.get_cmap('viridis')

    def __init__(self, colormap=None):
        self._colormap = colormap if colormap is not None else self.DEFAULT_COLOR_MAP

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def get_colored_slice(self, slice_type, seg_map):
        seg_map = self._normalize(seg_map)

        if slice_type == SliceType.SAGITAL:
            colored_slice = self._colormap(np.rot90(seg_map[:, :, :, :, int(seg_map.shape[4] / 2)]), 2)
        elif slice_type == SliceType.CORONAL:
            colored_slice = self._colormap(np.rot90(seg_map[:, :, :, int(seg_map.shape[3] / 2), :]), 2)
        elif slice_type == SliceType.AXIAL:
            colored_slice = self._colormap((seg_map[:, :, int(seg_map.shape[2] / 2), :, :])).squeeze(1)
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return np.transpose(np.uint8(colored_slice[:, :, :, :3] * 255.0), axes=[0, 3, 1, 2])


class ImageSlicer(object):

    def __init__(self):
        pass

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def get_slice(self, slice_type, image):
        image = self._normalize(image)

        if slice_type == SliceType.SAGITAL:
            slice = image[:, :, :, :, int(image.shape[4] / 2)]
        elif slice_type == SliceType.CORONAL:
            slice = image[:, :, :, int(image.shape[3] / 2), :]
        elif slice_type == SliceType.AXIAL:
            slice = image[:, :, int(image.shape[2] / 2), :, :]
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return slice


class ImageReconstructor(object):

    def __init__(self, image_size: List[int], patch_size: List[int], step: List[int],
                 models: List[torch.nn.Module] = None, normalize: bool = False, segment: bool = False):
        self._patch_size = patch_size
        self._image_size = image_size
        self._step = step
        self._models = models
        self._do_normalize = normalize
        self._do_segment = segment
        self._transform = Compose([ToNumpyArray()])

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

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

            if self._models is not None:
                p = torch.Tensor().new_tensor(p, device="cuda:0")

                if self._do_normalize:
                    p = self._models[0].forward(p).cpu().detach().numpy()
                elif self._do_segment:
                    p = self._models[0].forward(p)
                    p = torch.argmax(torch.nn.functional.softmax(self._models[1].forward(p), dim=1), dim=1,
                                     keepdim=True).float().cpu().detach().numpy()

            img[z:z + self._patch_size[0], y:y + self._patch_size[1], x:x + self._patch_size[2]] += p[0][0]
            divisor[z:z + self._patch_size[0], y:y + self._patch_size[1], x:x + self._patch_size[2]] += 1

        if self._do_segment:
            return np.floor(img / divisor)
        else:
            return img / divisor
