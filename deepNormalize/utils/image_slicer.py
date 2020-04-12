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
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from samitorch.inputs.transformers import ToNumpyArray
from torchvision.transforms import Compose

from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import EPSILON, ISEG_ID, MRBRAINS_ID, ABIDE_ID


class LabelMapper(object):
    DEFAULT_COLOR_MAP = plt.get_cmap('jet')

    def __init__(self, colormap=None):
        self._colormap = colormap if colormap is not None else self.DEFAULT_COLOR_MAP

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def get_label_map(self, dataset_ids: torch.Tensor):
        label_map = np.zeros((dataset_ids.size(0), 160, 160))

        label_map[torch.where(dataset_ids == ISEG_ID)] = ISEG_ID + 1
        label_map[torch.where(dataset_ids == MRBRAINS_ID)] = MRBRAINS_ID + 1
        label_map[torch.where(dataset_ids == ABIDE_ID)] = ABIDE_ID + 1

        label_map = self._normalize(label_map)

        colored_label_map = self._colormap(label_map)

        return np.transpose(np.uint8(colored_label_map[:, :, :, :3] * 255.0), axes=[0, 3, 1, 2])


class FeatureMapSlicer(object):
    DEFAULT_COLOR_MAP = plt.get_cmap('jet')

    def __init__(self, colormap=None):
        self._colormap = colormap if colormap is not None else self.DEFAULT_COLOR_MAP

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def get_colored_slice(self, slice_type, feature_map):
        feature_map = self._normalize(feature_map)

        if slice_type == SliceType.SAGITAL:
            colored_slice = self._colormap(np.rot90(feature_map[:, :, :, :, int(feature_map.shape[4] / 2)]), 2)
        elif slice_type == SliceType.CORONAL:
            colored_slice = self._colormap(np.rot90(feature_map[:, :, :, int(feature_map.shape[3] / 2), :]), 2)
        elif slice_type == SliceType.AXIAL:
            colored_slice = self._colormap((feature_map[:, :, int(feature_map.shape[2] / 2), :, :])).squeeze(0)
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return np.transpose(np.uint8(colored_slice[:, :, :, :3] * 255.0), axes=[0, 3, 1, 2])


class SegmentationSlicer(object):
    DEFAULT_COLOR_MAP = plt.get_cmap('viridis')

    def __init__(self, colormap=None):
        self._colormap = colormap if colormap is not None else self.DEFAULT_COLOR_MAP

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def get_colored_slice(self, slice_type, seg_map, slice):
        seg_map = self._normalize(seg_map)

        if slice_type == SliceType.SAGITAL:
            colored_slice = self._colormap(np.rot90(seg_map[:, :, :, :, slice]), 2)
        elif slice_type == SliceType.CORONAL:
            colored_slice = self._colormap(np.rot90(seg_map[:, :, :, slice, :]), 2)
        elif slice_type == SliceType.AXIAL:
            colored_slice = self._colormap((seg_map[:, :, slice, :, :])).squeeze(1)
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return np.transpose(np.uint8(colored_slice[:, :, :, :3] * 255.0), axes=[0, 3, 1, 2])


class ImageSlicer(object):

    def __init__(self):
        pass

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def get_slice(self, slice_type, image, slice):
        image = self._normalize(image)

        if slice_type == SliceType.SAGITAL:
            slice = image[:, :, :, :, slice]
        elif slice_type == SliceType.CORONAL:
            slice = image[:, :, :, slice, :]
        elif slice_type == SliceType.AXIAL:
            slice = image[:, :, slice, :, :]
        else:
            raise NotImplementedError("The provided slice type ({}) not found.".format(slice_type))

        return slice


# class ImageReconstructor(object):
#
#     def __init__(self, image_size: List[int], patch_size: List[int], step: List[int],
#                  models: List[torch.nn.Module] = None, normalize: bool = False, segment: bool = False):
#         self._patch_size = patch_size
#         self._image_size = image_size
#         self._step = step
#         self._models = models
#         self._do_normalize = normalize
#         self._do_segment = segment
#         self._transform = Compose([ToNumpyArray()])
#
#     @staticmethod
#     def _normalize(img):
#         return (img - np.min(img)) / (np.ptp(img) + EPSILON)
#
#     def reconstruct_from_patches_3d(self, patches: List[np.ndarray]):
#         img = np.zeros(self._image_size)
#         divisor = np.zeros(self._image_size)
#
#         n_d = self._image_size[0] - self._patch_size[0] + 1
#         n_h = self._image_size[1] - self._patch_size[1] + 1
#         n_w = self._image_size[2] - self._patch_size[2] + 1
#
#         for p, (z, y, x) in zip(patches, product(range(0, n_d, self._step[0]),
#                                                  range(0, n_h, self._step[1]),
#                                                  range(0, n_w, self._step[2]))):
#             if not isinstance(p, np.ndarray):
#                 p = self._transform(p)
#                 p = np.expand_dims(p, 0)
#
#             if self._models is not None:
#                 p = torch.Tensor().new_tensor(p, device="cuda:0")
#
#                 if self._do_normalize:
#                     p = self._models[0].forward(p).cpu().detach().numpy()
#                 elif self._do_segment:
#                     p = self._models[0].forward(p)
#                     p = torch.argmax(torch.nn.functional.softmax(self._models[1].forward(p), dim=1), dim=1,
#                                      keepdim=True).float().cpu().detach().numpy()
#
#             img[z:z + self._patch_size[0], y:y + self._patch_size[1], x:x + self._patch_size[2]] += p[0][0]
#             divisor[z:z + self._patch_size[0], y:y + self._patch_size[1], x:x + self._patch_size[2]] += 1
#
#         if self._do_segment:
#             return np.floor(img / divisor)
#         else:
#             return img / divisor


class ImageReconstructor(object):

    def __init__(self, image_size: List[int], patch_size: List[int], step: List[int],
                 models: List[torch.nn.Module] = None, normalize: bool = False, segment: bool = False,
                 test_image: np.ndarray = None):
        self._patch_size = patch_size
        self._image_size = image_size
        self._step = step
        self._models = models
        self._do_normalize = normalize
        self._do_segment = segment
        self._transform = Compose([ToNumpyArray()])
        self._test_image = test_image

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def reconstruct_from_patches_3d(self, patches: Union[List[np.ndarray], List[slice]]):
        img = np.zeros(self._image_size)
        divisor = np.zeros(self._image_size)

        n_d = self._image_size[0] - self._patch_size[1] + 1
        n_h = self._image_size[1] - self._patch_size[2] + 1
        n_w = self._image_size[2] - self._patch_size[3] + 1

        for p, (z, y, x) in zip(patches, product(range(0, n_d, self._step[1]),
                                                 range(0, n_h, self._step[2]),
                                                 range(0, n_w, self._step[3]))):
            if isinstance(p, tuple):
                p = self._test_image[p]
                p = np.expand_dims(p, 0)

            elif not isinstance(p, np.ndarray):
                p = self._transform(p)
                p = np.expand_dims(p, 0)

            if self._models is not None:
                p = torch.Tensor().new_tensor(p, device="cuda:0")

                if self._do_normalize:
                    if len(p.size()) < 5:
                        p = torch.unsqueeze(p, 0)
                    p = self._models[0].forward(p).cpu().detach().numpy()
                elif self._do_segment:
                    if len(p.size()) < 5:
                        p = torch.unsqueeze(p, 0)
                    p = self._models[0].forward(p)
                    p = torch.argmax(torch.nn.functional.softmax(self._models[1].forward(p), dim=1), dim=1,
                                     keepdim=True).float().cpu().detach().numpy()

            img[z:z + self._patch_size[1], y:y + self._patch_size[2], x:x + self._patch_size[3]] += p[0][0]
            divisor[z:z + self._patch_size[1], y:y + self._patch_size[2], x:x + self._patch_size[3]] += 1

        if self._do_segment:
            return np.clip(np.round(img / divisor), a_min=0, a_max=3)
        else:
            return img / divisor
