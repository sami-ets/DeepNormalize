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
import matplotlib.pyplot as plt
import numpy as np

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
