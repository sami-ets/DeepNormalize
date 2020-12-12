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
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from samitorch.inputs.transformers import ToNumpyArray, PadToShape
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from deepNormalize.inputs.datasets import SingleNDArrayDataset
from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import EPSILON
from deepNormalize.utils.constants import ISEG_ID, MRBRAINS_ID, ABIDE_ID, GENERATOR
from deepNormalize.utils.slices import SliceBuilder

PATCH = 0
SLICE = 1
SEGMENTER = 1


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
#                  models: List[torch.nn.Module] = None, normalize: bool = False, segment: bool = False,
#                  normalize_and_segment: bool = False, test_image: np.ndarray = None, is_multimodal=False):
#         self._patch_size = patch_size
#         self._image_size = image_size
#         self._step = step
#         self._models = models
#         self._do_normalize = normalize
#         self._do_segment = segment
#         self._do_normalize_and_segment = normalize_and_segment
#         self._transform = Compose([ToNumpyArray()])
#         self._test_image = test_image
#         self._is_multimodal = is_multimodal
#
#     @staticmethod
#     def _normalize(img):
#         return (img - np.min(img)) / (np.ptp(img) + EPSILON)
#
#     def reconstruct_from_patches_3d(self, patches: Union[List[np.ndarray], List[slice]]):
#         img = np.zeros(self._image_size)
#         divisor = np.zeros(self._image_size)
#
#         if self._is_multimodal:
#             n_d = self._image_size[1] - self._patch_size[1] + 1
#             n_h = self._image_size[2] - self._patch_size[2] + 1
#             n_w = self._image_size[3] - self._patch_size[3] + 1
#         else:
#             n_d = self._image_size[0] - self._patch_size[1] + 1
#             n_h = self._image_size[1] - self._patch_size[2] + 1
#             n_w = self._image_size[2] - self._patch_size[3] + 1
#
#         for p, (z, y, x) in zip(patches, product(range(0, n_d, self._step[1]),
#                                                  range(0, n_h, self._step[2]),
#                                                  range(0, n_w, self._step[3]))):
#             if isinstance(p, tuple):
#                 p = self._test_image[p]
#                 p = np.expand_dims(p, 0)
#
#             elif not isinstance(p, np.ndarray):
#                 p = self._transform(p)
#                 p = np.expand_dims(p, 0)
#
#             if self._models is not None:
#                 p = torch.Tensor().new_tensor(p, device="cuda:0")
#
#                 if self._do_normalize:
#                     if len(p.size()) < 5:
#                         p = torch.unsqueeze(p, 0)
#                     p = torch.nn.functional.sigmoid(self._models[0].forward(p)).cpu().detach().numpy()
#                 elif self._do_segment:
#                     if len(p.size()) < 5:
#                         p = torch.unsqueeze(p, 0)
#                     p = torch.argmax(torch.nn.functional.softmax(self._models[0].forward(p), dim=1), dim=1,
#                                      keepdim=True).float().cpu().detach().numpy()
#                 elif self._do_normalize_and_segment:
#                     if len(p.size()) < 5:
#                         p = torch.unsqueeze(p, 0)
#                     p = torch.nn.functional.sigmoid(self._models[0].forward(p))
#                     p = torch.argmax(torch.nn.functional.softmax(self._models[1].forward(p), dim=1), dim=1,
#                                      keepdim=True).float().cpu().detach().numpy()
#
#             if self._is_multimodal:
#                 img[0:2, z:z + self._patch_size[1], y:y + self._patch_size[2], x:x + self._patch_size[3]] += p[0]
#                 divisor[0:2, z:z + self._patch_size[1], y:y + self._patch_size[2], x:x + self._patch_size[3]] += 1
#             else:
#                 img[z:z + self._patch_size[1], y:y + self._patch_size[2], x:x + self._patch_size[3]] += p[0][0]
#                 divisor[z:z + self._patch_size[1], y:y + self._patch_size[2], x:x + self._patch_size[3]] += 1
#
#         if self._do_segment or self._do_normalize_and_segment:
#             return np.clip(np.round(img / divisor), a_min=0, a_max=3)
#         else:
#             return img / divisor

class ImageReconstructor(object):

    def __init__(self, images: List[np.ndarray], patch_size: Tuple[int, int, int, int],
                 reconstructed_image_size: Tuple[int, int, int, int], step: Tuple[int, int, int, int],
                 batch_size: int = 5, models: List[torch.nn.Module] = None, normalize: bool = False,
                 is_ground_truth: bool = False, normalize_and_segment: bool = False, segment: bool = False,
                 is_multimodal=False, prob_bias=0.0,
                 prob_noise=0.0, alpha=0.0, snr=0.0):
        self._patch_size = patch_size
        self._reconstructed_image_size = reconstructed_image_size
        self._step = step
        self._models = models
        self._do_normalize = normalize
        self._do_segment = segment
        self._is_ground_truth = is_ground_truth
        self._do_normalize_and_segment = normalize_and_segment
        self._transform = Compose([ToNumpyArray()])
        self._is_multimodal = is_multimodal
        self._batch_size = batch_size

        transformed_images = []
        for image in images:
            transform = PadToShape(target_shape=self._reconstructed_image_size)
            transformed_images.append(transform(image))

        self._images = transformed_images

        self._overlap_maps = list(
            map(lambda image: SliceBuilder(image, self._patch_size, self._step).build_overlap_map(),
                self._images))

        self._datasets = list(map(lambda image: SingleNDArrayDataset(image,
                                                                     patch_size=self._patch_size,
                                                                     step=self._step,
                                                                     prob_bias=prob_bias,
                                                                     prob_noise=prob_noise,
                                                                     alpha=alpha,
                                                                     snr=snr),
                                  self._images))

    @staticmethod
    def custom_collate(batch):
        patches = [item[0].unsqueeze(0) for item in batch]
        slices = [(item[1][0], item[1][1], item[1][2], item[1][3]) for item in batch]
        return [torch.cat(patches, dim=0), slices]

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def reconstruct_from_patches_3d(self):
        data_loaders = list(map(
            lambda dataset: DataLoader(dataset, batch_size=self._batch_size, num_workers=0, drop_last=False,
                                       shuffle=False, pin_memory=False, collate_fn=self.custom_collate),
            self._datasets))

        if len(self._datasets) == 2:

            reconstructed_image = [np.zeros(self._datasets[0].image_shape), np.zeros(self._datasets[1].image_shape)]

            for idx, (iseg_inputs, mrbrains_inputs) in enumerate(zip(data_loaders[ISEG_ID], data_loaders[MRBRAINS_ID])):
                inputs = torch.cat((iseg_inputs[PATCH], mrbrains_inputs[PATCH]))
                slices = [iseg_inputs[SLICE], mrbrains_inputs[SLICE]]

                if self._do_normalize:
                    patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs.cuda())))

                elif self._do_normalize_and_segment:
                    normalized_patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs.cuda())))
                    patches = torch.argmax(
                        torch.nn.functional.softmax(self._models[SEGMENTER](normalized_patches), dim=1), dim=1,
                        keepdim=True)
                elif self._do_segment:
                    patches = torch.argmax(
                        torch.nn.functional.softmax(self._models[0](inputs.cuda()), dim=1), dim=1, keepdim=True)
                else:
                    patches = inputs

                for pred_patch, slice in zip(patches[0:self._batch_size], slices[ISEG_ID]):
                    reconstructed_image[ISEG_ID][slice] = reconstructed_image[ISEG_ID][slice] + \
                                                          pred_patch.data.cpu().numpy()

                for pred_patch, slice in zip(patches[self._batch_size:self._batch_size * 2], slices[MRBRAINS_ID]):
                    reconstructed_image[MRBRAINS_ID][slice] = reconstructed_image[MRBRAINS_ID][slice] + \
                                                              pred_patch.data.cpu().numpy()

            if self._do_normalize_and_segment or self._is_ground_truth:
                reconstructed_image[ISEG_ID] = np.clip(
                    np.round(reconstructed_image[ISEG_ID] * self._overlap_maps[ISEG_ID]), a_min=0, a_max=3)
                reconstructed_image[MRBRAINS_ID] = np.clip(
                    np.round(reconstructed_image[MRBRAINS_ID] * self._overlap_maps[MRBRAINS_ID]), a_min=0, a_max=3)
            else:
                reconstructed_image[ISEG_ID] = reconstructed_image[ISEG_ID] * self._overlap_maps[ISEG_ID]
                reconstructed_image[MRBRAINS_ID] = reconstructed_image[MRBRAINS_ID] * self._overlap_maps[MRBRAINS_ID]

            return {"iSEG": reconstructed_image[ISEG_ID],
                    "MRBrainS": reconstructed_image[MRBRAINS_ID]}

        if len(self._datasets) == 3:
            reconstructed_image = [np.zeros(self._datasets[0].image_shape), np.zeros(self._datasets[1].image_shape),
                                   np.zeros(self._datasets[2].image_shape)]
            for idx, (iseg_inputs, mrbrains_inputs, abide_inputs) in enumerate(
                    zip(data_loaders[ISEG_ID], data_loaders[MRBRAINS_ID], data_loaders[ABIDE_ID])):
                inputs = torch.cat((iseg_inputs[PATCH], mrbrains_inputs[PATCH], abide_inputs[PATCH]))
                slices = [iseg_inputs[SLICE], mrbrains_inputs[SLICE], abide_inputs[SLICE]]

                if self._do_normalize:
                    patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs.cuda())))

                elif self._do_normalize_and_segment:
                    normalized_patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs.cuda())))
                    patches = torch.argmax(
                        torch.nn.functional.softmax(self._models[SEGMENTER](normalized_patches), dim=1), dim=1,
                        keepdim=True)
                else:
                    patches = inputs

                for pred_patch, slice in zip(patches[0:self._batch_size], slices[ISEG_ID]):
                    reconstructed_image[ISEG_ID][slice] = reconstructed_image[ISEG_ID][slice] + \
                                                          pred_patch.data.cpu().numpy()

                for pred_patch, slice in zip(patches[self._batch_size:self._batch_size * 2], slices[MRBRAINS_ID]):
                    reconstructed_image[MRBRAINS_ID][slice] = reconstructed_image[MRBRAINS_ID][slice] + \
                                                              pred_patch.data.cpu().numpy()

                for pred_patch, slice in zip(patches[self._batch_size * 2:self._batch_size * 3], slices[ABIDE_ID]):
                    reconstructed_image[ABIDE_ID][slice] = reconstructed_image[ABIDE_ID][slice] + \
                                                           pred_patch.data.cpu().numpy()

            reconstructed_image[ISEG_ID] = reconstructed_image[ISEG_ID] * self._overlap_maps[ISEG_ID]
            reconstructed_image[MRBRAINS_ID] = reconstructed_image[MRBRAINS_ID] * self._overlap_maps[MRBRAINS_ID]
            reconstructed_image[ABIDE_ID] = reconstructed_image[ABIDE_ID] * self._overlap_maps[ABIDE_ID]

            if self._do_normalize_and_segment:
                reconstructed_image[ISEG_ID] = np.clip(np.round(reconstructed_image[ISEG_ID]), a_min=0, a_max=3)
                reconstructed_image[MRBRAINS_ID] = np.clip(np.round(reconstructed_image[MRBRAINS_ID]), a_min=0, a_max=3)
                reconstructed_image[ABIDE_ID] = np.clip(np.round(reconstructed_image[ABIDE_ID]), a_min=0, a_max=3)

            return {"iSEG": reconstructed_image[ISEG_ID],
                    "MRBrainS": reconstructed_image[MRBRAINS_ID],
                    "ABIDE": reconstructed_image[ABIDE_ID]}
