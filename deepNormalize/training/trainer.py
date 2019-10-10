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

import time
from datetime import timedelta
from typing import List

import numpy as np
import torch
from ignite.metrics.confusion_matrix import ConfusionMatrix
from kerosene.config.trainers import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from kerosene.nn.functional import js_div
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.devices import on_single_device
from kerosene.utils.tensors import flatten, to_onehot
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader

from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import IMAGE_TARGET, DATASET_ID, EPSILON
from deepNormalize.utils.constants import ISEG, MRBrainS
from deepNormalize.utils.image_slicer import AdaptedImageSlicer, SegmentationSlicer
from deepNormalize.utils.utils import to_html, to_html_per_dataset, to_html_JS, to_html_time


class DeepNormalizeTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, test_data_loader: DataLoader,
                 run_config: RunConfiguration):
        super(DeepNormalizeTrainer, self).__init__("DeepNormalizeTrainer", train_data_loader, valid_data_loader,
                                                   test_data_loader, model_trainers, run_config)

        self._training_config = training_config
        self._slicer = AdaptedImageSlicer()
        self._seg_slicer = SegmentationSlicer()
        self._segmenter = self._model_trainers[0]
        self._class_hausdorff_distance_gauge = AverageGauge()
        self._mean_hausdorff_distance_gauge = AverageGauge()
        self._per_dataset_hausdorff_distance_gauge = AverageGauge()
        self._iSEG_dice_gauge = AverageGauge()
        self._MRBrainS_dice_gauge = AverageGauge()
        self._iSEG_hausdorff_gauge = AverageGauge()
        self._MRBrainS_hausdorff_gauge = AverageGauge()
        self._class_dice_gauge = AverageGauge()
        self._js_div_inputs_gauge = AverageGauge()
        self._general_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._iSEG_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._MRBrainS_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._start_time = 0
        self._stop_time = 0

    def train_step(self, inputs, target):
        self._segmenter.zero_grad()

        seg_pred = self._segmenter.forward(inputs)
        seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                          num_classes=4))
        self._segmenter.update_train_loss(seg_loss.mean().loss)

        metric = self._segmenter.compute_metric(torch.nn.functional.softmax(seg_pred, dim=1),
                                                torch.squeeze(target[IMAGE_TARGET], dim=1).long())
        self._segmenter.update_train_metric(metric.mean())

        seg_loss.mean().backward()

        if not on_single_device(self._run_config.devices):
            self.average_gradients(self._segmenter)

        self._segmenter.step()

        real_count = self.count(torch.cat((target[DATASET_ID].cpu().detach(), torch.Tensor().new_full(
            size=(inputs.size(0) // 2,),
            fill_value=2,
            dtype=torch.long,
            device="cpu",
            requires_grad=False)), dim=0), 3)
        self.custom_variables["Pie Plot True"] = real_count

        if self.current_train_step % 100 == 0:
            self._update_plots(inputs.cpu().detach(), seg_pred.cpu().detach(), target[IMAGE_TARGET].cpu().detach())

        self.custom_variables["Input Intensity Histogram"] = flatten(inputs.cpu().detach())
        self.custom_variables["Background Input Intensity Histogram"] = inputs[
            torch.where(target[IMAGE_TARGET] == 0)].cpu().detach()
        self.custom_variables["CSF Input Intensity Histogram"] = inputs[
            torch.where(target[IMAGE_TARGET] == 1)].cpu().detach()
        self.custom_variables["GM Input Intensity Histogram"] = inputs[
            torch.where(target[IMAGE_TARGET] == 2)].cpu().detach()
        self.custom_variables["WM Input Intensity Histogram"] = inputs[
            torch.where(target[IMAGE_TARGET] == 3)].cpu().detach()

    def validate_step(self, inputs, target):
        seg_pred = self._segmenter.forward(inputs)
        seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                          num_classes=4))
        self._segmenter.update_valid_loss(seg_loss.mean())
        metric = self._segmenter.compute_metric(torch.nn.functional.softmax(seg_pred, dim=1),
                                                torch.squeeze(target[IMAGE_TARGET], dim=1).long())
        self._segmenter.update_valid_metric(metric.mean())

        self._iSEG_dice_gauge.update(np.array(self._segmenter.compute_metric(
            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG)], dim=1),
            torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG)],
                          dim=1).long()).numpy()))

        self._iSEG_hausdorff_gauge.update(self.compute_mean_hausdorff_distance(
            to_onehot(
                torch.argmax(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG)], dim=1),
                    dim=1), num_classes=4),
            to_onehot(
                torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG)], dim=1).long(),
                num_classes=4))[-3:])

        self._iSEG_confusion_matrix_gauge.update((
            to_onehot(
                torch.argmax(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG)], dim=1),
                    dim=1, keepdim=False),
                num_classes=4),
            torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG)].long(), dim=1)))

    def test_step(self, inputs, target):
        seg_pred = self._segmenter.forward(inputs)
        seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                          num_classes=4))
        self._segmenter.update_test_loss(seg_loss.mean())
        metric = self._segmenter.compute_metric(torch.nn.functional.softmax(seg_pred, dim=1),
                                                torch.squeeze(target[IMAGE_TARGET], dim=1).long())
        self._segmenter.update_test_metric(metric.mean())

        self._class_dice_gauge.update(np.array(metric.numpy()))

        if seg_pred[torch.where(target[DATASET_ID] == ISEG)].shape[0] != 0:
            self._iSEG_dice_gauge.update(np.array(self._segmenter.compute_metric(
                torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG)], dim=1),
                torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG)],
                              dim=1).long()).numpy()))

            self._iSEG_hausdorff_gauge.update(self.compute_mean_hausdorff_distance(
                to_onehot(
                    torch.argmax(
                        torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG)], dim=1),
                        dim=1), num_classes=4),
                to_onehot(
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG)], dim=1).long(),
                    num_classes=4))[-3:])

            self._iSEG_confusion_matrix_gauge.update((
                to_onehot(
                    torch.argmax(
                        torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG)], dim=1),
                        dim=1, keepdim=False),
                    num_classes=4),
                torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG)].long(), dim=1)))

        else:
            self._iSEG_dice_gauge.update(np.zeros((3,)))
            self._iSEG_hausdorff_gauge.update(np.zeros((3,)))

        if seg_pred[torch.where(target[DATASET_ID] == MRBrainS)].shape[0] != 0:
            self._MRBrainS_dice_gauge.update(np.array(self._segmenter.compute_metric(
                torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBrainS)], dim=1),
                torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBrainS)],
                              dim=1).long()).numpy()))

            self._MRBrainS_hausdorff_gauge.update(self.compute_mean_hausdorff_distance(
                to_onehot(
                    torch.argmax(
                        torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBrainS)], dim=1),
                        dim=1), num_classes=4),
                to_onehot(
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBrainS)], dim=1).long(),
                    num_classes=4))[-3:])

            self._MRBrainS_confusion_matrix_gauge.update((
                to_onehot(
                    torch.argmax(
                        torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBrainS)], dim=1),
                        dim=1, keepdim=False),
                    num_classes=4),
                torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBrainS)].long(), dim=1)))
        else:
            self._MRBrainS_dice_gauge.update(np.zeros((3,)))
            self._MRBrainS_hausdorff_gauge.update(np.zeros((3,)))

        self._class_hausdorff_distance_gauge.update(
            self.compute_mean_hausdorff_distance(
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4),
                to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(), num_classes=4))[-3:])

        self._general_confusion_matrix_gauge.update((
            to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                      num_classes=4),
            torch.squeeze(target[IMAGE_TARGET].long(), dim=1)))

        inputs_reshaped = inputs.reshape(inputs.shape[0],
                                         inputs.shape[1] * inputs.shape[2] * inputs.shape[3] * inputs.shape[4])

        inputs_ = torch.Tensor().new_zeros((inputs_reshaped.shape[0], 256))
        for image in range(inputs_reshaped.shape[0]):
            inputs_[image] = torch.nn.functional.softmax(torch.histc(inputs_reshaped[image], bins=256), dim=0)

        self._js_div_inputs_gauge.update(js_div(inputs_))

    def _update_plots(self, inputs, segmenter_predictions, target):
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=5, mode="trilinear",
                                                 align_corners=True).numpy()
        segmenter_predictions = torch.nn.functional.interpolate(
            torch.argmax(torch.nn.functional.softmax(segmenter_predictions, dim=1), dim=1, keepdim=True).float(),
            scale_factor=5, mode="nearest").numpy()

        target = torch.nn.functional.interpolate(target.float(), scale_factor=5, mode="nearest").numpy()

        inputs = self._normalize(inputs)
        segmenter_predictions = self._normalize(segmenter_predictions)
        target = self._normalize(target)

        self.custom_variables["Input Batch"] = self._slicer.get_slice(SliceType.AXIAL, inputs)
        self._custom_variables["Segmented Batch"] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                                       segmenter_predictions)
        self._custom_variables["Segmentation Ground Truth Batch"] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                                                       target)

    def scheduler_step(self):
        self._segmenter.scheduler_step()

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    @staticmethod
    def average_gradients(model):
        size = float(torch.distributed.get_world_size())
        for param in model.parameters():
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= size

    @staticmethod
    def merge_tensors(tensor_0, tensor_1):
        return torch.cat((tensor_0, tensor_1), dim=0)

    def on_training_begin(self):
        self._start_time = time.time()

    def on_training_end(self):
        self._stop_time = time.time()

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        self.custom_variables["GPU {} Memory".format(self._run_config.local_rank)] = np.array(
            [torch.cuda.memory_allocated() / (1024.0 * 1024.0)])

        self.custom_variables["Runtime"] = to_html_time(timedelta(seconds=time.time() - self._start_time))

        self.custom_variables["Mean Hausdorff Distance"] = np.array(
            [self._class_hausdorff_distance_gauge.compute().mean()])
        self.custom_variables["Metric Table"] = to_html(["CSF", "Grey Matter", "White Matter"], ["DSC", "HD"],
                                                        [self._class_dice_gauge.compute(),
                                                         self._class_hausdorff_distance_gauge.compute()])
        self.custom_variables["Per-Dataset Metric Table"] = to_html_per_dataset(
            ["CSF", "Grey Matter", "White Matter"],
            ["DSC", "HD"],
            [[self._iSEG_dice_gauge.compute(),
              self._iSEG_hausdorff_gauge.compute()],
             [self._MRBrainS_dice_gauge.compute(),
              self._MRBrainS_hausdorff_gauge.compute()]],
            ["iSEG", "MRBrainS"])
        self.custom_variables["Jensen-Shannon Table"] = to_html_JS(["Input data", "Generated Data"],
                                                                   ["JS Divergence"],
                                                                   [self._js_div_inputs_gauge.compute().numpy(),
                                                                    np.array([0])])
        self.custom_variables["Confusion Matrix"] = np.array(
            np.rot90(self._general_confusion_matrix_gauge.compute().cpu().detach().numpy()))

        if self._iSEG_confusion_matrix_gauge._num_examples != 0:
            self.custom_variables["iSEG Confusion Matrix"] = np.array(
                np.rot90(self._iSEG_confusion_matrix_gauge.compute().cpu().detach().numpy()))
        else:
            self.custom_variables["iSEG Confusion Matrix"] = np.zeros((4, 4))

        if self._MRBrainS_confusion_matrix_gauge._num_examples != 0:
            self.custom_variables["MRBrainS Confusion Matrix"] = np.array(
                np.rot90(self._MRBrainS_confusion_matrix_gauge.compute().cpu().detach().numpy()))
        else:
            self.custom_variables["MRBrainS Confusion Matrix"] = np.zeros((4, 4))

        self.custom_variables["Jensen-Shannon Divergence Inputs"] = np.array(
            [self._js_div_inputs_gauge.compute().numpy()])

        self._MRBrainS_confusion_matrix_gauge.reset()
        self._iSEG_confusion_matrix_gauge.reset()
        self._general_confusion_matrix_gauge.reset()
        self._class_hausdorff_distance_gauge.reset()
        self._class_dice_gauge.reset()
        self._js_div_inputs_gauge.reset()

    @staticmethod
    def count(tensor, n_classes):
        count = torch.Tensor().new_zeros(size=(n_classes,), device="cpu")
        for i in range(n_classes):
            count[i] = torch.sum(tensor == i).int()
        return count

    def compute_mean_hausdorff_distance(self, seg_pred, target):
        distances = np.zeros((4,))
        for channel in range(seg_pred.size(1)):
            distances[channel] = (
                                         directed_hausdorff(
                                             flatten(seg_pred[:, channel, ...]).cpu().detach().numpy(),
                                             flatten(target[:, channel, ...]).cpu().detach().numpy())[0] +
                                         directed_hausdorff(
                                             flatten(target[:, channel, ...]).cpu().detach().numpy(),
                                             flatten(seg_pred[:, channel, ...]).cpu().detach().numpy())[0]) / 2.0
        return distances

    def finalize(self):
        pass

    def on_batch_end(self):
        pass

    def on_test_batch_begin(self):
        pass

    def on_test_batch_end(self):
        pass

    def on_test_epoch_begin(self):
        pass

    def on_test_epoch_end(self):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_train_epoch_begin(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_batch_begin(self):
        pass

    def on_valid_batch_end(self):
        pass

    def on_validation_epoch_begin(self):
        pass

    def on_validation_epoch_end(self):
        pass
