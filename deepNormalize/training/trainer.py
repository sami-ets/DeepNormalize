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

from typing import List

import numpy as np
import torch
from ignite.metrics.confusion_matrix import ConfusionMatrix
from kerosene.config.trainers import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.devices import on_single_device
from kerosene.utils.tensors import flatten, to_onehot
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader

from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import GENERATOR, SEGMENTER, DISCRIMINATOR, IMAGE_TARGET, DATASET_ID, EPSILON
from deepNormalize.utils.image_slicer import AdaptedImageSlicer, SegmentationSlicer
from deepNormalize.utils.utils import to_html


class DeepNormalizeTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, run_config: RunConfiguration):
        super(DeepNormalizeTrainer, self).__init__("DeepNormalizeTrainer", train_data_loader, valid_data_loader,
                                                   model_trainers, run_config)

        self._training_config = training_config
        self._patience_discriminator = training_config.patience_discriminator
        self._patience_segmentation = training_config.patience_segmentation
        self._slicer = AdaptedImageSlicer()
        self._seg_slicer = SegmentationSlicer()
        self._generator = self._model_trainers[GENERATOR]
        self._segmenter = self._model_trainers[SEGMENTER]
        self._class_hausdorff_distance_gauge = AverageGauge()
        self._mean_hausdorff_distance_gauge = AverageGauge()
        self._class_dice_gauge = AverageGauge()
        self._confusion_matrix_gauge = ConfusionMatrix(num_classes=4)

    def train_step(self, inputs, target):
        seg_pred = torch.Tensor().new_zeros(
            size=(self._training_config.batch_size, 1, 32, 32, 32), dtype=torch.float, device="cpu")

        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            self._generator.zero_grad()

            if self.current_train_step % self._training_config.variables["train_generator_every_n_steps"] == 0:
                gen_loss = self._generator.compute_loss(gen_pred, inputs)
                self._generator.update_train_loss(gen_loss.loss)
                gen_loss.backward()

                if not on_single_device(self._run_config.devices):
                    self.average_gradients(self._generator)

                self._generator.step()

        if self._should_activate_segmentation():
            self._generator.zero_grad()
            self._segmenter.zero_grad()

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_train_loss(seg_loss.mean())

            metric = self._segmenter.compute_metric(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            self._segmenter.update_train_metric(metric.mean())

            if self.current_train_step % self._training_config.variables["train_generator_every_n_steps"] == 0:
                seg_loss.mean().backward(retain_graph=True)
            else:
                seg_loss.mean().backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._segmenter)

            self._segmenter.step()
            self._generator.step()

        if self.current_train_step % 100 == 0:
            self._update_plots(inputs.cpu().detach(), gen_pred.cpu().detach(), seg_pred.cpu().detach(),
                               target[IMAGE_TARGET].cpu().detach())

        self.custom_variables["Generated Intensity Histogram"] = flatten(gen_pred.cpu().detach())
        self.custom_variables["Input Intensity Histogram"] = flatten(inputs.cpu().detach())

    def validate_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_valid_loss(gen_loss)

        if self._should_activate_segmentation():
            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_valid_loss(gen_loss)

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_valid_loss(seg_loss.mean())
            metric = self._segmenter.compute_metric(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            self._segmenter.update_valid_metric(metric.mean())

            seg_pred_ = to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4)
            target_ = to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(), num_classes=4)

            distances = np.zeros((4,))
            for channel in range(seg_pred_.size(1)):
                distances[channel] = max(
                    directed_hausdorff(flatten(seg_pred_[:, channel, ...]).cpu().detach().numpy(),
                                       flatten(target_[:, channel, ...]).cpu().detach().numpy())[0],
                    directed_hausdorff(flatten(target_[:, channel, ...]).cpu().detach().numpy(),
                                       flatten(seg_pred_[:, channel, ...]).cpu().detach().numpy())[0])

            self._class_hausdorff_distance_gauge.update(distances)
            self._class_dice_gauge.update(np.array(metric.numpy()))

            self._confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                          num_classes=4),
                torch.squeeze(target[IMAGE_TARGET].long(), dim=1)))

    def _update_plots(self, inputs, generator_predictions, segmenter_predictions, target):
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=5, mode="trilinear",
                                                 align_corners=True).numpy()
        generator_predictions = torch.nn.functional.interpolate(generator_predictions, scale_factor=5, mode="trilinear",
                                                                align_corners=True).numpy()
        segmenter_predictions = torch.nn.functional.interpolate(
            torch.argmax(torch.nn.functional.softmax(segmenter_predictions, dim=1), dim=1, keepdim=True).float(),
            scale_factor=5, mode="nearest").numpy()

        target = torch.nn.functional.interpolate(target.float(), scale_factor=5, mode="nearest").numpy()

        inputs = self._normalize(inputs)
        generator_predictions = self._normalize(generator_predictions)
        segmenter_predictions = self._normalize(segmenter_predictions)
        target = self._normalize(target)

        self.custom_variables["Input Batch"] = self._slicer.get_slice(SliceType.AXIAL, inputs)
        self.custom_variables["Generated Batch"] = self._slicer.get_slice(SliceType.AXIAL, generator_predictions)
        self._custom_variables["Segmented Batch"] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                                       segmenter_predictions)
        self._custom_variables["Segmentation Ground Truth Batch"] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                                                       target)

    def scheduler_step(self):
        self._generator.scheduler_step()

        if self._should_activate_segmentation():
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

    def _should_activate_autoencoder(self):
        return self._current_epoch < self._patience_discriminator

    def _should_activate_discriminator_loss(self):
        return self._patience_discriminator <= self._current_epoch < self._patience_segmentation

    def _should_activate_segmentation(self):
        return self._current_epoch >= self._patience_segmentation

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        self.custom_variables["GPU {} Memory".format(self._run_config.local_rank)] = np.array(
            [torch.cuda.memory_allocated() / (1024.0 * 1024.0)])

        if self._should_activate_autoencoder():
            self.custom_variables["Mean Hausdorff Distance"] = np.array([0])
            self.custom_variables["Metric Table"] = to_html(["CSF", "Grey Matter", "White Matter"], ["DSC", "HD"],
                                                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            self.custom_variables["Confusion Matrix"] = np.zeros((4, 4))

        if self._should_activate_segmentation():
            self.custom_variables["Mean Hausdorff Distance"] = np.array([self._class_hausdorff_distance_gauge.compute()[
                                                                         -3:].mean()])
            self.custom_variables["Metric Table"] = to_html(["CSF", "Grey Matter", "White Matter"], ["DSC", "HD"],
                                                            [self._class_dice_gauge.compute(),
                                                             self._class_hausdorff_distance_gauge.compute()[-3:]])
            self.custom_variables["Confusion Matrix"] = np.array(
                self._confusion_matrix_gauge.compute().cpu().detach().numpy())

        self._class_hausdorff_distance_gauge.reset()
        self._class_dice_gauge.reset()
        self._confusion_matrix_gauge.reset()
