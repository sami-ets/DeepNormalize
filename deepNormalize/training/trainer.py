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
import uuid
from datetime import timedelta
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch
from fastai.utils.mem import gpu_mem_get
from ignite.metrics.confusion_matrix import ConfusionMatrix
from kerosene.configs.configs import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from kerosene.nn.functional import js_div
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.tensors import flatten, to_onehot
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader, Dataset

from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import GENERATOR, SEGMENTER, DISCRIMINATOR, IMAGE_TARGET, DATASET_ID, ABIDE_ID, \
    NON_AUGMENTED_INPUTS, AUGMENTED_INPUTS
from deepNormalize.utils.constants import ISEG_ID, MRBRAINS_ID
from deepNormalize.utils.image_slicer import ImageSlicer, SegmentationSlicer, LabelMapper, FeatureMapSlicer
from deepNormalize.utils.utils import to_html, to_html_per_dataset, to_html_JS, to_html_time, natural_sort

pynvml.nvmlInit()


class DeepNormalizeTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, test_data_loader: DataLoader,
                 reconstruction_datasets: List[Dataset], augmented_reconstruction_datasets: List[Dataset],
                 normalize_reconstructors: list, input_reconstructors: list, segmentation_reconstructors: list,
                 augmented_reconstructors: list, run_config: RunConfiguration, dataset_config: dict):
        super(DeepNormalizeTrainer, self).__init__("DeepNormalizeTrainer", train_data_loader, valid_data_loader,
                                                   test_data_loader, model_trainers, run_config)

        self._training_config = training_config
        self._run_config = run_config
        self._dataset_configs = dataset_config
        self._patience_segmentation = training_config.patience_segmentation
        self._slicer = ImageSlicer()
        self._seg_slicer = SegmentationSlicer()
        self._fm_slicer = FeatureMapSlicer()
        self._label_mapper = LabelMapper()
        self._reconstruction_datasets = reconstruction_datasets
        self._augmented_reconstruction_datasets = augmented_reconstruction_datasets
        self._normalize_reconstructors = normalize_reconstructors
        self._input_reconstructors = input_reconstructors
        self._segmentation_reconstructors = segmentation_reconstructors
        self._augmented_reconstructors = augmented_reconstructors
        self._num_datasets = len(input_reconstructors)
        self._generator = self._model_trainers[GENERATOR]
        self._discriminator = self._model_trainers[DISCRIMINATOR]
        self._segmenter = self._model_trainers[SEGMENTER]
        self._D_G_X_as_X_training_gauge = AverageGauge()
        self._D_G_X_as_X_validation_gauge = AverageGauge()
        self._D_G_X_as_X_test_gauge = AverageGauge()
        self._total_loss_training_gauge = AverageGauge()
        self._total_loss_validation_gauge = AverageGauge()
        self._total_loss_test_gauge = AverageGauge()
        self._class_hausdorff_distance_gauge = AverageGauge()
        self._mean_hausdorff_distance_gauge = AverageGauge()
        self._per_dataset_hausdorff_distance_gauge = AverageGauge()
        self._iSEG_dice_gauge = AverageGauge()
        self._MRBrainS_dice_gauge = AverageGauge()
        self._ABIDE_dice_gauge = AverageGauge()
        self._iSEG_hausdorff_gauge = AverageGauge()
        self._MRBrainS_hausdorff_gauge = AverageGauge()
        self._ABIDE_hausdorff_gauge = AverageGauge()
        self._valid_dice_gauge = AverageGauge()
        self._class_dice_gauge = AverageGauge()
        self._class_dice_gauge_on_reconstructed_images = AverageGauge()
        self._class_dice_gauge_on_reconstructed_iseg_images = AverageGauge()
        self._class_dice_gauge_on_reconstructed_mrbrains_images = AverageGauge()
        self._class_dice_gauge_on_reconstructed_abide_images = AverageGauge()
        self._js_div_inputs_gauge = AverageGauge()
        self._js_div_gen_gauge = AverageGauge()
        self._general_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._iSEG_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._MRBrainS_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._ABIDE_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._discriminator_confusion_matrix_gauge = ConfusionMatrix(num_classes=self._num_datasets + 1)
        self._discriminator_confusion_matrix_gauge_training = ConfusionMatrix(num_classes=self._num_datasets + 1)
        self._previous_mean_dice = 0.0
        self._previous_per_dataset_table = ""
        self._start_time = time.time()
        print("Total number of parameters: {}".format(sum(p.numel() for p in self._segmenter.parameters()) +
                                                      sum(p.numel() for p in self._generator.parameters()) +
                                                      sum(p.numel() for p in self._discriminator.parameters())))
        pynvml.nvmlInit()

    def train_step(self, inputs, target):
        disc_pred = None
        seg_pred = torch.Tensor().new_zeros(
            size=(self._training_config.batch_size, 1, 32, 32, 32), dtype=torch.float, device="cpu")

        gen_pred = torch.nn.functional.relu(self._generator.forward(inputs[AUGMENTED_INPUTS]))

        if self._should_activate_autoencoder():
            self._generator.zero_grad()
            self._discriminator.zero_grad()
            self._segmenter.zero_grad()

            if self.current_train_step % self._training_config.variables["train_generator_every_n_steps"] == 0:
                gen_loss = self._generator.compute_loss("MSELoss", gen_pred, inputs[AUGMENTED_INPUTS])
                self._generator.update_train_loss("MSELoss", gen_loss)
                gen_loss.backward()

                self._generator.step()

            disc_loss, disc_pred, disc_target, x_conv1, x_layer1, x_layer2, x_layer3 = self._train_discriminator(
                inputs[NON_AUGMENTED_INPUTS],
                gen_pred.detach(),
                target[DATASET_ID])
            disc_loss.backward()
            self._discriminator.step()

            # Pretrain segmenter.
            seg_pred = self._segmenter.forward(inputs[AUGMENTED_INPUTS])
            seg_loss = self._segmenter.compute_loss("DiceLoss", torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_train_loss("DiceLoss", seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            metric["IoU"] = metric["IoU"].mean()
            self._segmenter.update_train_metrics(metric)

            seg_loss.mean().backward()
            self._segmenter.step()

        if self._should_activate_segmentation():
            self._generator.zero_grad()
            self._discriminator.zero_grad()
            self._segmenter.zero_grad()

            gen_loss = self._generator.compute_loss("MSELoss", gen_pred, inputs[AUGMENTED_INPUTS])
            self._generator.update_train_loss("MSELoss", gen_loss)

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss("DiceLoss", torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_train_loss("DiceLoss", seg_loss.mean())

            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            metric["IoU"] = metric["IoU"].mean()
            self._segmenter.update_train_metrics(metric)

            if self.current_train_step % self._training_config.variables["train_generator_every_n_steps_seg"] == 0:
                disc_loss_as_X = self._evaluate_loss_D_G_X_as_X(gen_pred,
                                                                torch.Tensor().new_full(
                                                                    fill_value=self._num_datasets,
                                                                    size=(gen_pred.size(0),),
                                                                    dtype=torch.long,
                                                                    device=inputs[NON_AUGMENTED_INPUTS].device,
                                                                    requires_grad=False))

                total_loss = self._training_config.variables["seg_ratio"] * seg_loss.mean() + \
                             self._training_config.variables["disc_ratio"] * disc_loss_as_X
                self._D_G_X_as_X_training_gauge.update(disc_loss_as_X.item())
                self._total_loss_training_gauge.update(total_loss.item())

                total_loss.backward()

                self._segmenter.step()
                self._generator.step()

            else:
                seg_loss.mean().backward()

                self._segmenter.step()
                self._generator.step()

            self._discriminator.zero_grad()

            disc_loss, disc_pred, disc_target, x_conv1, x_layer1, x_layer2, x_layer3 = self._train_discriminator(
                inputs[NON_AUGMENTED_INPUTS],
                gen_pred.detach(),
                target[DATASET_ID])
            disc_loss.backward()

            self._discriminator.step()

        self._discriminator_confusion_matrix_gauge_training.update((
            to_onehot(torch.argmax(torch.nn.functional.softmax(disc_pred, dim=1), dim=1),
                      num_classes=self._num_datasets + 1),
            disc_target))

        if disc_pred is not None:
            self._make_disc_pie_plots(disc_pred, inputs, target)

        if self.current_train_step % 100 == 0:
            self._update_image_plots(inputs[AUGMENTED_INPUTS].cpu().detach(), gen_pred.cpu().detach(),
                                     seg_pred.cpu().detach(),
                                     target[IMAGE_TARGET].cpu().detach(), target[DATASET_ID].cpu().detach())

        if self._run_config.local_rank == 0:
            if self.current_train_step % 100 == 0:
                self.custom_variables["Generated Intensity Histogram"] = flatten(gen_pred.cpu().detach())
                self.custom_variables["Input Intensity Histogram"] = flatten(
                    inputs[NON_AUGMENTED_INPUTS].cpu().detach())
                self.custom_variables["Per-Dataset Histograms"] = cv2.imread(
                    self._construct_class_histogram(inputs, target, gen_pred)).transpose((2, 0, 1))
                self.custom_variables["Background Generated Intensity Histogram"] = gen_pred[
                    torch.where(target[IMAGE_TARGET] == 0)].cpu().detach()
                self.custom_variables["CSF Generated Intensity Histogram"] = gen_pred[
                    torch.where(target[IMAGE_TARGET] == 1)].cpu().detach()
                self.custom_variables["GM Generated Intensity Histogram"] = gen_pred[
                    torch.where(target[IMAGE_TARGET] == 2)].cpu().detach()
                self.custom_variables["WM Generated Intensity Histogram"] = gen_pred[
                    torch.where(target[IMAGE_TARGET] == 3)].cpu().detach()
                self.custom_variables["Background Input Intensity Histogram"] = inputs[NON_AUGMENTED_INPUTS][
                    torch.where(target[IMAGE_TARGET] == 0)].cpu().detach()
                self.custom_variables["CSF Input Intensity Histogram"] = inputs[NON_AUGMENTED_INPUTS][
                    torch.where(target[IMAGE_TARGET] == 1)].cpu().detach()
                self.custom_variables["GM Input Intensity Histogram"] = inputs[NON_AUGMENTED_INPUTS][
                    torch.where(target[IMAGE_TARGET] == 2)].cpu().detach()
                self.custom_variables["WM Input Intensity Histogram"] = inputs[NON_AUGMENTED_INPUTS][
                    torch.where(target[IMAGE_TARGET] == 3)].cpu().detach()
                self.custom_variables["Conv1 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(torch.nn.functional.interpolate(x_conv1.cpu().detach(), scale_factor=5, mode="trilinear", align_corners=True).numpy()[0], 0))
                self.custom_variables["Layer1 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(torch.nn.functional.interpolate(x_layer1.cpu().detach(), scale_factor=10, mode="trilinear", align_corners=True).numpy()[0], 0))
                self.custom_variables["Layer2 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(torch.nn.functional.interpolate(x_layer2.cpu().detach(), scale_factor=20, mode="trilinear", align_corners=True).numpy()[0], 0))
                self.custom_variables["Layer3 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(torch.nn.functional.interpolate(x_layer3.cpu().detach(), scale_factor=20, mode="trilinear", align_corners=True).numpy()[0], 0))

    def validate_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs[AUGMENTED_INPUTS])

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_loss("MSELoss", gen_pred, inputs[AUGMENTED_INPUTS])
            self._generator.update_valid_loss("MSELoss", gen_loss)

            disc_loss, disc_pred, _ = self._validate_discriminator(inputs[NON_AUGMENTED_INPUTS], gen_pred,
                                                                   target[DATASET_ID])

            seg_pred = self._segmenter.forward(inputs[AUGMENTED_INPUTS])
            seg_loss = self._segmenter.compute_loss("DiceLoss", torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_valid_loss("DiceLoss", seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            metric["IoU"] = metric["IoU"].mean()
            self._segmenter.update_valid_metrics(metric)
            self._valid_dice_gauge.update(metric["Dice"])

        if self._should_activate_segmentation():
            gen_loss = self._generator.compute_loss("MSELoss", gen_pred, inputs[AUGMENTED_INPUTS])
            self._generator.update_valid_loss("MSELoss", gen_loss)

            disc_loss, disc_pred, _ = self._validate_discriminator(inputs[NON_AUGMENTED_INPUTS], gen_pred,
                                                                   target[DATASET_ID])

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss("DiceLoss", torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_valid_loss("DiceLoss", seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            metric["IoU"] = metric["IoU"].mean()
            self._segmenter.update_valid_metrics(metric)
            self._valid_dice_gauge.update(metric["Dice"])

            disc_loss_as_X = self._evaluate_loss_D_G_X_as_X(gen_pred,
                                                            torch.Tensor().new_full(
                                                                fill_value=self._num_datasets,
                                                                size=(gen_pred.size(0),),
                                                                dtype=torch.long,
                                                                device=inputs[NON_AUGMENTED_INPUTS].device,
                                                                requires_grad=False))

            total_loss = self._training_config.variables["seg_ratio"] * seg_loss.mean() + \
                         self._training_config.variables["disc_ratio"] * disc_loss_as_X
            self._D_G_X_as_X_validation_gauge.update(disc_loss_as_X.item())
            self._total_loss_validation_gauge.update(total_loss.item())

    def test_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs[AUGMENTED_INPUTS])

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_loss("MSELoss", gen_pred, inputs[AUGMENTED_INPUTS])
            self._generator.update_test_loss("MSELoss", gen_loss)

            disc_loss, disc_pred, _ = self._validate_discriminator(inputs[NON_AUGMENTED_INPUTS], gen_pred,
                                                                   target[DATASET_ID], test=True)

            seg_pred = self._segmenter.forward(inputs[AUGMENTED_INPUTS])
            seg_loss = self._segmenter.compute_loss("DiceLoss", torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_test_loss("DiceLoss", seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())

            self._class_dice_gauge.update(np.array(metric["Dice"]))
            metric["Dice"] = metric["Dice"].mean()
            metric["IoU"] = metric["IoU"].mean()
            self._segmenter.update_test_metrics(metric)

        if self._should_activate_segmentation():
            gen_loss = self._generator.compute_loss("MSELoss", gen_pred, inputs[AUGMENTED_INPUTS])
            self._generator.update_test_loss("MSELoss", gen_loss)

            disc_loss, disc_pred, disc_target = self._validate_discriminator(inputs[NON_AUGMENTED_INPUTS], gen_pred,
                                                                             target[DATASET_ID],
                                                                             test=True)

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss("DiceLoss", torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_test_loss("DiceLoss", seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())

            self._class_dice_gauge.update(np.array(metric["Dice"]))
            metric["Dice"] = metric["Dice"].mean()
            metric["IoU"] = metric["IoU"].mean()
            self._segmenter.update_test_metrics(metric)

            disc_loss_as_X = self._evaluate_loss_D_G_X_as_X(gen_pred,
                                                            torch.Tensor().new_full(
                                                                fill_value=self._num_datasets,
                                                                size=(gen_pred.size(0),),
                                                                dtype=torch.long,
                                                                device=inputs[NON_AUGMENTED_INPUTS].device,
                                                                requires_grad=False))

            total_loss = self._training_config.variables["seg_ratio"] * seg_loss.mean() + \
                         self._training_config.variables["disc_ratio"] * disc_loss_as_X

            self._D_G_X_as_X_test_gauge.update(disc_loss_as_X.item())
            self._total_loss_test_gauge.update(total_loss.item())

            if seg_pred[torch.where(target[DATASET_ID] == ISEG_ID)].shape[0] != 0:
                self._iSEG_dice_gauge.update(np.array(self._segmenter.compute_metrics(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG_ID)], dim=1),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG_ID)],
                                  dim=1).long())["Dice"].numpy()))

                self._iSEG_hausdorff_gauge.update(self._compute_mean_hausdorff_distance(
                    to_onehot(
                        torch.argmax(
                            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG_ID)], dim=1),
                            dim=1), num_classes=4),
                    to_onehot(
                        torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG_ID)], dim=1).long(),
                        num_classes=4))[-3:])

                self._iSEG_confusion_matrix_gauge.update((
                    to_onehot(
                        torch.argmax(
                            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG_ID)], dim=1),
                            dim=1, keepdim=False),
                        num_classes=4),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG_ID)].long(), dim=1)))

            else:
                self._iSEG_dice_gauge.update(np.zeros((3,)))
                self._iSEG_hausdorff_gauge.update(np.zeros((3,)))

            if seg_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)].shape[0] != 0:
                self._MRBrainS_dice_gauge.update(np.array(self._segmenter.compute_metrics(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)], dim=1),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBRAINS_ID)],
                                  dim=1).long())["Dice"].numpy()))

                self._MRBrainS_hausdorff_gauge.update(self._compute_mean_hausdorff_distance(
                    to_onehot(
                        torch.argmax(
                            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)],
                                                        dim=1),
                            dim=1), num_classes=4),
                    to_onehot(
                        torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBRAINS_ID)],
                                      dim=1).long(),
                        num_classes=4))[-3:])

                self._MRBrainS_confusion_matrix_gauge.update((
                    to_onehot(
                        torch.argmax(
                            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)],
                                                        dim=1),
                            dim=1, keepdim=False),
                        num_classes=4),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBRAINS_ID)].long(), dim=1)))

            if seg_pred[torch.where(target[DATASET_ID] == ABIDE_ID)].shape[0] != 0:
                self._ABIDE_dice_gauge.update(np.array(self._segmenter.compute_metrics(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ABIDE_ID)], dim=1),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ABIDE_ID)],
                                  dim=1).long())["Dice"].numpy()))

                self._ABIDE_hausdorff_gauge.update(self._compute_mean_hausdorff_distance(
                    to_onehot(
                        torch.argmax(
                            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ABIDE_ID)], dim=1),
                            dim=1), num_classes=4),
                    to_onehot(
                        torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ABIDE_ID)], dim=1).long(),
                        num_classes=4))[-3:])

                self._ABIDE_confusion_matrix_gauge.update((
                    to_onehot(
                        torch.argmax(
                            torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ABIDE_ID)], dim=1),
                            dim=1, keepdim=False),
                        num_classes=4),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ABIDE_ID)].long(), dim=1)))

            self._class_hausdorff_distance_gauge.update(
                self._compute_mean_hausdorff_distance(
                    to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4),
                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(), num_classes=4))[-3:])

            self._general_confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                          num_classes=4),
                torch.squeeze(target[IMAGE_TARGET].long(), dim=1)))

            self._discriminator_confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(disc_pred, dim=1), dim=1),
                          num_classes=self._num_datasets + 1),
                disc_target))

            inputs_reshaped = inputs[AUGMENTED_INPUTS].reshape(inputs[AUGMENTED_INPUTS].shape[0],
                                                               inputs[AUGMENTED_INPUTS].shape[1] *
                                                               inputs[AUGMENTED_INPUTS].shape[2] *
                                                               inputs[AUGMENTED_INPUTS].shape[3] *
                                                               inputs[AUGMENTED_INPUTS].shape[4])

            gen_pred_reshaped = gen_pred.reshape(gen_pred.shape[0],
                                                 gen_pred.shape[1] * gen_pred.shape[2] * gen_pred.shape[3] *
                                                 gen_pred.shape[4])
            inputs_ = torch.Tensor().new_zeros((inputs_reshaped.shape[0], 256))
            gen_pred_ = torch.Tensor().new_zeros((gen_pred_reshaped.shape[0], 256))
            for image in range(inputs_reshaped.shape[0]):
                inputs_[image] = torch.nn.functional.softmax(torch.histc(inputs_reshaped[image], bins=256), dim=0)
                gen_pred_[image] = torch.nn.functional.softmax(torch.histc(gen_pred_reshaped[image].float(), bins=256),
                                                               dim=0)

            self._js_div_inputs_gauge.update(js_div(inputs_).item())
            self._js_div_gen_gauge.update(js_div(gen_pred_).item())

    def scheduler_step(self):
        self._generator.scheduler_step()
        self._discriminator.scheduler_step()
        self._segmenter.scheduler_step()

    def on_epoch_begin(self):
        self._D_G_X_as_X_training_gauge.reset()
        self._D_G_X_as_X_validation_gauge.reset()
        self._D_G_X_as_X_test_gauge.reset()
        self._total_loss_training_gauge.reset()
        self._total_loss_validation_gauge.reset()
        self._total_loss_test_gauge.reset()
        self._class_hausdorff_distance_gauge.reset()
        self._mean_hausdorff_distance_gauge.reset()
        self._per_dataset_hausdorff_distance_gauge.reset()
        self._iSEG_dice_gauge.reset()
        self._MRBrainS_dice_gauge.reset()
        self._ABIDE_dice_gauge.reset()
        self._iSEG_hausdorff_gauge.reset()
        self._MRBrainS_hausdorff_gauge.reset()
        self._ABIDE_hausdorff_gauge.reset()
        self._class_dice_gauge.reset()
        self._js_div_inputs_gauge.reset()
        self._js_div_gen_gauge.reset()
        self._general_confusion_matrix_gauge.reset()
        self._iSEG_confusion_matrix_gauge.reset()
        self._MRBrainS_confusion_matrix_gauge.reset()
        self._ABIDE_confusion_matrix_gauge.reset()
        self._discriminator_confusion_matrix_gauge.reset()

        if self.epoch == self._training_config.patience_segmentation:
            self.model_trainers[GENERATOR].optimizer_lr = 0.0001

    def on_train_batch_end(self):
        self.custom_variables["GPU {} Memory".format(self._run_config.local_rank)] = [
            np.array(gpu_mem_get(self._run_config.local_rank))]

    def on_train_epoch_end(self):
        if self._run_config.local_rank == 0:
            self.custom_variables["D(G(X)) | X"] = [self._D_G_X_as_X_training_gauge.compute()]
            self.custom_variables["Total Loss"] = [self._total_loss_training_gauge.compute()]

    def on_valid_epoch_end(self):
        if self._run_config.local_rank == 0:
            self.custom_variables["D(G(X)) | X"] = [self._D_G_X_as_X_validation_gauge.compute()]
            self.custom_variables["Total Loss"] = [self._total_loss_validation_gauge.compute()]

    def on_training_end(self):
        if self._discriminator_confusion_matrix_gauge_training._num_examples != 0:
            self.custom_variables["Discriminator Confusion Matrix Training"] = np.array(
                np.fliplr(self._discriminator_confusion_matrix_gauge_training.compute().cpu().detach().numpy()))
        else:
            self.custom_variables["Discriminator Confusion Matrix Training"] = np.zeros(
                (self._num_datasets + 1, self._num_datasets + 1))

    def on_test_epoch_end(self):
        if self._run_config.local_rank == 0:
            if self.epoch % 5 == 0:
                all_patches = list(map(lambda dataset: natural_sort([sample.x for sample in dataset._samples]),
                                       self._reconstruction_datasets))
                ground_truth_patches = list(map(lambda dataset: natural_sort([sample.y for sample in dataset._samples]),
                                                self._reconstruction_datasets))
                img_input = {k: v for (k, v) in zip(self._dataset_configs.keys(), list(
                    map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
                        self._input_reconstructors)))}
                img_gt = {k: v for (k, v) in zip(self._dataset_configs.keys(), list(
                    map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches),
                        ground_truth_patches, self._input_reconstructors)))}
                img_norm = {k: v for (k, v) in zip(self._dataset_configs.keys(), list(
                    map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
                        self._normalize_reconstructors)))}
                img_seg = {k: v for (k, v) in zip(self._dataset_configs.keys(), list(
                    map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
                        self._segmentation_reconstructors)))}

                if self._training_config.data_augmentation:
                    augmented_patches = list(
                        map(lambda dataset: natural_sort([sample.x for sample in dataset._samples]),
                            self._augmented_reconstruction_datasets))
                    img_augmented = list(
                        map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches),
                            augmented_patches,
                            self._augmented_reconstructors))
                    augmented_minus_inputs = list()
                    norm_minus_augmented = list()

                    for input, augmented in zip(img_input, img_augmented):
                        augmented_minus_inputs.append(augmented - input)

                    for norm, augmented in zip(img_norm, img_augmented):
                        norm_minus_augmented.append(norm - augmented)

                for dataset in self._dataset_configs.keys():
                    self.custom_variables[
                        "Reconstructed Normalized {} Image".format(dataset)] = self._slicer.get_slice(
                        SliceType.AXIAL, np.expand_dims(np.expand_dims(img_norm[dataset], 0), 0))
                    self.custom_variables[
                        "Reconstructed Segmented {} Image".format(dataset)] = self._seg_slicer.get_colored_slice(
                        SliceType.AXIAL, np.expand_dims(np.expand_dims(img_seg[dataset], 0), 0)).squeeze(0)
                    self.custom_variables[
                        "Reconstructed Ground Truth {} Image".format(dataset)] = self._seg_slicer.get_colored_slice(
                        SliceType.AXIAL, np.expand_dims(np.expand_dims(img_gt[dataset], 0), 0)).squeeze(0)
                    self.custom_variables[
                        "Reconstructed Input {} Image".format(dataset)] = self._slicer.get_slice(
                        SliceType.AXIAL, np.expand_dims(np.expand_dims(img_input[dataset], 0), 0))
                    self.custom_variables[
                        "Reconstructed Input {} Image".format(dataset)] = self._slicer.get_slice(
                        SliceType.AXIAL, np.expand_dims(np.expand_dims(img_input[dataset], 0), 0))

                    metric = self._segmenter.compute_metrics(
                        to_onehot(torch.tensor(img_seg[dataset]).unsqueeze(0).long(), num_classes=4),
                        torch.tensor(img_gt[dataset]).unsqueeze(0).long())
                    self._class_dice_gauge_on_reconstructed_images.update(np.array(metric["Dice"]))

                    if self._training_config.data_augmentation:
                        self.custom_variables[
                            "Reconstructed Initial Noise {} Image".format(
                                dataset)] = self._seg_slicer.get_colored_slice(
                            SliceType.AXIAL,
                            np.expand_dims(np.expand_dims(augmented_minus_inputs[dataset], 0), 0)).squeeze(0)
                        self.custom_variables[
                            "Reconstructed Noise {} After Normalization".format(
                                dataset)] = self._seg_slicer.get_colored_slice(
                            SliceType.AXIAL,
                            np.expand_dims(np.expand_dims(norm_minus_augmented[dataset], 0), 0)).squeeze(0)
                    else:
                        self.custom_variables[
                            "Reconstructed Initial Noise {} Image".format(
                                dataset)] = np.zeros((224, 192))
                        self.custom_variables[
                            "Reconstructed Noise {} After Normalization".format(
                                dataset)] = np.zeros((224, 192))

                if "iSEG" in img_seg:
                    metric = self._segmenter.compute_metrics(
                        to_onehot(torch.tensor(img_seg["iSEG"]).unsqueeze(0).long(), num_classes=4),
                        torch.tensor(img_gt["iSEG"]).unsqueeze(0).long())
                    self._class_dice_gauge_on_reconstructed_iseg_images.update(np.array(metric["Dice"]))
                else:
                    self._class_dice_gauge_on_reconstructed_iseg_images.update(np.array([0.0, 0.0, 0.0]))
                if "MRBrainS" in img_seg:
                    metric = self._segmenter.compute_metrics(
                        to_onehot(torch.tensor(img_seg["MRBrainS"]).unsqueeze(0).long(), num_classes=4),
                        torch.tensor(img_gt["MRBrainS"]).unsqueeze(0).long())
                    self._class_dice_gauge_on_reconstructed_mrbrains_images.update(np.array(metric["Dice"]))
                else:
                    self._class_dice_gauge_on_reconstructed_mrbrains_images.update(np.array([0.0, 0.0, 0.0]))
                if "ABIDE" in img_seg:
                    metric = self._segmenter.compute_metrics(
                        to_onehot(torch.tensor(img_seg["ABIDE"]).unsqueeze(0).long(), num_classes=4),
                        torch.tensor(img_gt["ABIDE"]).unsqueeze(0).long())
                    self._class_dice_gauge_on_reconstructed_abide_images.update(np.array(metric["Dice"]))
                else:
                    self._class_dice_gauge_on_reconstructed_abide_images.update(np.array([0.0, 0.0, 0.0]))

                if len(img_input) == 3:
                    self.custom_variables["Reconstructed Images Histograms"] = cv2.imread(
                        self._construct_histrogram(img_norm["iSEG"],
                                                   img_input["iSEG"],
                                                   img_norm["MRBrainS"],
                                                   img_input["MRBrainS"],
                                                   img_norm["ABIDE"],
                                                   img_input["ABIDE"])).transpose((2, 0, 1))
                elif len(img_input) == 2:
                    self.custom_variables["Reconstructed Images Histograms"] = cv2.imread(
                        self._construct_double_histrogram(img_norm["iSEG"],
                                                          img_input["iSEG"],
                                                          img_norm["MRBrainS"],
                                                          img_input["MRBrainS"])).transpose((2, 0, 1))
                elif len(img_input) == 1:
                    self.custom_variables["Reconstructed Images Histograms"] = cv2.imread(
                        self._construct_single_histrogram(img_norm[list(self._dataset_configs.keys())[0]],
                                                          img_input[list(self._dataset_configs.keys())[0]],
                                                          )).transpose((2, 0, 1))

            if "ABIDE" not in self._dataset_configs.keys():
                self.custom_variables["Reconstructed Normalized ABIDE Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Segmented ABIDE Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Ground Truth ABIDE Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Input ABIDE Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Initial Noise ABIDE Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Noise ABIDE After Normalization"] = np.zeros((224, 192))
            if "iSEG" not in self._dataset_configs.keys():
                self.custom_variables["Reconstructed Normalized iSEG Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Segmented iSEG Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Ground iSEG ABIDE Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Input iSEG Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Initial Noise iSEG Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Noise iSEG After Normalization"] = np.zeros((224, 192))
            if "MRBrainS" not in self._dataset_configs.keys():
                self.custom_variables["Reconstructed Normalized MRBrainS Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Segmented MRBrainS Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Ground Truth MRBrainS Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Input MRBrainS Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Initial Noise MRBrainS Image"] = np.zeros((224, 192))
                self.custom_variables["Reconstructed Noise MRBrainS After Normalization"] = np.zeros((224, 192))

            self.custom_variables["Runtime"] = to_html_time(timedelta(seconds=time.time() - self._start_time))

            if self._general_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["Confusion Matrix"] = np.array(
                    np.fliplr(self._general_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["Confusion Matrix"] = np.zeros((4, 4))

            if self._iSEG_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["iSEG Confusion Matrix"] = np.array(
                    np.fliplr(self._iSEG_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["iSEG Confusion Matrix"] = np.zeros((4, 4))

            if self._MRBrainS_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["MRBrainS Confusion Matrix"] = np.array(
                    np.fliplr(self._MRBrainS_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["MRBrainS Confusion Matrix"] = np.zeros((4, 4))

            if self._ABIDE_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["ABIDE Confusion Matrix"] = np.array(
                    np.fliplr(self._ABIDE_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["ABIDE Confusion Matrix"] = np.zeros((4, 4))

            if self._discriminator_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["Discriminator Confusion Matrix"] = np.array(
                    np.fliplr(self._discriminator_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["Discriminator Confusion Matrix"] = np.zeros(
                    (self._num_datasets + 1, self._num_datasets + 1))

            self.custom_variables["Metric Table"] = to_html(["CSF", "Grey Matter", "White Matter"],
                                                            ["DSC", "HD"],
                                                            [
                                                                self._class_dice_gauge.compute() if self._class_dice_gauge.has_been_updated() else np.array(
                                                                    [0.0, 0.0, 0.0]),
                                                                self._class_hausdorff_distance_gauge.compute() if self._class_hausdorff_distance_gauge.has_been_updated() else np.array(
                                                                    [0.0, 0.0, 0.0])
                                                            ])

            self.custom_variables[
                "Dice score per class per epoch"] = self._class_dice_gauge.compute() if self._class_dice_gauge.has_been_updated() else np.array(
                [0.0, 0.0, 0.0])
            self.custom_variables[
                "Dice score per class per epoch on reconstructed image"] = self._class_dice_gauge_on_reconstructed_images.compute() if self._class_dice_gauge_on_reconstructed_images.has_been_updated() else np.array(
                [0.0, 0.0, 0.0])
            self.custom_variables[
                "Dice score per class per epoch on reconstructed iSEG image"] = self._class_dice_gauge_on_reconstructed_iseg_images.compute() if self._class_dice_gauge_on_reconstructed_iseg_images.has_been_updated() else np.array(
                [0.0, 0.0, 0.0])
            self.custom_variables[
                "Dice score per class per epoch on reconstructed MRBrainS image"] = self._class_dice_gauge_on_reconstructed_mrbrains_images.compute() if self._class_dice_gauge_on_reconstructed_mrbrains_images.has_been_updated() else np.array(
                [0.0, 0.0, 0.0])
            self.custom_variables[
                "Dice score per class per epoch on reconstructed ABIDE image"] = self._class_dice_gauge_on_reconstructed_abide_images.compute() if self._class_dice_gauge_on_reconstructed_abide_images.has_been_updated() else np.array(
                [0.0, 0.0, 0.0])

            if self._valid_dice_gauge.compute() > self._previous_mean_dice:
                new_table = to_html_per_dataset(
                    ["CSF", "Grey Matter", "White Matter"],
                    ["DSC", "HD"],
                    [
                        [
                            self._iSEG_dice_gauge.compute() if self._iSEG_dice_gauge.has_been_updated() else np.array(
                                [0.0, 0.0, 0.0]),
                            self._iSEG_hausdorff_gauge.compute() if self._iSEG_hausdorff_gauge.has_been_updated() else np.array(
                                [0.0, 0.0, 0.0])],
                        [
                            self._MRBrainS_dice_gauge.compute() if self._MRBrainS_dice_gauge.has_been_updated() else np.array(
                                [0.0, 0.0, 0.0]),
                            self._MRBrainS_hausdorff_gauge.compute() if self._MRBrainS_hausdorff_gauge.has_been_updated() else np.array(
                                [0.0, 0.0, 0.0])],
                        [
                            self._ABIDE_dice_gauge.compute() if self._ABIDE_dice_gauge.has_been_updated() else np.array(
                                [0.0, 0.0, 0.0]),
                            self._ABIDE_hausdorff_gauge.compute() if self._ABIDE_hausdorff_gauge.has_been_updated() else np.array(
                                [0.0, 0.0, 0.0])]],
                    ["iSEG", "MRBrainS", "ABIDE"])

                self.custom_variables["Per-Dataset Metric Table"] = new_table
                self._previous_mean_dice = self._valid_dice_gauge.compute()
                self._previous_per_dataset_table = new_table
            else:
                self.custom_variables["Per-Dataset Metric Table"] = self._previous_per_dataset_table
            self._valid_dice_gauge.reset()

            self.custom_variables["Jensen-Shannon Table"] = to_html_JS(["Input data", "Generated Data"],
                                                                       ["JS Divergence"],
                                                                       [
                                                                           self._js_div_inputs_gauge.compute() if self._js_div_gen_gauge.has_been_updated() else np.array(
                                                                               [0.0]),
                                                                           self._js_div_gen_gauge.compute() if self._js_div_gen_gauge.has_been_updated() else np.array(
                                                                               [0.0])])
            self.custom_variables["Jensen-Shannon Divergence"] = [
                self._js_div_inputs_gauge.compute(),
                self._js_div_gen_gauge.compute()]
            self.custom_variables["Mean Hausdorff Distance"] = [
                self._class_hausdorff_distance_gauge.compute().mean() if self._class_hausdorff_distance_gauge.has_been_updated() else np.array(
                    [0.0])]
            self.custom_variables["D(G(X)) | X"] = [self._D_G_X_as_X_test_gauge.compute()]
            self.custom_variables["Total Loss"] = [self._total_loss_test_gauge.compute()]

    @staticmethod
    def _merge_tensors(tensor_0, tensor_1):
        return torch.cat((tensor_0, tensor_1), dim=0)

    def _should_activate_autoencoder(self):
        return self._current_epoch < self._patience_segmentation

    def _should_activate_segmentation(self):
        return self._current_epoch >= self._patience_segmentation

    def _update_image_plots(self, inputs, generator_predictions, segmenter_predictions, target, dataset_ids):
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=5, mode="trilinear",
                                                 align_corners=True).numpy()
        generator_predictions = torch.nn.functional.interpolate(generator_predictions, scale_factor=5, mode="trilinear",
                                                                align_corners=True).numpy()
        segmenter_predictions = torch.nn.functional.interpolate(
            torch.argmax(torch.nn.functional.softmax(segmenter_predictions, dim=1), dim=1, keepdim=True).float(),
            scale_factor=5, mode="nearest").numpy()

        target = torch.nn.functional.interpolate(target.float(), scale_factor=5, mode="nearest").numpy()

        self.custom_variables["Input Batch Process {}".format(self._run_config.local_rank)] = self._slicer.get_slice(
            SliceType.AXIAL, inputs)
        self.custom_variables[
            "Generated Batch Process {}".format(self._run_config.local_rank)] = self._slicer.get_slice(SliceType.AXIAL,
                                                                                                       generator_predictions)
        self.custom_variables[
            "Segmented Batch Process {}".format(self._run_config.local_rank)] = self._seg_slicer.get_colored_slice(
            SliceType.AXIAL,
            segmenter_predictions)
        self.custom_variables["Segmentation Ground Truth Batch Process {}".format(
            self._run_config.local_rank)] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                               target)
        self.custom_variables[
            "Label Map Batch Process {}".format(self._run_config.local_rank)] = self._label_mapper.get_label_map(
            dataset_ids)

    @staticmethod
    def _construct_histrogram(gen_pred_iseg, input_iseg, gen_pred_mrbrains, input_mrbrains, gen_pred_abide,
                              input_abide):
        fig1, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2,
                                                                  figsize=(12, 10))

        _, bins, _ = ax1.hist(gen_pred_iseg[gen_pred_iseg > 0].flatten(), bins=128,
                              density=False, label="iSEG")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Generated iSEG Histogram")
        ax1.legend()

        _ = ax2.hist(gen_pred_mrbrains[gen_pred_mrbrains > 0].flatten(), bins=bins,
                     density=False, label="MRBrainS")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Generated MRBrainS Histogram")
        ax2.legend()

        _ = ax3.hist(gen_pred_abide[gen_pred_abide > 0].flatten(), bins=bins,
                     density=False, label="ABIDE")
        ax3.set_xlabel("Intensity")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Generated ABIDE Histogram")
        ax3.legend()

        _, bins, _ = ax4.hist(input_iseg[input_iseg > 0].flatten(), bins=128,
                              density=False, label="iSEG")
        ax4.set_xlabel("Intensity")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Input iSEG Histogram")
        ax4.legend()
        _ = ax5.hist(input_mrbrains[input_mrbrains > 0].flatten(), bins=bins,
                     density=False, label="MRBrainS")
        ax5.set_xlabel("Intensity")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Input MRBrainS Histogram")
        ax5.legend()

        _ = ax6.hist(input_abide[input_abide > 0].flatten(), bins=bins,
                     density=False, label="ABIDE")
        ax6.set_xlabel("Intensity")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Input ABIDE Histogram")
        ax6.legend()

        fig1.tight_layout()
        id = str(uuid.uuid4())
        fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

        fig1.clf()
        plt.close(fig1)

        return "/tmp/histograms-{}.png".format(str(id))

    @staticmethod
    def _construct_double_histrogram(gen_pred_iseg, input_iseg, gen_pred_mrbrains, input_mrbrains):
        fig1, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                      figsize=(12, 10))

        _, bins, _ = ax1.hist(input_iseg[input_iseg > 0].flatten(), bins=128,
                              density=False, label="iSEG")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Input iSEG Histogram")
        ax1.legend()

        _ = ax3.hist(gen_pred_iseg[gen_pred_iseg > 0].flatten(), bins=bins,
                     density=False, label="iSEG")
        ax3.set_xlabel("Intensity")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Generated iSEG Histogram")
        ax3.legend()

        _, bins, _ = ax2.hist(input_mrbrains[input_mrbrains > 0].flatten(), bins=128,
                              density=False, label="MRBrainS")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Input MRBrainS Histogram")
        ax2.legend()

        _ = ax4.hist(gen_pred_mrbrains[gen_pred_mrbrains > 0].flatten(), bins=bins,
                     density=False, label="MRBrainS")
        ax4.set_xlabel("Intensity")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Generated MRBrainS Histogram")
        ax4.legend()

        fig1.tight_layout()
        id = str(uuid.uuid4())
        fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

        fig1.clf()
        plt.close(fig1)

        return "/tmp/histograms-{}.png".format(str(id))

    @staticmethod
    def _construct_single_histrogram(gen_pred, input):
        fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                        figsize=(12, 10))

        _, bins, _ = ax1.hist(input[input > 0].flatten(), bins=128,
                              density=False, label="Input")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Input Histogram")
        ax1.legend()

        _ = ax2.hist(gen_pred[gen_pred > 0].flatten(), bins=bins,
                     density=False, label="iSEG")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Generated Histogram")
        ax2.legend()

        fig1.tight_layout()
        id = str(uuid.uuid4())
        fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

        fig1.clf()
        plt.close(fig1)

        return "/tmp/histograms-{}.png".format(str(id))

    @staticmethod
    def _construct_class_histogram(inputs, target, gen_pred):
        iseg_inputs = inputs[NON_AUGMENTED_INPUTS][torch.where(target[DATASET_ID] == ISEG_ID)]
        iseg_targets = target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG_ID)]
        iseg_gen_pred = gen_pred[torch.where(target[DATASET_ID] == ISEG_ID)]
        mrbrains_inputs = inputs[NON_AUGMENTED_INPUTS][torch.where(target[DATASET_ID] == MRBRAINS_ID)]
        mrbrains_targets = target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBRAINS_ID)]
        mrbrains_gen_pred = gen_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)]
        abide_inputs = inputs[NON_AUGMENTED_INPUTS][torch.where(target[DATASET_ID] == ABIDE_ID)]
        abide_targets = target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ABIDE_ID)]
        abide_gen_pred = gen_pred[torch.where(target[DATASET_ID] == ABIDE_ID)]

        fig1, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(nrows=4, ncols=2,
                                                                              figsize=(12, 10))

        _, bins, _ = ax1.hist(iseg_gen_pred[torch.where(iseg_targets == 0)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax1.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 0)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax1.hist(abide_gen_pred[torch.where(abide_targets == 0)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Generated Background Histogram")
        ax1.legend()

        _, bins, _ = ax2.hist(iseg_gen_pred[torch.where(iseg_targets == 1)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax2.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 1)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax2.hist(abide_gen_pred[torch.where(abide_targets == 1)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Generated CSF Histogram")
        ax2.legend()

        _, bins, _ = ax3.hist(iseg_gen_pred[torch.where(iseg_targets == 2)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax3.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 2)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax3.hist(abide_gen_pred[torch.where(abide_targets == 2)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax3.set_xlabel("Intensity")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Generated Gray Matter Histogram")
        ax3.legend()

        _, bins, _ = ax4.hist(iseg_gen_pred[torch.where(iseg_targets == 3)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax4.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 3)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax4.hist(abide_gen_pred[torch.where(abide_targets == 3)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax4.set_xlabel("Intensity")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Generated White Matter Histogram")
        ax4.legend()

        _, bins, _ = ax5.hist(iseg_inputs[torch.where(iseg_targets == 0)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax5.hist(mrbrains_inputs[torch.where(mrbrains_targets == 0)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax5.hist(abide_inputs[torch.where(abide_targets == 0)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax5.set_xlabel("Intensity")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Input Background Histogram")
        ax5.legend()

        _, bins, _ = ax6.hist(iseg_inputs[torch.where(iseg_targets == 1)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax6.hist(mrbrains_inputs[torch.where(mrbrains_targets == 1)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax6.hist(abide_inputs[torch.where(abide_targets == 1)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax6.set_xlabel("Intensity")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Input CSF Histogram")
        ax6.legend()

        _, bins, _ = ax7.hist(iseg_inputs[torch.where(iseg_targets == 2)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax7.hist(mrbrains_inputs[torch.where(mrbrains_targets == 2)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax7.hist(abide_inputs[torch.where(abide_targets == 2)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax7.set_xlabel("Intensity")
        ax7.set_ylabel("Frequency")
        ax7.set_title("Input Gray Matter Histogram")
        ax7.legend()

        _, bins, _ = ax8.hist(iseg_inputs[torch.where(iseg_targets == 3)].cpu().detach().numpy(), bins=128,
                              density=False, label="iSEG")
        _ = ax8.hist(mrbrains_inputs[torch.where(mrbrains_targets == 3)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="MRBrainS")
        _ = ax8.hist(abide_inputs[torch.where(abide_targets == 3)].cpu().detach().numpy(), bins=bins,
                     alpha=0.75, density=False, label="ABIDE")
        ax8.set_xlabel("Intensity")
        ax8.set_ylabel("Frequency")
        ax8.set_title("Input White Matter Histogram")
        ax8.legend()
        fig1.tight_layout()
        id = str(uuid.uuid4())
        fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

        fig1.clf()
        plt.close(fig1)
        return "/tmp/histograms-{}.png".format(str(id))

    @staticmethod
    def _count(tensor, n_classes):
        count = torch.Tensor().new_zeros(size=(n_classes,), device="cpu")
        for i in range(n_classes):
            count[i] = torch.sum(tensor == i).int()
        return count

    def _make_disc_pie_plots(self, disc_pred, inputs, target):
        count = self._count(torch.argmax(disc_pred.cpu().detach(), dim=1), self._num_datasets + 1)
        real_count = self._count(torch.cat((target[DATASET_ID].cpu().detach(), torch.Tensor().new_full(
            size=(inputs[NON_AUGMENTED_INPUTS].size(0) // 2,),
            fill_value=self._num_datasets,
            dtype=torch.long,
            device="cpu",
            requires_grad=False)), dim=0), self._num_datasets + 1)
        self.custom_variables["Pie Plot"] = count
        self.custom_variables["Pie Plot True"] = real_count

    def _evaluate_loss_D_G_X_as_X(self, inputs, target):
        pred_D_G_X = self._discriminator.forward(inputs)
        ones = torch.Tensor().new_ones(size=pred_D_G_X.size(), device=pred_D_G_X.device, dtype=pred_D_G_X.dtype,
                                       requires_grad=False)
        loss_D_G_X_as_X = self._discriminator.compute_loss("NLLLoss",
                                                           torch.nn.functional.log_softmax(
                                                               ones - torch.nn.functional.softmax(pred_D_G_X, dim=1),
                                                               dim=1),
                                                           target)

        return loss_D_G_X_as_X

    def _compute_mean_hausdorff_distance(self, seg_pred, target):
        distances = np.zeros((4,))
        for channel in range(seg_pred.size(1)):
            distances[channel] = max(
                directed_hausdorff(
                    flatten(seg_pred[:, channel, ...]).cpu().detach().numpy(),
                    flatten(target[:, channel, ...]).cpu().detach().numpy())[0],
                directed_hausdorff(
                    flatten(target[:, channel, ...]).cpu().detach().numpy(),
                    flatten(seg_pred[:, channel, ...]).cpu().detach().numpy())[0])
        return distances

    def _train_discriminator(self, inputs, gen_pred, target):
        # Forward on real data.
        pred_D_X, x_conv1, x_layer1, x_layer2, x_layer3 = self._discriminator.forward(inputs)

        # Compute loss on real data with real targets.
        loss_D_X = self._discriminator.compute_loss("NLLLoss", torch.nn.functional.log_softmax(pred_D_X, dim=1), target)

        # Forward on fake data.
        pred_D_G_X, _, _, _, _ = self._discriminator.forward(gen_pred)

        # Choose randomly 8 predictions (to balance with real domains).
        # choices = np.random.choice(a=pred_D_G_X.size(0), size=(int(pred_D_G_X.size(0) / 2),), replace=False)
        # pred_D_G_X = pred_D_G_X[choices]

        # Forge bad class (K+1) tensor.
        y_bad = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=self._num_datasets,
                                        dtype=torch.long, device=target.device, requires_grad=False)

        # Compute loss on fake predictions with bad class tensor.
        loss_D_G_X = self._discriminator.compute_loss("NLLLoss", torch.nn.functional.log_softmax(pred_D_G_X, dim=1),
                                                      y_bad)

        # disc_loss = ((self._num_datasets / self._num_datasets + 1.0) * loss_D_X + (
        #         1.0 / self._num_datasets + 1.0) * loss_D_G_X) / 2.0

        disc_loss = (loss_D_X + loss_D_G_X) / 2.0

        self._discriminator.update_train_loss("NLLLoss", disc_loss)

        pred = self._merge_tensors(pred_D_X, pred_D_G_X)
        target = self._merge_tensors(target, y_bad)

        metric = self._discriminator.compute_metrics(pred, target)
        self._discriminator.update_train_metrics(metric)

        return disc_loss, pred, target, x_conv1, x_layer1, x_layer2, x_layer3

    def _validate_discriminator(self, inputs, gen_pred, target, test=False):
        # Forward on real data.
        pred_D_X = self._discriminator.forward(inputs)

        # Compute loss on real data with real targets.
        loss_D_X = self._discriminator.compute_loss("NLLLoss", torch.nn.functional.log_softmax(pred_D_X, dim=1), target)

        # Forward on fake data.
        pred_D_G_X = self._discriminator.forward(gen_pred)

        # Choose randomly 8 predictions (to balance with real domains).
        # choices = np.random.choice(a=pred_D_G_X.size(0), size=(int(pred_D_G_X.size(0) / 2),), replace=False)
        # pred_D_G_X = pred_D_G_X[choices]

        # Forge bad class (K+1) tensor.
        y_bad = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=self._num_datasets,
                                        dtype=torch.long, device=target.device, requires_grad=False)

        # Compute loss on fake predictions with bad class tensor.
        loss_D_G_X = self._discriminator.compute_loss("NLLLoss", torch.nn.functional.log_softmax(pred_D_G_X, dim=1),
                                                      y_bad)

        # disc_loss = ((self._num_datasets / self._num_datasets + 1.0) * loss_D_X + (
        #         1.0 / self._num_datasets + 1.0) * loss_D_G_X) / 2.0

        disc_loss = (loss_D_X + loss_D_G_X) / 2.0

        pred = self._merge_tensors(pred_D_X, pred_D_G_X)
        target = self._merge_tensors(target, y_bad)

        metric = self._discriminator.compute_metrics(pred, target)
        if test:
            self._discriminator.update_test_loss("NLLLoss", disc_loss)
            self._discriminator.update_test_metrics(metric)
        else:
            self._discriminator.update_valid_loss("NLLLoss", disc_loss)
            self._discriminator.update_valid_metrics(metric)

        return disc_loss, pred, target
