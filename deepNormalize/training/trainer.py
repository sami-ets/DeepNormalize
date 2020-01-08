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
import pynvml
import torch
from fastai.utils.mem import gpu_mem_get
from ignite.metrics.confusion_matrix import ConfusionMatrix
from kerosene.configs.configs import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from kerosene.nn.functional import js_div
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.devices import on_multiple_gpus
from kerosene.utils.tensors import flatten, to_onehot
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader, Dataset

from deepNormalize.inputs.images import SliceType
from deepNormalize.utils.constants import GENERATOR, SEGMENTER, DISCRIMINATOR, IMAGE_TARGET, DATASET_ID
from deepNormalize.utils.constants import ISEG, MRBrainS
from deepNormalize.utils.image_slicer import ImageSlicer, SegmentationSlicer, ImageReconstructor
from deepNormalize.utils.utils import to_html, to_html_per_dataset, to_html_JS, to_html_time


class DeepNormalizeTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, test_data_loader: DataLoader,
                 reconstruction_datasets: List[Dataset], run_config: RunConfiguration):
        super(DeepNormalizeTrainer, self).__init__("DeepNormalizeTrainer", train_data_loader, valid_data_loader,
                                                   test_data_loader, model_trainers, run_config)

        self._training_config = training_config
        self._run_config = run_config
        self._patience_segmentation = training_config.patience_segmentation
        self._slicer = ImageSlicer()
        self._seg_slicer = SegmentationSlicer()
        self._reconstruction_datasets = reconstruction_datasets
        self._normalized_reconstructor = ImageReconstructor([128, 160, 160], [32, 32, 32], [8, 8, 8],
                                                            [self._model_trainers[GENERATOR]], normalize=True)
        self._segmented_reconstructor = ImageReconstructor([128, 160, 160], [32, 32, 32], [8, 8, 8],
                                                           [self._model_trainers[GENERATOR],
                                                            self._model_trainers[SEGMENTER]], segment=True)
        self._input_reconstructor = ImageReconstructor([128, 160, 160], [32, 32, 32], [8, 8, 8])
        self._generator = self._model_trainers[GENERATOR]
        self._discriminator = self._model_trainers[DISCRIMINATOR]
        self._segmenter = self._model_trainers[SEGMENTER]
        self._D_G_X_as_X_training_gauge = AverageGauge()
        self._D_G_X_as_X_validation_gauge = AverageGauge()
        self._D_G_X_as_X_test_gauge = AverageGauge()
        self._class_hausdorff_distance_gauge = AverageGauge()
        self._mean_hausdorff_distance_gauge = AverageGauge()
        self._per_dataset_hausdorff_distance_gauge = AverageGauge()
        self._iSEG_dice_gauge = AverageGauge()
        self._MRBrainS_dice_gauge = AverageGauge()
        self._iSEG_hausdorff_gauge = AverageGauge()
        self._MRBrainS_hausdorff_gauge = AverageGauge()
        self._class_dice_gauge = AverageGauge()
        self._js_div_inputs_gauge = AverageGauge()
        self._js_div_gen_gauge = AverageGauge()
        self._general_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._iSEG_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._MRBrainS_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._discriminator_confusion_matrix_gauge = ConfusionMatrix(num_classes=3)
        self._start_time = 0
        self._stop_time = 0
        print("Total number of parameters: {}".format(sum(p.numel() for p in self._segmenter.parameters()) +
                                                      sum(p.numel() for p in self._generator.parameters()) +
                                                      sum(p.numel() for p in self._discriminator.parameters())))
        pynvml.nvmlInit()

    def train_step(self, inputs, target):
        disc_pred = None
        seg_pred = torch.Tensor().new_zeros(
            size=(self._training_config.batch_size, 1, 32, 32, 32), dtype=torch.float, device="cpu")

        gen_pred = torch.nn.functional.relu(self._generator.forward(inputs))

        if self._should_activate_autoencoder():
            self._generator.zero_grad()
            self._discriminator.zero_grad()

            if self.current_train_step % self._training_config.variables["train_generator_every_n_steps"] == 0:
                gen_loss = self._generator.compute_loss(gen_pred, inputs)
                self._generator.update_train_loss(gen_loss.loss)
                gen_loss.backward()

                self._generator.step()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred.detach(), target[DATASET_ID])
            disc_loss.backward()

            self._discriminator.step()

            # Pretrain segmenter.
            seg_pred = self._segmenter.forward(gen_pred.detach())
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_train_loss(seg_loss.mean().loss)
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            self._segmenter.update_train_metrics(metric)

            seg_loss.mean().backward()

            self._segmenter.step()

        if self._should_activate_segmentation():
            self._generator.zero_grad()
            self._discriminator.zero_grad()
            self._segmenter.zero_grad()

            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_train_loss(gen_loss.loss)

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_train_loss(seg_loss.mean().loss)

            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            self._segmenter.update_train_metrics(metric)

            if self.current_train_step % self._training_config.variables["train_generator_every_n_steps_seg"] == 0:
                disc_loss_as_X = self.evaluate_loss_D_G_X_as_X(gen_pred,
                                                               torch.Tensor().new_full(
                                                                   size=(inputs.size(0),),
                                                                   fill_value=2,
                                                                   dtype=torch.long,
                                                                   device=inputs.device,
                                                                   requires_grad=False))
                self._D_G_X_as_X_training_gauge.update(float(disc_loss_as_X.loss))
                total_loss = self._training_config.variables["disc_ratio"] * disc_loss_as_X + \
                             self._training_config.variables["seg_ratio"] * seg_loss.mean()

                total_loss.backward()

                self._segmenter.step()
                self._generator.step()

            else:
                seg_loss.mean().backward()

                self._segmenter.step()
                self._generator.step()

            self._discriminator.zero_grad()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred.detach(), target[DATASET_ID])
            disc_loss.backward()

            self._discriminator.step()

        if disc_pred is not None:
            count = self.count(torch.argmax(disc_pred.cpu().detach(), dim=1), 3)
            real_count = self.count(torch.cat((target[DATASET_ID].cpu().detach(), torch.Tensor().new_full(
                size=(inputs.size(0) // 2,),
                fill_value=2,
                dtype=torch.long,
                device="cpu",
                requires_grad=False)), dim=0), 3)
            self.custom_variables["Pie Plot"] = count
            self.custom_variables["Pie Plot True"] = real_count

        if self._run_config.local_rank == 0:
            if self.current_train_step % 100 == 0:
                self._update_plots(inputs.cpu().detach(), gen_pred.cpu().detach(), seg_pred.cpu().detach(),
                                   target[IMAGE_TARGET].cpu().detach())

            self.custom_variables["Generated Intensity Histogram"] = flatten(gen_pred.cpu().detach())
            self.custom_variables["Input Intensity Histogram"] = flatten(inputs.cpu().detach())
            self.custom_variables["Background Generated Intensity Histogram"] = gen_pred[
                torch.where(target[IMAGE_TARGET] == 0)].cpu().detach()
            self.custom_variables["CSF Generated Intensity Histogram"] = gen_pred[
                torch.where(target[IMAGE_TARGET] == 1)].cpu().detach()
            self.custom_variables["GM Generated Intensity Histogram"] = gen_pred[
                torch.where(target[IMAGE_TARGET] == 2)].cpu().detach()
            self.custom_variables["WM Generated Intensity Histogram"] = gen_pred[
                torch.where(target[IMAGE_TARGET] == 3)].cpu().detach()
            self.custom_variables["Background Input Intensity Histogram"] = inputs[
                torch.where(target[IMAGE_TARGET] == 0)].cpu().detach()
            self.custom_variables["CSF Input Intensity Histogram"] = inputs[
                torch.where(target[IMAGE_TARGET] == 1)].cpu().detach()
            self.custom_variables["GM Input Intensity Histogram"] = inputs[
                torch.where(target[IMAGE_TARGET] == 2)].cpu().detach()
            self.custom_variables["WM Input Intensity Histogram"] = inputs[
                torch.where(target[IMAGE_TARGET] == 3)].cpu().detach()

    def validate_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_valid_loss(gen_loss.loss)

            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID])

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_valid_loss(seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            self._segmenter.update_valid_metrics(metric)

        if self._should_activate_segmentation():
            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_valid_loss(gen_loss.loss)

            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID])

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_valid_loss(seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            self._segmenter.update_valid_metrics(metric)

            disc_loss_as_X = self.evaluate_loss_D_G_X_as_X(gen_pred,
                                                           torch.Tensor().new_full(
                                                               size=(inputs.size(0),),
                                                               fill_value=2,
                                                               dtype=torch.long,
                                                               device=inputs.device,
                                                               requires_grad=False))
            self._D_G_X_as_X_validation_gauge.update(float(disc_loss_as_X.loss))

    def test_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_test_loss(gen_loss.loss)

            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID], test=True)

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_test_loss(seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            self._segmenter.update_test_metrics(metric)

        if self._should_activate_segmentation():
            gen_loss = self._generator.compute_loss(gen_pred, inputs)
            self._generator.update_test_loss(gen_loss.loss)

            _, disc_pred = self.validate_discriminator(inputs, gen_pred, target[DATASET_ID], test=True)

            seg_pred = self._segmenter.forward(gen_pred)
            seg_loss = self._segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                              num_classes=4))
            self._segmenter.update_test_loss(seg_loss.mean())
            metric = self._segmenter.compute_metrics(torch.nn.functional.softmax(seg_pred, dim=1),
                                                     torch.squeeze(target[IMAGE_TARGET], dim=1).long())
            metric["Dice"] = metric["Dice"].mean()
            self._segmenter.update_test_metrics(metric)

            self._class_dice_gauge.update(np.array(metric.numpy()))

            if seg_pred[torch.where(target[DATASET_ID] == ISEG)].shape[0] != 0:
                self._iSEG_dice_gauge.update(np.array(self._segmenter.compute_metrics(
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
                self._MRBrainS_dice_gauge.update(np.array(self._segmenter.compute_metrics(
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

            disc_loss_as_X = self.evaluate_loss_D_G_X_as_X(gen_pred,
                                                           torch.Tensor().new_full(
                                                               size=(inputs.size(0),),
                                                               fill_value=2,
                                                               dtype=torch.long,
                                                               device=inputs.device,
                                                               requires_grad=False))
            self._D_G_X_as_X_test_gauge.update(float(disc_loss_as_X.loss))

            self._class_hausdorff_distance_gauge.update(
                self.compute_mean_hausdorff_distance(
                    to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4),
                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(), num_classes=4))[-3:])

            self._general_confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                          num_classes=4),
                torch.squeeze(target[IMAGE_TARGET].long(), dim=1)))

            self._discriminator_confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(disc_pred, dim=1), dim=1), num_classes=3),
                torch.squeeze(target[DATASET_ID].long(), dim=1)))

            inputs_reshaped = inputs.reshape(inputs.shape[0],
                                             inputs.shape[1] * inputs.shape[2] * inputs.shape[3] * inputs.shape[4])

            gen_pred_reshaped = gen_pred.reshape(gen_pred.shape[0],
                                                 gen_pred.shape[1] * gen_pred.shape[2] * gen_pred.shape[3] *
                                                 gen_pred.shape[4])
            inputs_ = torch.Tensor().new_zeros((inputs_reshaped.shape[0], 256))
            gen_pred_ = torch.Tensor().new_zeros((gen_pred_reshaped.shape[0], 256))
            for image in range(inputs_reshaped.shape[0]):
                inputs_[image] = torch.nn.functional.softmax(torch.histc(inputs_reshaped[image], bins=256), dim=0)
                gen_pred_[image] = torch.nn.functional.softmax(torch.histc(gen_pred_reshaped[image].float(), bins=256),
                                                               dim=0)

            self._js_div_inputs_gauge.update(js_div(inputs_))
            self._js_div_gen_gauge.update(js_div(gen_pred_))

    def _update_plots(self, inputs, generator_predictions, segmenter_predictions, target):
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=5, mode="trilinear",
                                                 align_corners=True).numpy()
        generator_predictions = torch.nn.functional.interpolate(generator_predictions, scale_factor=5, mode="trilinear",
                                                                align_corners=True).numpy()
        segmenter_predictions = torch.nn.functional.interpolate(
            torch.argmax(torch.nn.functional.softmax(segmenter_predictions, dim=1), dim=1, keepdim=True).float(),
            scale_factor=5, mode="nearest").numpy()

        target = torch.nn.functional.interpolate(target.float(), scale_factor=5, mode="nearest").numpy()

        self.custom_variables["Input Batch"] = self._slicer.get_slice(SliceType.AXIAL, inputs)
        self.custom_variables["Generated Batch"] = self._slicer.get_slice(SliceType.AXIAL, generator_predictions)
        self.custom_variables["Segmented Batch"] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                                      segmenter_predictions)
        self.custom_variables["Segmentation Ground Truth Batch"] = self._seg_slicer.get_colored_slice(SliceType.AXIAL,
                                                                                                      target)

    def scheduler_step(self):
        self._generator.scheduler_step()

        if self._should_activate_segmentation():
            self._discriminator.scheduler_step()
            self._segmenter.scheduler_step()

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
        return self._current_epoch < self._patience_segmentation

    def _should_activate_segmentation(self):
        return self._current_epoch >= self._patience_segmentation

    def on_training_begin(self):
        self._start_time = time.time()

    def on_training_end(self):
        self._stop_time = time.time()

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        self.custom_variables["GPU {} Memory".format(self._run_config.local_rank)] = \
            [np.array(gpu_mem_get(self._run_config.local_rank))]

        if self._run_config.local_rank == 0:
            self.custom_variables["Runtime"] = to_html_time(timedelta(seconds=time.time() - self._start_time))

            all_patches = list(
                map(lambda dataset: [sample.x for _, sample in enumerate(dataset)], self._reconstruction_datasets))
            img = list(
                map(lambda patches: self._normalized_reconstructor.reconstruct_from_patches_3d(patches), all_patches))
            img_input = list(
                map(lambda patches: self._normalized_reconstructor.reconstruct_from_patches_3d(patches), all_patches))
            img_seg = list(
                map(lambda patches: self._normalized_reconstructor.reconstruct_from_patches_3d(patches), all_patches))

            self.custom_variables["Reconstructed Normalized iSEG Image"] = self._slicer.get_slice(SliceType.AXIAL,
                                                                                                  np.expand_dims(
                                                                                                      np.expand_dims(
                                                                                                          img[0], 0),
                                                                                                      0))
            self.custom_variables["Reconstructed Segmented iSEG Image"] = self._seg_slicer.get_colored_slice(
                SliceType.AXIAL, np.expand_dims(np.expand_dims(img_seg[0], 0), 0)).squeeze(0)
            self.custom_variables["Reconstructed Input iSEG Image"] = self._slicer.get_slice(SliceType.AXIAL,
                                                                                             np.expand_dims(
                                                                                                 np.expand_dims(
                                                                                                     img_input[0], 0),
                                                                                                 0))

            self.custom_variables["Reconstructed Normalized MRBrainS Image"] = self._slicer.get_slice(SliceType.AXIAL,
                                                                                                      np.expand_dims(
                                                                                                          np.expand_dims(
                                                                                                              img[1],
                                                                                                              0), 0))
            self.custom_variables["Reconstructed Segmented MRBrainS Image"] = self._seg_slicer.get_colored_slice(
                SliceType.AXIAL, np.expand_dims(np.expand_dims(img_seg[1], 0), 0)).squeeze(0)
            self.custom_variables["Reconstructed Input MRBrainS Image"] = self._slicer.get_slice(SliceType.AXIAL,
                                                                                                 np.expand_dims(
                                                                                                     np.expand_dims(
                                                                                                         img_input[1],
                                                                                                         0), 0))
            if self._general_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["Confusion Matrix"] = np.array(
                    np.rot90(self._general_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["Confusion Matrix"] = np.zeros((4, 4))

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

            if self._discriminator_confusion_matrix_gauge._num_examples != 0:
                self.custom_variables["Discriminator Confusion Matrix"] = np.array(
                    np.rot90(self._discriminator_confusion_matrix_gauge.compute().cpu().detach().numpy()))
            else:
                self.custom_variables["Discriminator Confusion Matrix"] = np.zeros((3, 3))

            if self._should_activate_autoencoder():
                self.custom_variables["D(G(X)) | X"] = [np.array([0, 0, 0])]
                self.custom_variables["Metric Table"] = to_html(["CSF", "Grey Matter", "White Matter"], ["DSC", "HD"],
                                                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                self.custom_variables["Per-Dataset Metric Table"] = to_html_per_dataset(
                    ["CSF", "Grey Matter", "White Matter"],
                    ["DSC", "HD"],
                    [[[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]]],
                    ["iSEG", "MRBrainS"])
                self.custom_variables["Jensen-Shannon Table"] = to_html_JS(["Input data", "Generated Data"],
                                                                           ["JS Divergence"],
                                                                           [0.0, 0.0])
                self.custom_variables["Jensen-Shannon Divergence"] = np.zeros((2,))
                self.custom_variables["Mean Hausdorff Distance"] = np.zeros((1,))

            if self._should_activate_segmentation():
                self.custom_variables["D(G(X)) | X"] = [np.array([self._D_G_X_as_X_training_gauge.compute(),
                                                                  self._D_G_X_as_X_validation_gauge.compute(),
                                                                  self._D_G_X_as_X_test_gauge.compute()])]

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
                                                                            self._js_div_gen_gauge.compute().numpy()])
                self.custom_variables["Jensen-Shannon Divergence"] = np.array(
                    [self._js_div_inputs_gauge.compute().numpy(), self._js_div_gen_gauge.compute().numpy()])

                self.custom_variables["Mean Hausdorff Distance"] = np.array(
                    [self._class_hausdorff_distance_gauge.compute().mean()])

        if on_multiple_gpus(self._run_config.devices):
            torch.distributed.barrier()

        self._D_G_X_as_X_training_gauge.reset()
        self._D_G_X_as_X_validation_gauge.reset()
        self._D_G_X_as_X_test_gauge.reset()
        self._MRBrainS_confusion_matrix_gauge.reset()
        self._iSEG_confusion_matrix_gauge.reset()
        self._general_confusion_matrix_gauge.reset()
        self._discriminator_confusion_matrix_gauge.reset()
        self._class_hausdorff_distance_gauge.reset()
        self._class_dice_gauge.reset()
        self._js_div_inputs_gauge.reset()
        self._js_div_gen_gauge.reset()

        if on_multiple_gpus(self._run_config.devices):
            torch.distributed.barrier()

    @staticmethod
    def count(tensor, n_classes):
        count = torch.Tensor().new_zeros(size=(n_classes,), device="cpu")
        for i in range(n_classes):
            count[i] = torch.sum(tensor == i).int()
        return count

    def evaluate_loss_D_G_X_as_X(self, inputs, target):
        pred_D_G_X = self._discriminator.forward(inputs)
        ones = torch.Tensor().new_ones(size=pred_D_G_X.size(), device=pred_D_G_X.device, dtype=pred_D_G_X.dtype,
                                       requires_grad=False)
        loss_D_G_X_as_X = self._discriminator.compute_loss(
            torch.nn.functional.log_softmax(ones - torch.nn.functional.softmax(pred_D_G_X, dim=1), dim=1),
            target)
        return loss_D_G_X_as_X

    def train_discriminator(self, inputs, gen_pred, target):
        # Forward on real data.
        pred_D_X = self._discriminator.forward(inputs)

        # Compute loss on real data with real targets.
        loss_D_X = self._discriminator.compute_loss(torch.nn.functional.log_softmax(pred_D_X, dim=1), target)

        # Forward on fake data.
        pred_D_G_X = self._discriminator.forward(gen_pred)

        # Choose randomly 8 predictions (to balance with real domains).
        choices = np.random.choice(a=pred_D_G_X.size(0), size=(int(pred_D_G_X.size(0) / 2),), replace=True)
        pred_D_G_X = pred_D_G_X[choices]

        # Forge bad class (K+1) tensor.
        y_bad = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=2, dtype=torch.long,
                                        device=target.device, requires_grad=False)

        # Compute loss on fake predictions with bad class tensor.
        loss_D_G_X = self._discriminator.compute_loss(torch.nn.functional.log_softmax(pred_D_G_X, dim=1), y_bad)

        disc_loss = ((2 / 3) * loss_D_X +
                     ((1 / 3) * loss_D_G_X)) * 0.5  # 1/3 because fake images represents 1/3 of total count.
        self._discriminator.update_train_loss(disc_loss.loss)

        pred = self.merge_tensors(pred_D_X, pred_D_G_X)
        target = self.merge_tensors(target, y_bad)

        metric = self._discriminator.compute_metrics(pred, target)
        self._discriminator.update_train_metrics(metric)

        return disc_loss, pred

    def validate_discriminator(self, inputs, gen_pred, target, test=False):
        # Forward on real data.
        pred_D_X = self._discriminator.forward(inputs)

        # Compute loss on real data with real targets.
        loss_D_X = self._discriminator.compute_loss(torch.nn.functional.log_softmax(pred_D_X, dim=1), target)

        # Forward on fake data.
        pred_D_G_X = self._discriminator.forward(gen_pred)

        # Choose randomly 6 predictions (to balance with real domains).
        choices = np.random.choice(a=pred_D_G_X.size(0), size=(int(pred_D_G_X.size(0) / 2),), replace=True)
        pred_D_G_X = pred_D_G_X[choices]

        # Forge bad class (K+1) tensor.
        y_bad = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=2, dtype=torch.long,
                                        device=target.device, requires_grad=False)

        # Compute loss on fake predictions with bad class tensor.
        loss_D_G_X = self._discriminator.compute_loss(torch.nn.functional.log_softmax(pred_D_G_X, dim=1), y_bad)

        disc_loss = ((2 / 3) * loss_D_X +
                     ((1 / 3) * loss_D_G_X)) * 0.5  # 1/3 because fake images represents 1/3 of total count.

        pred = self.merge_tensors(pred_D_X, pred_D_G_X)
        target = self.merge_tensors(target, y_bad)

        metric = self._discriminator.compute_metrics(pred, target)
        if test:
            self._discriminator.update_test_loss(disc_loss)
            self._discriminator.update_test_metrics(metric)
        else:
            self._discriminator.update_valid_loss(disc_loss)
            self._discriminator.update_valid_metrics(metric)

        return disc_loss, pred

    def compute_mean_hausdorff_distance(self, seg_pred, target):
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
