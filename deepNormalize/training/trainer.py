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
from kerosene.config.trainers import RunConfiguration
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.distributed import on_single_device
from kerosene.utils.tensors import flatten, to_onehot
from torch.utils.data import DataLoader

from deepNormalize.inputs.images import SliceType
from deepNormalize.logger.image_slicer import AdaptedImageSlicer
from deepNormalize.utils.constants import GENERATOR, SEGMENTER, DISCRIMINATOR, IMAGE_TARGET, DATASET_ID, EPSILON


class DeepNormalizeTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, run_config: RunConfiguration):
        super(DeepNormalizeTrainer, self).__init__("DeepNormalizeTrainer", train_data_loader, valid_data_loader,
                                                   model_trainers, run_config)

        self._training_config = training_config
        self._patience_discriminator = training_config.patience_discriminator
        self._patience_segmentation = training_config.patience_segmentation
        self._with_discriminator = None
        self._with_segmentation = None
        self._generator_should_be_autoencoder = None
        self._slicer = AdaptedImageSlicer()
        self._generator = self._model_trainers[GENERATOR]
        self._discriminator = self._model_trainers[DISCRIMINATOR]
        self._segmenter = self._model_trainers[SEGMENTER]

    def train_step(self, inputs, target):
        disc_pred = None

        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_train_loss(gen_pred, inputs)
            gen_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._generator)

            self._generator.step()
            self._generator.zero_grad()

            if self.current_train_step % self._training_config.variables["train_discriminator_every_n_steps"] == 0:
                disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred.detach(), target[DATASET_ID])
                disc_loss.backward()

                if not on_single_device(self._run_config.devices):
                    self.average_gradients(self._discriminator)

                self._discriminator.step()
                self._discriminator.zero_grad()

        if self._should_activate_discriminator_loss():
            loss_D_G_X_as_X = self.evaluate_loss_D_G_X_as_X(gen_pred, torch.Tensor().new_full(size=(inputs.size(0),),
                                                                                              fill_value=2,
                                                                                              dtype=torch.long,
                                                                                              device=inputs.device,
                                                                                              requires_grad=False))
            gen_loss = self._training_config.variables["lambda"] * loss_D_G_X_as_X
            gen_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._generator)

            self._generator.step()
            self._generator.zero_grad()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred.detach(), target[DATASET_ID])
            disc_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._discriminator)

            self._discriminator.step()
            self._discriminator.zero_grad()

        if self._should_activate_segmentation():
            seg_pred = self._segmenter.forward(gen_pred.detach())
            seg_loss = self._segmenter.compute_train_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                                          to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                                    num_classes=4))
            seg_loss.backward()

            loss_D_G_X_as_X = self.evaluate_loss_D_G_X_as_X(gen_pred, torch.Tensor().new_full(size=(inputs.size(0),),
                                                                                              fill_value=2,
                                                                                              dtype=torch.long,
                                                                                              device=inputs.device,
                                                                                              requires_grad=False))
            gen_loss = self._training_config.variables["lambda"] * (loss_D_G_X_as_X + seg_loss.loss.data)
            gen_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._segmenter)
                self.average_gradients(self._generator)

            self._generator.step()
            self._segmenter.step()
            self._generator.zero_grad()
            self._segmenter.zero_grad()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred.detach(), target[DATASET_ID])
            disc_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._discriminator)
            self._discriminator.step()
            self._discriminator.zero_grad()

        if disc_pred is not None:
            count = self.count(torch.argmax(disc_pred, dim=1), 3)
            self.custom_variables["Pie Plot"] = count

        if self.current_train_step % 100 == 0:
            self._update_plots(inputs, gen_pred)

        self.custom_variables["Generated Intensity Histogram"] = flatten(gen_pred.cpu())
        self.custom_variables["Input Intensity Histogram"] = flatten(inputs.cpu())

    def validate_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            self._generator.compute_valid_loss(gen_pred, inputs)
            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID])

        if self._should_activate_discriminator_loss():
            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID])

        if self._should_activate_segmentation():
            seg_pred = self._segmenter.forward(gen_pred)
            self._segmenter.compute_valid_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                               to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(),
                                                         num_classes=4))

    def _update_plots(self, inputs, generator_predictions):
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=5, mode="trilinear",
                                                 align_corners=True).cpu().numpy()
        generator_predictions = torch.nn.functional.interpolate(generator_predictions, scale_factor=5, mode="trilinear",
                                                                align_corners=True).cpu().detach().numpy()

        inputs = self._normalize(inputs)
        generator_predictions = self._normalize(generator_predictions)

        self.custom_variables["Input Batch"] = self._slicer.get_slice(SliceType.AXIAL, inputs)
        self.custom_variables["Generated Batch"] = self._slicer.get_slice(SliceType.AXIAL, generator_predictions)

    def scheduler_step(self):
        self._generator.scheduler_step()

        if self._should_activate_discriminator_loss():
            self._discriminator.scheduler_step()

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
        self._with_discriminator = self._should_activate_discriminator_loss()

    def on_epoch_end(self):
        pass

    @staticmethod
    def count(tensor, n_classes):
        count = torch.Tensor().new_zeros(size=(n_classes,), device="cpu")
        for i in range(n_classes):
            count[i] = torch.sum(tensor == i).int()
        return count

    def evaluate_loss_D_G_X_as_X(self, inputs, target):
        pred_D_G_X = self._discriminator.forward(inputs)
        ones = torch.Tensor().new_ones(size=pred_D_G_X.size(), device=pred_D_G_X.device, dtype=pred_D_G_X.dtype)
        loss_D_G_X_as_X = self._discriminator.compute_train_loss(ones - pred_D_G_X, target)
        return loss_D_G_X_as_X

    def train_discriminator(self, inputs, gen_pred, target):
        # Forward on real data.
        pred_D_X = self._discriminator.forward(inputs)

        # Compute loss on real data with real targets.
        loss_D_X = self._discriminator.compute_train_loss(pred_D_X, target)

        # Forward on fake data.
        pred_D_G_X = self._discriminator.forward(gen_pred)

        # Choose randomly 6 predictions (to balance with real domains).
        choices = np.random.choice(a=pred_D_G_X.size(0), size=(int(pred_D_G_X.size(0) / 2),), replace=True)
        pred_D_G_X = pred_D_G_X[choices]

        # Forge bad class (K+1) tensor.
        y_bad = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=2, dtype=torch.long,
                                        device=target.device, requires_grad=False)

        # Compute loss on fake predictions with bad class tensor.
        loss_D_G_X = self._discriminator.compute_train_loss(pred_D_G_X, y_bad)

        disc_loss = (loss_D_X + ((1 / 3) * loss_D_G_X)) * 0.5  # 1/3 because fake images represents 1/3 of total count.

        pred = self.merge_tensors(pred_D_X, pred_D_G_X)
        target = self.merge_tensors(target, y_bad)

        self._discriminator.compute_train_metric(pred, target)

        return disc_loss, pred

    def validate_discriminator(self, inputs, gen_pred, target):
        # Choose randomly 6 predictions (to balance with real domains).
        choices = np.random.choice(a=gen_pred.size(0), size=(int(gen_pred.size(0) / 2),), replace=True)
        gen_pred = gen_pred[choices]

        # Forge bad class (K+1) tensor.
        y_bad = torch.Tensor().new_full(size=(gen_pred.size(0),), fill_value=2, dtype=torch.long,
                                        device=target.device, requires_grad=False)

        merged_inputs = self.merge_tensors(inputs, gen_pred)
        merged_targets = self.merge_tensors(target, y_bad)

        choices = np.random.choice(a=merged_inputs.size(0), size=(int(merged_inputs.size(0)),), replace=True)
        inputs = merged_inputs[choices]
        target = merged_targets[choices]

        pred = self._discriminator.forward(inputs)

        disc_loss = self._discriminator.compute_valid_loss(pred, target)
        self._discriminator.compute_valid_metric(pred, target)
        return disc_loss, pred
