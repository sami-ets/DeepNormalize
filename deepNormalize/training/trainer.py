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

import os
from typing import List

import numpy as np
import torch
from kerosene.config.trainers import RunConfiguration
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.distributed import on_single_device
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
        # Autoencoder.
        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            gen_loss = self._generator.compute_train_loss(gen_pred, target[IMAGE_TARGET])
            gen_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._generator)

            self._generator.step()
            self._generator.zero_grad()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred.detach(), target)
            disc_loss.backward()
            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._discriminator)

            self._discriminator.step()
            self._discriminator.zero_grad()

        if self._should_activate_discriminator_loss():
            loss_D_G_X_as_X = self.evaluate_loss_D_G_X_as_X(self._discriminator, gen_pred)
            gen_loss = self._training_config.variables.lambda_ * loss_D_G_X_as_X
            gen_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._generator)

            self._generator.step()
            self._generator.zero_grad()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred, target)
            disc_loss.backward()

            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._discriminator)

            self._discriminator.step()
            self._discriminator.zero_grad()

        if self._should_activate_segmentation():
            seg_pred = self._segmenter.forward(gen_pred.detach())
            seg_loss = self._segmenter.compute_train_loss(seg_pred, target[IMAGE_TARGET])
            seg_loss.backward(retain_graph=True)
            self._generator.optimizer.reset()
            self._discriminator.optimizer.reset()
            self._segmenter.optimizer.reset()

            loss_D_G_X_as_X = self.evaluate_loss_D_G_X_as_X(self._discriminator, gen_pred)
            gen_loss = self._training_config.variables.lambda_ * loss_D_G_X_as_X
            gen_loss.backward()
            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._segmenter)
                self.average_gradients(self._generator)
            self._generator.step()
            self._segmenter.step()
            self._generator.zero_grad()
            self._segmenter.zero_grad()

            disc_loss, disc_pred = self.train_discriminator(inputs, gen_pred, target)
            disc_loss.backward()
            if not on_single_device(self._run_config.devices):
                self.average_gradients(self._discriminator)
            self._discriminator.step()
            self._discriminator.zero_grad()

        if disc_pred is not None:
            count = self.count(torch.argmax(disc_pred, dim=1), 3)

        if self.current_train_step % 100 == 0:
            self._update_plots(inputs, gen_pred)

        self.custom_variables["Pie Plot"] = count if count is not None else torch.Tensor().new_zeros((3,),
                                                                                                     dtype=torch.int8,
                                                                                                     device="cpu")

    def validate_step(self, inputs, target):
        gen_pred = self._generator.forward(inputs)

        if self._should_activate_autoencoder():
            self._generator.compute_valid_loss(gen_pred, target[IMAGE_TARGET])
            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID])

        if self._should_activate_discriminator_loss():
            self.validate_discriminator(inputs, gen_pred, target[DATASET_ID])

        if self._should_activate_segmentation():
            seg_pred = self._segmenter.forward(gen_pred.detach())
            self._segmenter.compute_valid_loss(seg_pred, target[IMAGE_TARGET])

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

    @staticmethod
    def _reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= int(os.environ['WORLD_SIZE'])
        return rt

    def _should_activate_discriminator_loss(self):
        return self._current_epoch >= self._patience_discriminator

    def _should_activate_segmentation(self):
        return self._current_epoch >= self._patience_segmentation

    def _should_activate_autoencoder(self):
        return self._current_epoch < self._patience_discriminator

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

    def evaluate_loss_D_G_X_as_X(self, inputs, detach=False):
        pred_D_G_X = self._discriminator.forward(inputs, detach=detach)
        y = torch.Tensor().new_full(size=(inputs.size(0),), fill_value=2, dtype=torch.long, device=inputs.device,
                                    requires_grad=False)
        ones = torch.Tensor().new_ones(size=pred_D_G_X.size(), device=pred_D_G_X.device, dtype=pred_D_G_X.dtype)
        loss_D_G_X_as_X = self._discriminator.compute_train_loss(ones - pred_D_G_X, y)
        return loss_D_G_X_as_X

    def train_discriminator(self, inputs, gen_pred, target):
        merged_inputs = self.merge_tensors(inputs, gen_pred)
        pred = self._discriminator.forward(merged_inputs)
        y_bad = torch.Tensor().new_full(size=(target[DATASET_ID].size(0),), fill_value=2, dtype=torch.long,
                                        device=target[DATASET_ID].device, requires_grad=False)
        merged_targets = self.merge_tensors(target[DATASET_ID], y_bad)
        disc_loss = self._discriminator.compute_train_loss(pred, merged_targets)
        self._discriminator.compute_train_metric(pred, merged_targets)
        return disc_loss, pred

    def validate_discriminator(self, inputs, gen_pred, target):
        merged_inputs = self.merge_tensors(inputs, gen_pred)
        pred = self._discriminator.forward(merged_inputs)
        y_bad = torch.Tensor().new_full(size=(target[DATASET_ID].size(0),), fill_value=2, dtype=torch.long,
                                        device=target[DATASET_ID].device, requires_grad=False)
        merged_targets = self.merge_tensors(target[DATASET_ID], y_bad)
        disc_loss = self._discriminator.compute_valid_loss(pred, merged_targets)
        self._discriminator.compute_valid_metric(pred, merged_targets)
        return disc_loss, pred

    def train_step_(self, inputs, target):
        # Loss variable declaration.
        loss_D = torch.Tensor().new_zeros((1,), dtype=torch.float32).cuda()
        custom_loss = torch.Tensor().new_zeros((1,), dtype=torch.float32).cuda()
        loss_D_G_X_as_X = torch.Tensor().new_zeros((1,), dtype=torch.float32).cuda()
        loss_S_G_X = torch.Tensor().new_zeros((1,), dtype=torch.float32).cuda()
        mse_loss = torch.Tensor().new_zeros((1,), dtype=torch.float32).cuda()

        if self._with_autoencoder:
            mse_loss, generated_batch = self._generator_trainer.train_batch_as_autoencoder(batch)

            return loss_D, loss_D_G_X_as_X, custom_loss, loss_S_G_X, generated_batch, None, mse_loss

        if not self._with_segmentation:
            # Deactivate segmenter gradients.
            self._segmenter_trainer.disable_gradients()

            loss_D_G_X_as_X, generated_batch = self._generator_trainer.train_batch(batch)

            self._discriminator_trainer.reset_optimizer()

            loss_D, pred_D_X, pred_D_G_X = self._discriminator_trainer.train_batch(batch, generated_batch)

            count = self.count(torch.argmax(torch.cat((pred_D_X.x, pred_D_G_X.x), dim=0), dim=1), 3)

            self._class_pie_plot.update(sizes=count)
            self._discriminator_trainer.update_metric(torch.cat((pred_D_X.x, pred_D_G_X.x)),
                                                      torch.cat(
                                                          (pred_D_X.dataset_id.long(),
                                                           pred_D_G_X.dataset_id.long())))

            return loss_D, loss_D_G_X_as_X, custom_loss, loss_S_G_X, generated_batch, None, mse_loss

        else:
            self._generator_trainer.disable_gradients()
            self._discriminator_trainer.disable_gradients()
            self._segmenter_trainer.enable_gradients()

            generated_batch = self._generator_trainer.predict(batch, detach=True)
            loss_S_G_X, segmented_batch = self._segmenter_trainer.train_batch(generated_batch)

            self._generator_trainer.enable_gradients()
            self._discriminator_trainer.enable_gradients()
            self._segmenter_trainer.disable_gradients()

            self._generator_trainer.reset_optimizer()
            self._discriminator_trainer.reset_optimizer()
            self._segmenter_trainer.reset_optimizer()

            custom_loss, loss_D_G_X_as_X = self._generator_trainer.train_batch_with_segmentation_loss(batch,
                                                                                                      loss_S_G_X)

            pred_D_X = self._discriminator_trainer.predict(batch)
            pred_D_G_X = self._discriminator_trainer.predict(generated_batch)

            self._discriminator_trainer.update_metric(torch.cat((pred_D_X.x, pred_D_G_X.x)),
                                                      torch.cat(
                                                          (pred_D_X.dataset_id.long(),
                                                           pred_D_G_X.dataset_id.long())))

            self._segmenter_trainer.update_metric(segmented_batch.x, torch.squeeze(batch.y, dim=1).long())

            return loss_D, loss_D_G_X_as_X, custom_loss, loss_S_G_X, generated_batch, segmented_batch, mse_loss

    def _validate_epoch_(self, epoch_num: int):
        self._at_validation_begin()

        if self._running_config.local_rank == 0:
            self.LOGGER.info("Validating...")

        with torch.no_grad():
            for i, (batch_d0, batch_d1) in enumerate(
                    zip(self._dataloaders[2], self._dataloaders[3]), 0):
                batch = concat_batches(batch_d0, batch_d1)

                batch = self._prepare_batch(batch=batch,
                                            input_device=torch.device('cpu'),
                                            output_device=self._running_config.device)

                generated_batch = self._generator_trainer.predict(batch)

                if self._with_autoencoder:
                    mse_loss, generated_batch = self._generator_trainer.validate_batch_as_autoencoder(batch)

                    self._generator_trainer.update_loss_gauge(mse_loss.item(), batch.x.size(0), phase="validation")

                elif self._with_segmentation:
                    loss_S_G_X, segmented_batch = self._segmenter_trainer.validate_batch(batch)

                    self._segmenter_trainer.validation_loss_gauge.update(loss_S_G_X.item(),
                                                                         batch.x.size(0))
                    self._segmenter_trainer.update_metric(segmented_batch.x, batch.y)
                    self._segmenter_trainer.update_metric_gauge(self._segmenter_trainer.compute_metric(),
                                                                batch.x.size(0), phase="validation")

                if not self._with_autoencoder:
                    loss_D, pred_D_X, pred_D_G_X = self._discriminator_trainer.validate_batch(batch, generated_batch)
                    loss_D_G_X_as_X = self._generator_trainer.evaluate_loss_D_G_X_as_X(
                        generated_batch)

                    self._discriminator_trainer.validation_loss_gauge.update(loss_D.item(),
                                                                             batch.x.size(0))
                    self._discriminator_trainer.update_metric((torch.cat((pred_D_X.x, pred_D_G_X.x))),
                                                              torch.cat(
                                                                  (pred_D_X.dataset_id.long(),
                                                                   pred_D_G_X.dataset_id.long())))
                    self._discriminator_trainer.update_metric_gauge(self._discriminator_trainer.compute_metric(),
                                                                    batch.x.size(0))
                    self._generator_trainer.validation_loss_gauge.update(loss_D_G_X_as_X.item(), batch.x.size(0))

            if self._running_config.local_rank == 0:
                self.LOGGER.info(
                    "Validation Epoch: {} Generator loss: {} Discriminator loss: {} Segmenter loss: {} Segmenter Dice Score: {}".format(
                        epoch_num,
                        self._generator_trainer.validation_loss_gauge.average,
                        self._discriminator_trainer.validation_loss_gauge.average,
                        self._segmenter_trainer.validation_loss_gauge.average,
                        self._segmenter_trainer.validation_metric_gauge.average))

        del batch
        del generated_batch
        if not self._with_autoencoder:
            del loss_D
            del loss_D_G_X_as_X
            del pred_D_X
        if self._with_segmentation:
            del segmented_batch
            del loss_S_G_X

        self._at_validation_end()
