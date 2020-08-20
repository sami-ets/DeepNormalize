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

import cv2
import numpy as np
import os
import pynvml
import torch
from ignite.metrics.confusion_matrix import ConfusionMatrix
from kerosene.configs.configs import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from kerosene.nn.functional import js_div
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.constants import CHECKPOINT_EXT
from kerosene.utils.files import should_create_dir
from kerosene.utils.tensors import to_onehot
from samitorch.utils.tensors import flatten
from torch.utils.data import DataLoader, Dataset

from deepNormalize.inputs.datasets import SliceDataset
from deepNormalize.inputs.images import SliceType
from deepNormalize.inputs.pools import ImagePool
from deepNormalize.metrics.metrics import mean_hausdorff_distance
from deepNormalize.training.sampler import Sampler
from deepNormalize.utils.constants import GENERATOR, SEGMENTER, DISCRIMINATOR, IMAGE_TARGET, DATASET_ID, ABIDE_ID, \
    NON_AUGMENTED_INPUTS, AUGMENTED_INPUTS, NON_AUGMENTED_TARGETS, AUGMENTED_TARGETS
from deepNormalize.utils.constants import ISEG_ID, MRBRAINS_ID
from deepNormalize.utils.image_slicer import ImageSlicer, SegmentationSlicer, LabelMapper, FeatureMapSlicer
from deepNormalize.utils.utils import to_html, to_html_per_dataset, to_html_JS, to_html_time, count, \
    construct_triple_histrogram, construct_double_histrogram, construct_single_histogram, construct_class_histogram, \
    get_all_patches, rebuild_augmented_images, save_augmented_rebuilt_images, \
    rebuild_image, save_rebuilt_image


class ResNetTrainerNewLoss(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, test_data_loader: DataLoader,
                 reconstruction_datasets: List[Dataset],
                 normalize_reconstructors: list, input_reconstructors: list, segmentation_reconstructors: list,
                 augmented_reconstructors: list, gt_reconstructors: list, run_config: RunConfiguration,
                 dataset_config: dict, save_folder: str):
        super(ResNetTrainerNewLoss, self).__init__("ResNetTrainer", train_data_loader, valid_data_loader,
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
        self._normalize_reconstructors = normalize_reconstructors
        self._input_reconstructors = input_reconstructors
        self._gt_reconstructors = gt_reconstructors
        self._segmentation_reconstructors = segmentation_reconstructors
        self._augmented_reconstructors = augmented_reconstructors
        self._num_real_datasets = len(input_reconstructors)
        self._num_datasets = self._num_real_datasets + 1
        self._fake_class_id = self._num_datasets - 1
        self._D_G_X_as_X_train_gauge = AverageGauge()
        self._D_G_X_as_X_valid_gauge = AverageGauge()
        self._D_G_X_as_X_test_gauge = AverageGauge()
        self._total_loss_train_gauge = AverageGauge()
        self._total_loss_valid_gauge = AverageGauge()
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
        self._class_dice_gauge_on_patches = AverageGauge()
        self._class_dice_gauge_on_reconstructed_images = AverageGauge()
        self._class_dice_gauge_on_reconstructed_iseg_images = AverageGauge()
        self._class_dice_gauge_on_reconstructed_mrbrains_images = AverageGauge()
        self._class_dice_gauge_on_reconstructed_abide_images = AverageGauge()
        self._hausdorff_distance_gauge_on_reconstructed_iseg_images = AverageGauge()
        self._hausdorff_distance_gauge_on_reconstructed_mrbrains_images = AverageGauge()
        self._hausdorff_distance_gauge_on_reconstructed_abide_images = AverageGauge()
        self._js_div_inputs_gauge = AverageGauge()
        self._js_div_gen_gauge = AverageGauge()
        self._general_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._generator_output_confusion_matrix_iseg = ConfusionMatrix(num_classes=2)
        self._generator_output_confusion_matrix_mrbrains = ConfusionMatrix(num_classes=2)
        self._iSEG_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._MRBrainS_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._ABIDE_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._discriminator_confusion_matrix_gauge = ConfusionMatrix(num_classes=self._num_datasets)
        self._discriminator_confusion_matrix_gauge_training = ConfusionMatrix(num_classes=self._num_datasets)
        self._previous_mean_dice = 0.0
        self._previous_per_dataset_table = ""
        self._start_time = time.time()
        self._save_folder = save_folder
        self._sampler = Sampler(0.50)
        self._discriminator_loss_train_gauge = AverageGauge()
        self._discriminator_loss_valid_gauge = AverageGauge()
        self._discriminator_loss_test_gauge = AverageGauge()
        self._fake_T1_pool = ImagePool()
        self._real_T1_pool = ImagePool()
        self._n_critics = training_config.n_critics
        self._is_sliced = True if isinstance(self._reconstruction_datasets[0], SliceDataset) else False
        print("Total number of parameters: {}".format(
            sum(p.numel() for p in self._model_trainers[SEGMENTER].parameters()) +
            sum(p.numel() for p in self._model_trainers[GENERATOR].parameters()) +
            sum(p.numel() for p in self._model_trainers[DISCRIMINATOR].parameters())))
        pynvml.nvmlInit()

    def _train_d(self, D: ModelTrainer, real, fake, target, loss_gauge: AverageGauge):
        D.zero_grad()

        # Forward on real data.
        pred_D_X, x_conv1, x_layer1, x_layer2, x_layer3 = D.forward(real)
        loss_D_real = D.compute_and_update_train_loss("Pred Real", torch.nn.functional.log_softmax(pred_D_X, dim=1),
                                                      target)

        # Forward on fake data.
        pred_D_G_X, _, _, _, _ = D.forward(fake.detach())
        # Forge bad class (K+1) tensor.
        y_fake = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=self._fake_class_id,
                                         dtype=torch.long, device=target.device, requires_grad=False)
        loss_D_fake = D.compute_and_update_train_loss("Pred Fake", torch.nn.functional.log_softmax(pred_D_G_X, dim=1),
                                                      y_fake)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real)
        loss_D.backward()
        loss_gauge.update(loss_D.item())

        # Forge bad class (K+1) tensor.
        pred = torch.cat((pred_D_X, pred_D_G_X), dim=0)
        target = torch.cat((target, y_fake), dim=0)

        metric = D.compute_metrics(pred, target)
        D.update_train_metrics(metric)

        D.step()

        return loss_D, pred, target, x_conv1, x_layer1, x_layer2, x_layer3

    def _valid_d(self, D: ModelTrainer, real, fake, target, loss_gauge: AverageGauge):
        # Forward on real data.
        pred_D_X, _, _, _, _ = D.forward(real)
        loss_D_real = D.compute_and_update_valid_loss("Pred Real", torch.nn.functional.log_softmax(pred_D_X, dim=1),
                                                      target)

        # Forward on fake data.
        pred_D_G_X, _, _, _, _ = D.forward(fake.detach())
        # Forge bad class (K+1) tensor.
        y_fake = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=self._fake_class_id,
                                         dtype=torch.long, device=target.device, requires_grad=False)
        loss_D_fake = D.compute_and_update_valid_loss("Pred Fake", torch.nn.functional.log_softmax(pred_D_G_X, dim=1),
                                                      y_fake)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) / 2
        loss_gauge.update(loss_D.item())

        pred = torch.cat((pred_D_X, pred_D_G_X), dim=0)
        target = torch.cat((target, y_fake), dim=0)

        metric = D.compute_metrics(pred, target)
        D.update_valid_metrics(metric)

    def _test_d(self, D: ModelTrainer, real, fake, target, loss_gauge: AverageGauge):
        # Forward on real data.
        pred_D_X, _, _, _, _ = D.forward(real)

        loss_D_real = D.compute_and_update_test_loss("Pred Real", torch.nn.functional.log_softmax(pred_D_X, dim=1),
                                                     target)

        # Forward on fake data.
        pred_D_G_X, _, _, _, _ = D.forward(fake.detach())
        # Forge bad class (K+1) tensor.
        y_fake = torch.Tensor().new_full(size=(pred_D_G_X.size(0),), fill_value=self._fake_class_id,
                                         dtype=torch.long, device=target.device, requires_grad=False)
        loss_D_fake = D.compute_and_update_test_loss("Pred Fake", torch.nn.functional.log_softmax(pred_D_G_X, dim=1),
                                                     y_fake)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real)
        loss_gauge.update(loss_D.item())

        pred = torch.cat((pred_D_X, pred_D_G_X), dim=0)
        target = torch.cat((target, y_fake), dim=0)

        metric = D.compute_metrics(pred, target)
        D.update_test_metrics(metric)

        return loss_D, pred, target

    def _train_g(self, G: ModelTrainer, real, backward=True):
        G.zero_grad()

        gen_pred = torch.nn.functional.sigmoid(G.forward(real))

        loss_G = G.compute_and_update_train_loss("MSELoss", gen_pred, real)

        metric = G.compute_metric("MeanSquaredError", gen_pred, real)
        G.update_train_metric("MeanSquaredError", metric / 32768)

        if backward:
            loss_G.backward()
            G.step()

        return gen_pred

    def _valid_g(self, G: ModelTrainer, real):
        gen_pred = torch.nn.functional.sigmoid(G.forward(real))

        G.compute_and_update_valid_loss("MSELoss", gen_pred, real)

        metric = G.compute_metric("MeanSquaredError", gen_pred, real)
        G.update_valid_metric("MeanSquaredError", metric / 32768)

        return gen_pred

    def _test_g(self, G: ModelTrainer, real):
        gen_pred = torch.nn.functional.sigmoid(G.forward(real))

        G.compute_and_update_test_loss("MSELoss", gen_pred, real)

        metric = G.compute_metric("MeanSquaredError", gen_pred, real)
        G.update_test_metric("MeanSquaredError", metric / 32768)

        return gen_pred

    def _train_s(self, S: ModelTrainer, inputs, target, backward=True):
        S.zero_grad()

        target_ohe = to_onehot(torch.squeeze(target, dim=1).long(), num_classes=4)
        target = torch.squeeze(target, dim=1).long()

        seg_pred = torch.nn.functional.softmax(S.forward(inputs), dim=1)

        loss_S = S.compute_loss("DiceLoss", seg_pred, target_ohe)
        S.update_train_loss("DiceLoss", loss_S.mean())

        metrics = S.compute_metrics(seg_pred, target)
        metrics["Dice"] = metrics["Dice"].mean()
        metrics["IoU"] = metrics["IoU"].mean()
        S.update_train_metrics(metrics)

        if backward:
            loss_S.mean().backward()
            S.step()

        return seg_pred, loss_S

    def _valid_s(self, S: ModelTrainer, inputs, target):
        target_ohe = to_onehot(torch.squeeze(target, dim=1).long(), num_classes=4)
        target = torch.squeeze(target, dim=1).long()

        seg_pred = torch.nn.functional.softmax(S.forward(inputs), dim=1)

        loss_S = S.compute_loss("DiceLoss", seg_pred, target_ohe)
        S.update_valid_loss("DiceLoss", loss_S.mean())

        metrics = S.compute_metrics(seg_pred, target)
        metrics["Dice"] = metrics["Dice"].mean()
        metrics["IoU"] = metrics["IoU"].mean()
        S.update_valid_metrics(metrics)

        return seg_pred, loss_S

    def _test_s(self, S: ModelTrainer, inputs, target, metric_gauge: AverageGauge):
        target_ohe = to_onehot(torch.squeeze(target, dim=1).long(), num_classes=4)
        target = torch.squeeze(target, dim=1).long()

        seg_pred = torch.nn.functional.softmax(S.forward(inputs), dim=1)

        loss_S = S.compute_loss("DiceLoss", seg_pred, target_ohe)
        S.update_test_loss("DiceLoss", loss_S.mean())

        metrics = S.compute_metrics(seg_pred, target)
        metric_gauge.update(np.array(metrics["Dice"]))
        metrics["Dice"] = metrics["Dice"].mean()
        metrics["IoU"] = metrics["IoU"].mean()
        S.update_test_metrics(metrics)

        return seg_pred, loss_S

    def _loss_D_G_X_as_X(self, D: ModelTrainer, generated, real_target, fake_target, loss_gauge: AverageGauge):
        pred_D_G_X, _, _, _, _ = D.forward(generated)
        ones = torch.Tensor().new_ones(size=pred_D_G_X.size(), device=pred_D_G_X.device, dtype=pred_D_G_X.dtype,
                                       requires_grad=False)
        term_1 = self._model_trainers[DISCRIMINATOR].compute_loss("Pred Real",
                                                                  torch.nn.functional.log_softmax(
                                                                      ones - torch.nn.functional.softmax(pred_D_G_X,
                                                                                                         dim=1),
                                                                      dim=1),
                                                                  fake_target)
        term_2 = self._model_trainers[DISCRIMINATOR].compute_loss("Pred Real",
                                                                  torch.nn.functional.log_softmax(
                                                                      ones - torch.nn.functional.softmax(pred_D_G_X,
                                                                                                         dim=1),
                                                                      dim=1),
                                                                  real_target)
        loss_D_G_X_as_X = term_1 + term_2
        loss_gauge.update(loss_D_G_X_as_X.item())

        return loss_D_G_X_as_X

    def train_step(self, inputs, target):
        inputs, target = self._sampler(inputs, target)

        disc_pred = None
        disc_target = None

        if self._should_activate_autoencoder():
            gen_pred = self._train_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            for iter_critic in range(self._n_critics):
                real_images, real_targets = self._real_T1_pool.query(
                    (inputs[NON_AUGMENTED_INPUTS], target[NON_AUGMENTED_TARGETS]))
                fake_images, _ = self._fake_T1_pool.query((gen_pred, target[NON_AUGMENTED_TARGETS]))

                disc_loss, disc_pred, disc_target, x_conv1, x_layer1, x_layer2, x_layer3 = self._train_d(
                    self._model_trainers[DISCRIMINATOR], real_images, fake_images, real_targets[DATASET_ID],
                    self._discriminator_loss_train_gauge)

            seg_pred, _ = self._train_s(self._model_trainers[SEGMENTER], inputs[NON_AUGMENTED_INPUTS],
                                        target[NON_AUGMENTED_TARGETS][IMAGE_TARGET])

            if self.current_train_step % 500 == 0:
                self._update_image_plots(self.phase, inputs[NON_AUGMENTED_INPUTS].cpu().detach(),
                                         gen_pred.cpu().detach(),
                                         seg_pred.cpu().detach(),
                                         target[NON_AUGMENTED_TARGETS][IMAGE_TARGET].cpu().detach(),
                                         target[NON_AUGMENTED_TARGETS][DATASET_ID].cpu().detach())

                self._make_disc_pie_plots(disc_pred, disc_target)

        if self._should_activate_segmentation():
            gen_pred = self._train_g(self._model_trainers[GENERATOR], inputs[AUGMENTED_INPUTS], backward=False)

            for iter_critic in range(self._n_critics):
                real_images, real_targets = self._real_T1_pool.query(
                    (inputs[NON_AUGMENTED_INPUTS], target[NON_AUGMENTED_TARGETS]))
                fake_images, _ = self._fake_T1_pool.query((gen_pred, target[AUGMENTED_TARGETS]))

                disc_loss, disc_pred, disc_target, x_conv1, x_layer1, x_layer2, x_layer3 = self._train_d(
                    self._model_trainers[DISCRIMINATOR], real_images, fake_images, real_targets[DATASET_ID],
                    self._discriminator_loss_train_gauge)

            seg_pred, loss_S = self._train_s(self._model_trainers[SEGMENTER], gen_pred,
                                             target[AUGMENTED_TARGETS][IMAGE_TARGET], backward=False)

            fake_target = torch.Tensor().new_full(fill_value=self._fake_class_id, size=(gen_pred.size(0),),
                                                  dtype=torch.long, device=inputs[AUGMENTED_INPUTS].device,
                                                  requires_grad=False)

            disc_loss_as_X = self._loss_D_G_X_as_X(self._model_trainers[DISCRIMINATOR], gen_pred,
                                                   target[NON_AUGMENTED_TARGETS][DATASET_ID],
                                                   fake_target, self._D_G_X_as_X_train_gauge)

            total_loss = self._training_config.variables["seg_ratio"] * loss_S.mean() + \
                         self._training_config.variables["disc_ratio"] * disc_loss_as_X
            self._total_loss_train_gauge.update(total_loss.item())

            total_loss.backward()

            self._model_trainers[SEGMENTER].step()
            self._model_trainers[GENERATOR].step()

            if self.current_train_step % 500 == 0:
                self._update_image_plots(self.phase, inputs[AUGMENTED_INPUTS].cpu().detach(),
                                         gen_pred.cpu().detach(),
                                         seg_pred.cpu().detach(),
                                         target[AUGMENTED_TARGETS][IMAGE_TARGET].cpu().detach(),
                                         target[AUGMENTED_TARGETS][DATASET_ID].cpu().detach())

                self._make_disc_pie_plots(disc_pred, disc_target)

        self._discriminator_confusion_matrix_gauge_training.update((
            to_onehot(torch.argmax(torch.nn.functional.softmax(disc_pred, dim=1), dim=1),
                      num_classes=self._num_datasets), disc_target))

        if self.current_train_step % 500 == 0:
            self.custom_variables["Conv1 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(
                torch.nn.functional.interpolate(x_conv1.cpu().detach(), scale_factor=5, mode="trilinear",
                                                align_corners=True).numpy()[0], 0))
        self.custom_variables["Layer1 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(
            torch.nn.functional.interpolate(x_layer1.cpu().detach(), scale_factor=10, mode="trilinear",
                                            align_corners=True).numpy()[0], 0))
        self.custom_variables["Layer2 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(
            torch.nn.functional.interpolate(x_layer2.cpu().detach(), scale_factor=20, mode="trilinear",
                                            align_corners=True).numpy()[0], 0))
        self.custom_variables["Layer3 FM"] = self._fm_slicer.get_colored_slice(SliceType.AXIAL, np.expand_dims(
            torch.nn.functional.interpolate(x_layer3.cpu().detach(), scale_factor=20, mode="trilinear",
                                            align_corners=True).numpy()[0], 0))

    def validate_step(self, inputs, target):
        if self._should_activate_autoencoder():
            gen_pred = self._valid_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            self._valid_d(
                self._model_trainers[DISCRIMINATOR], inputs[NON_AUGMENTED_INPUTS], gen_pred, target[DATASET_ID],
                self._discriminator_loss_valid_gauge)

            seg_pred, _ = self._valid_s(self._model_trainers[SEGMENTER], inputs[NON_AUGMENTED_INPUTS],
                                        target[IMAGE_TARGET])

        if self._should_activate_segmentation():
            gen_pred = self._valid_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            self._valid_d(self._model_trainers[DISCRIMINATOR], inputs[NON_AUGMENTED_INPUTS], gen_pred,
                          target[DATASET_ID], self._discriminator_loss_valid_gauge)

            seg_pred, loss_S = self._valid_s(self._model_trainers[SEGMENTER], gen_pred, target[IMAGE_TARGET])

            fake_target = torch.Tensor().new_full(fill_value=self._fake_class_id, size=(gen_pred.size(0),),
                                                  dtype=torch.long, device=inputs[NON_AUGMENTED_INPUTS].device,
                                                  requires_grad=False)
            disc_loss_as_X = self._loss_D_G_X_as_X(self._model_trainers[DISCRIMINATOR], gen_pred,
                                                   target[NON_AUGMENTED_TARGETS][DATASET_ID], fake_target,
                                                   self._D_G_X_as_X_valid_gauge)

            total_loss = self._training_config.variables["seg_ratio"] * loss_S.mean() + \
                         self._training_config.variables["disc_ratio"] * disc_loss_as_X
            self._total_loss_valid_gauge.update(total_loss.item())

        if self.current_valid_step % 100 == 0:
            self._update_image_plots(self.phase, inputs[NON_AUGMENTED_INPUTS].cpu().detach(),
                                     gen_pred.cpu().detach(),
                                     seg_pred.cpu().detach(),
                                     target[IMAGE_TARGET].cpu().detach(),
                                     target[DATASET_ID].cpu().detach())

    def test_step(self, inputs, target):
        if self._should_activate_autoencoder():
            gen_pred = self._test_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            _, disc_pred, disc_target, = self._test_d(
                self._model_trainers[DISCRIMINATOR], inputs[NON_AUGMENTED_INPUTS], gen_pred, target[DATASET_ID],
                self._discriminator_loss_test_gauge)

            seg_pred, _ = self._test_s(self._model_trainers[SEGMENTER], inputs[NON_AUGMENTED_INPUTS],
                                       target[IMAGE_TARGET], self._class_dice_gauge_on_patches)

        if self._should_activate_segmentation():
            gen_pred = self._test_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            _, disc_pred, disc_target = self._test_d(self._model_trainers[DISCRIMINATOR], inputs[NON_AUGMENTED_INPUTS],
                                                     gen_pred, target[DATASET_ID], self._discriminator_loss_test_gauge)

            seg_pred, loss_S = self._test_s(self._model_trainers[SEGMENTER], gen_pred, target[IMAGE_TARGET],
                                            self._class_dice_gauge_on_patches)

            fake_target = torch.Tensor().new_full(fill_value=self._fake_class_id, size=(gen_pred.size(0),),
                                                  dtype=torch.long, device=inputs[NON_AUGMENTED_INPUTS].device,
                                                  requires_grad=False)
            disc_loss_as_X = self._loss_D_G_X_as_X(self._model_trainers[DISCRIMINATOR], gen_pred,
                                                   target[NON_AUGMENTED_TARGETS][DATASET_ID],
                                                   fake_target,
                                                   self._D_G_X_as_X_valid_gauge)

            total_loss = self._training_config.variables["seg_ratio"] * loss_S.mean() + \
                         self._training_config.variables["disc_ratio"] * disc_loss_as_X
            self._total_loss_test_gauge.update(total_loss.item())

            if seg_pred[torch.where(target[DATASET_ID] == ISEG_ID)].shape[0] != 0:
                self._iSEG_dice_gauge.update(np.array(self._model_trainers[SEGMENTER].compute_metrics(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ISEG_ID)], dim=1),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG_ID)],
                                  dim=1).long())["Dice"].numpy()))

                self._iSEG_hausdorff_gauge.update(mean_hausdorff_distance(
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
                self._MRBrainS_dice_gauge.update(np.array(self._model_trainers[SEGMENTER].compute_metrics(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)], dim=1),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBRAINS_ID)],
                                  dim=1).long())["Dice"].numpy()))

                self._MRBrainS_hausdorff_gauge.update(mean_hausdorff_distance(
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
            else:
                self._MRBrainS_dice_gauge.update(np.zeros((3,)))
                self._MRBrainS_hausdorff_gauge.update(np.zeros((3,)))

            if seg_pred[torch.where(target[DATASET_ID] == ABIDE_ID)].shape[0] != 0:
                self._ABIDE_dice_gauge.update(np.array(self._model_trainers[SEGMENTER].compute_metrics(
                    torch.nn.functional.softmax(seg_pred[torch.where(target[DATASET_ID] == ABIDE_ID)], dim=1),
                    torch.squeeze(target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ABIDE_ID)],
                                  dim=1).long())["Dice"].numpy()))

                self._ABIDE_hausdorff_gauge.update(mean_hausdorff_distance(
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
            else:
                self._ABIDE_dice_gauge.update(np.zeros((3,)))
                self._ABIDE_hausdorff_gauge.update(np.zeros((3,)))

            self._class_hausdorff_distance_gauge.update(
                mean_hausdorff_distance(
                    to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4),
                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(), num_classes=4))[-3:])

            self._general_confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                          num_classes=4),
                torch.squeeze(target[IMAGE_TARGET].long(), dim=1)))

            c, d, h, w = inputs[AUGMENTED_INPUTS].shape[1], inputs[AUGMENTED_INPUTS].shape[2], \
                         inputs[AUGMENTED_INPUTS].shape[3], inputs[AUGMENTED_INPUTS].shape[4]

            hist_inputs = torch.cat(
                [torch.histc(inputs[AUGMENTED_INPUTS][i].view(1, c * d * h * w), bins=256, min=0, max=1).unsqueeze(0)
                 for i in range(inputs[0].shape[0])]).unsqueeze(0)
            hist_inputs = hist_inputs / (c * d * h * w)
            hist_inputs = torch.nn.Softmax(dim=2)(hist_inputs)

            hist_gen = torch.cat(
                [torch.histc(gen_pred[i].view(1, c * d * h * w), bins=256, min=0, max=1).unsqueeze(0)
                 for i in range(gen_pred.shape[0])]).unsqueeze(0)
            hist_gen = hist_gen / (c * d * h * w)
            hist_gen = torch.nn.Softmax(dim=2)(hist_gen)

            self._js_div_inputs_gauge.update(js_div(hist_inputs).item())
            self._js_div_gen_gauge.update(js_div(hist_gen).item())

        self._discriminator_confusion_matrix_gauge.update((
            to_onehot(torch.argmax(torch.nn.functional.softmax(disc_pred, dim=1), dim=1),
                      num_classes=self._num_datasets),
            disc_target))

        if self.current_test_step % 100 == 0:
            self._update_histograms(inputs[NON_AUGMENTED_INPUTS], target, gen_pred)
            self._update_image_plots(self.phase, inputs[NON_AUGMENTED_INPUTS].cpu().detach(),
                                     gen_pred.cpu().detach(),
                                     seg_pred.cpu().detach(),
                                     target[IMAGE_TARGET].cpu().detach(),
                                     target[DATASET_ID].cpu().detach())

    def scheduler_step(self):
        self._model_trainers[GENERATOR].scheduler_step()
        self._model_trainers[DISCRIMINATOR].scheduler_step()
        self._model_trainers[SEGMENTER].scheduler_step()

    def on_epoch_begin(self):
        self._D_G_X_as_X_train_gauge.reset()
        self._D_G_X_as_X_valid_gauge.reset()
        self._D_G_X_as_X_test_gauge.reset()
        self._total_loss_train_gauge.reset()
        self._total_loss_valid_gauge.reset()
        self._total_loss_test_gauge.reset()
        self._class_hausdorff_distance_gauge.reset()
        self._mean_hausdorff_distance_gauge.reset()
        self._iSEG_dice_gauge.reset()
        self._MRBrainS_dice_gauge.reset()
        self._ABIDE_dice_gauge.reset()
        self._iSEG_hausdorff_gauge.reset()
        self._MRBrainS_hausdorff_gauge.reset()
        self._ABIDE_hausdorff_gauge.reset()
        self._class_dice_gauge_on_patches.reset()
        self._js_div_inputs_gauge.reset()
        self._js_div_gen_gauge.reset()
        self._general_confusion_matrix_gauge.reset()
        self._iSEG_confusion_matrix_gauge.reset()
        self._MRBrainS_confusion_matrix_gauge.reset()
        self._ABIDE_confusion_matrix_gauge.reset()
        self._discriminator_confusion_matrix_gauge.reset()
        self._discriminator_confusion_matrix_gauge_training.reset()
        self._discriminator_loss_train_gauge.reset()
        self._discriminator_loss_valid_gauge.reset()
        self._discriminator_loss_test_gauge.reset()

        if self._current_epoch == self._training_config.patience_segmentation:
            self._model_trainers[GENERATOR].optimizer_lr = 0.001

    def on_train_epoch_end(self):
        self.custom_variables["D(G(X)) | X"] = [self._D_G_X_as_X_train_gauge.compute()]
        self.custom_variables["Discriminator Loss"] = [self._discriminator_loss_train_gauge.compute()]
        self.custom_variables["Total Loss"] = [self._total_loss_train_gauge.compute()]

        if self._discriminator_confusion_matrix_gauge_training._num_examples != 0:
            self.custom_variables["Discriminator Confusion Matrix Training"] = np.array(
                np.fliplr(self._discriminator_confusion_matrix_gauge_training.compute().cpu().detach().numpy()))
        else:
            self.custom_variables["Discriminator Confusion Matrix Training"] = np.zeros(
                (self._num_datasets, self._num_datasets))

        self._save(str(self.epoch), "Generator", self._model_trainers[GENERATOR].model_state,
                   self._model_trainers[GENERATOR].optimizer_state, self._save_folder)
        self._save(str(self.epoch), "Discriminator", self._model_trainers[DISCRIMINATOR].model_state,
                   self._model_trainers[GENERATOR].optimizer_state, self._save_folder)
        self._save(str(self.epoch), "Segmenter", self._model_trainers[SEGMENTER].model_state,
                   self._model_trainers[GENERATOR].optimizer_state, self._save_folder)

    def on_valid_epoch_end(self):
        self.custom_variables["D(G(X)) | X"] = [self._D_G_X_as_X_valid_gauge.compute()]
        self.custom_variables["Discriminator Loss"] = [self._discriminator_loss_valid_gauge.compute()]
        self.custom_variables["Total Loss"] = [self._total_loss_valid_gauge.compute()]

    def on_test_epoch_end(self):
        if self.epoch % 10 == 0:
            self._per_dataset_hausdorff_distance_gauge.reset()
            self._class_dice_gauge_on_reconstructed_iseg_images.reset()
            self._class_dice_gauge_on_reconstructed_mrbrains_images.reset()
            self._class_dice_gauge_on_reconstructed_abide_images.reset()
            self._hausdorff_distance_gauge_on_reconstructed_iseg_images.reset()
            self._hausdorff_distance_gauge_on_reconstructed_mrbrains_images.reset()
            self._hausdorff_distance_gauge_on_reconstructed_abide_images.reset()

            all_patches, ground_truth_patches = get_all_patches(self._reconstruction_datasets, self._is_sliced)

            img_input = rebuild_image(self._dataset_configs.keys(), all_patches, self._input_reconstructors)
            img_gt = rebuild_image(self._dataset_configs.keys(), ground_truth_patches, self._gt_reconstructors)
            img_norm = rebuild_image(self._dataset_configs.keys(), all_patches, self._normalize_reconstructors)
            img_seg = rebuild_image(self._dataset_configs.keys(), all_patches, self._segmentation_reconstructors)

            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_input,
                               "Input")
            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_gt,
                               "Ground_Truth")
            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_norm,
                               "Normalized")
            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_seg,
                               "Segmented")

            if self._training_config.build_augmented_images:
                img_augmented = rebuild_image(self._dataset_configs.keys(), all_patches, self._augmented_reconstructors)
                augmented_minus_inputs, normalized_minus_inputs = rebuild_augmented_images(img_augmented, img_input,
                                                                                           img_gt, img_norm, img_seg)

                save_augmented_rebuilt_images(self._current_epoch, self._save_folder, self._dataset_configs.keys(),
                                              img_augmented, augmented_minus_inputs, normalized_minus_inputs)

            mean_mhd = []
            for dataset in self._dataset_configs.keys():
                self.custom_variables[
                    "Reconstructed Normalized {} Image".format(dataset)] = self._slicer.get_slice(
                    SliceType.AXIAL, np.expand_dims(np.expand_dims(img_norm[dataset], 0), 0), 160)
                self.custom_variables[
                    "Reconstructed Segmented {} Image".format(dataset)] = self._seg_slicer.get_colored_slice(
                    SliceType.AXIAL, np.expand_dims(np.expand_dims(img_seg[dataset], 0), 0), 160).squeeze(0)
                self.custom_variables[
                    "Reconstructed Ground Truth {} Image".format(dataset)] = self._seg_slicer.get_colored_slice(
                    SliceType.AXIAL, np.expand_dims(np.expand_dims(img_gt[dataset], 0), 0), 160).squeeze(0)
                self.custom_variables[
                    "Reconstructed Input {} Image".format(dataset)] = self._slicer.get_slice(
                    SliceType.AXIAL, np.expand_dims(np.expand_dims(img_input[dataset], 0), 0), 160)

                if self._training_config.build_augmented_images:
                    self.custom_variables[
                        "Reconstructed Augmented Input {} Image".format(dataset)] = self._slicer.get_slice(
                        SliceType.AXIAL, np.expand_dims(np.expand_dims(img_augmented[dataset], 0), 0), 160)
                    self.custom_variables[
                        "Reconstructed Initial Noise {} Image".format(
                            dataset)] = self._seg_slicer.get_colored_slice(
                        SliceType.AXIAL,
                        np.expand_dims(np.expand_dims(augmented_minus_inputs[dataset], 0), 0), 160).squeeze(0)
                    self.custom_variables[
                        "Reconstructed Noise {} After Normalization".format(
                            dataset)] = self._seg_slicer.get_colored_slice(
                        SliceType.AXIAL,
                        np.expand_dims(np.expand_dims(normalized_minus_inputs[dataset], 0), 0), 160).squeeze(0)
                else:
                    self.custom_variables["Reconstructed Augmented Input {} Image".format(
                        dataset)] = np.zeros((224, 192))
                    self.custom_variables[
                        "Reconstructed Initial Noise {} Image".format(
                            dataset)] = np.zeros((224, 192))
                    self.custom_variables[
                        "Reconstructed Noise {} After Normalization".format(
                            dataset)] = np.zeros((224, 192))

                mean_mhd.append(mean_hausdorff_distance(
                    to_onehot(torch.tensor(img_gt[dataset], dtype=torch.long), num_classes=4),
                    to_onehot(torch.tensor(img_seg[dataset], dtype=torch.long), num_classes=4))[-3:].mean())

                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg[dataset]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt[dataset]).unsqueeze(0).long())

                self._class_dice_gauge_on_reconstructed_images.update(np.array(metric["Dice"]))

            self._per_dataset_hausdorff_distance_gauge.update(np.array(mean_mhd))

            if "iSEG" in img_seg:
                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg["iSEG"]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt["iSEG"]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_iseg_images.update(np.array(metric["Dice"]))
                self._hausdorff_distance_gauge_on_reconstructed_iseg_images.update(mean_hausdorff_distance(
                    to_onehot(torch.tensor(img_gt["iSEG"], dtype=torch.long), num_classes=4),
                    to_onehot(torch.tensor(img_seg["iSEG"], dtype=torch.long), num_classes=4))[-3:])
            else:
                self._class_dice_gauge_on_reconstructed_iseg_images.update(np.array([0.0, 0.0, 0.0]))
                self._hausdorff_distance_gauge_on_reconstructed_iseg_images.update(np.array([0.0, 0.0, 0.0]))
            if "MRBrainS" in img_seg:
                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg["MRBrainS"]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt["MRBrainS"]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_mrbrains_images.update(np.array(metric["Dice"]))
                self._hausdorff_distance_gauge_on_reconstructed_mrbrains_images.update(mean_hausdorff_distance(
                    to_onehot(torch.tensor(img_gt["MRBrainS"], dtype=torch.long), num_classes=4),
                    to_onehot(torch.tensor(img_seg["MRBrainS"], dtype=torch.long), num_classes=4))[-3:])
            else:
                self._class_dice_gauge_on_reconstructed_mrbrains_images.update(np.array([0.0, 0.0, 0.0]))
                self._hausdorff_distance_gauge_on_reconstructed_mrbrains_images.update(np.array([0.0, 0.0, 0.0]))
            if "ABIDE" in img_seg:
                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg["ABIDE"]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt["ABIDE"]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_abide_images.update(np.array(metric["Dice"]))
                self._hausdorff_distance_gauge_on_reconstructed_abide_images.update(mean_hausdorff_distance(
                    to_onehot(torch.tensor(img_gt["ABIDE"], dtype=torch.long), num_classes=4),
                    to_onehot(torch.tensor(img_seg["ABIDE"], dtype=torch.long), num_classes=4))[-3:])
            else:
                self._class_dice_gauge_on_reconstructed_abide_images.update(np.array([0.0, 0.0, 0.0]))
                self._hausdorff_distance_gauge_on_reconstructed_abide_images.update(np.array([0.0, 0.0, 0.0]))

            if len(img_input) == 3:
                self.custom_variables["Reconstructed Images Histograms"] = cv2.imread(
                    construct_triple_histrogram(img_norm["iSEG"],
                                                img_input["iSEG"],
                                                img_norm["MRBrainS"],
                                                img_input["MRBrainS"],
                                                img_norm["ABIDE"],
                                                img_input["ABIDE"])).transpose((2, 0, 1))
            elif len(img_input) == 2:
                self.custom_variables["Reconstructed Images Histograms"] = cv2.imread(
                    construct_double_histrogram(img_norm["iSEG"],
                                                img_input["iSEG"],
                                                img_norm["MRBrainS"],
                                                img_input["MRBrainS"])).transpose((2, 0, 1))
            elif len(img_input) == 1:
                self.custom_variables["Reconstructed Images Histograms"] = cv2.imread(
                    construct_single_histogram(img_norm[list(self._dataset_configs.keys())[0]],
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
            self.custom_variables["Reconstructed Ground Truth iSEG Image"] = np.zeros((224, 192))
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
                (self._num_datasets, self._num_datasets))

        self.custom_variables["Metric Table"] = to_html(["CSF", "Grey Matter", "White Matter"],
                                                        ["DSC", "HD"],
                                                        [
                                                            self._class_dice_gauge_on_patches.compute() if self._class_dice_gauge_on_patches.has_been_updated() else np.array(
                                                                [0.0, 0.0, 0.0]),
                                                            self._class_hausdorff_distance_gauge.compute() if self._class_hausdorff_distance_gauge.has_been_updated() else np.array(
                                                                [0.0, 0.0, 0.0])
                                                        ])

        self.custom_variables[
            "Dice score per class per epoch"] = self._class_dice_gauge_on_patches.compute() if self._class_dice_gauge_on_patches.has_been_updated() else np.array(
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
        self.custom_variables[
            "Hausdorff Distance per class per epoch on reconstructed iSEG image"] = self._hausdorff_distance_gauge_on_reconstructed_iseg_images.compute() if self._hausdorff_distance_gauge_on_reconstructed_iseg_images.has_been_updated() else np.array(
            [0.0, 0.0, 0.0])
        self.custom_variables[
            "Hausdorff Distance per class per epoch on reconstructed MRBrainS image"] = self._hausdorff_distance_gauge_on_reconstructed_mrbrains_images.compute() if self._hausdorff_distance_gauge_on_reconstructed_mrbrains_images.has_been_updated() else np.array(
            [0.0, 0.0, 0.0])
        self.custom_variables[
            "Hausdorff Distance per class per epoch on reconstructed ABIDE image"] = self._hausdorff_distance_gauge_on_reconstructed_abide_images.compute() if self._hausdorff_distance_gauge_on_reconstructed_abide_images.has_been_updated() else np.array(
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
        self.custom_variables["Discriminator Loss"] = [self._discriminator_loss_test_gauge.compute()]
        self.custom_variables["Total Loss"] = [self._total_loss_test_gauge.compute()]
        self.custom_variables[
            "Per Dataset Mean Hausdorff Distance"] = self._per_dataset_hausdorff_distance_gauge.compute()

    def _update_image_plots(self, phase, inputs, generator_predictions, segmenter_predictions, target, dataset_ids):
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=5, mode="trilinear",
                                                 align_corners=True).numpy()
        generator_predictions = torch.nn.functional.interpolate(generator_predictions, scale_factor=5, mode="trilinear",
                                                                align_corners=True).numpy()
        segmenter_predictions = torch.nn.functional.interpolate(
            torch.argmax(torch.nn.functional.softmax(segmenter_predictions, dim=1), dim=1, keepdim=True).float(),
            scale_factor=5, mode="nearest").numpy()

        target = torch.nn.functional.interpolate(target.float(), scale_factor=5, mode="nearest").numpy()

        self.custom_variables[
            "{} Input Batch Process {}".format(phase, self._run_config.local_rank)] = self._slicer.get_slice(
            SliceType.AXIAL, inputs, inputs.shape[2] // 2)
        self.custom_variables[
            "{} Generated Batch Process {}".format(phase, self._run_config.local_rank)] = self._slicer.get_slice(
            SliceType.AXIAL, generator_predictions, generator_predictions.shape[2] // 2)
        self.custom_variables[
            "{} Segmented Batch Process {}".format(phase,
                                                   self._run_config.local_rank)] = self._seg_slicer.get_colored_slice(
            SliceType.AXIAL, segmenter_predictions, segmenter_predictions.shape[2] // 2)
        self.custom_variables[
            "{} Segmentation Ground Truth Batch Process {}".format(phase,
                                                                   self._run_config.local_rank)] = self._seg_slicer.get_colored_slice(
            SliceType.AXIAL, target, target.shape[2] // 2)
        self.custom_variables[
            "{} Label Map Batch Process {}".format(phase,
                                                   self._run_config.local_rank)] = self._label_mapper.get_label_map(
            dataset_ids)

    def _make_disc_pie_plots(self, disc_pred, target):
        count_ = count(torch.argmax(disc_pred.cpu().detach(), dim=1), self._num_datasets)
        real_count = count(target, self._num_datasets)
        self.custom_variables["Pie Plot"] = count_
        self.custom_variables["Pie Plot True"] = real_count

    def _should_activate_autoencoder(self):
        return self._current_epoch < self._patience_segmentation

    def _should_activate_segmentation(self):
        return self._current_epoch >= self._patience_segmentation

    def _update_histograms(self, inputs, target, gen_pred):
        self.custom_variables["Generated Intensity Histogram"] = flatten(gen_pred.cpu().detach())
        self.custom_variables["Input Intensity Histogram"] = flatten(inputs.cpu().detach())
        self.custom_variables["Per-Dataset Histograms"] = cv2.imread(
            construct_class_histogram(inputs, target, gen_pred)).transpose((2, 0, 1))
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

    @staticmethod
    def _save(epoch_num, model_name, model_state, optimizer_states, save_folder):
        if should_create_dir(save_folder, model_name):
            os.makedirs(os.path.join(save_folder, model_name))
        torch.save({"epoch_num": epoch_num,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer_states},
                   os.path.join(save_folder, model_name, "{}_{}{}".format(model_name, epoch_num, CHECKPOINT_EXT)))
