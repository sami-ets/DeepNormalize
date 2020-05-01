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
import pynvml
import torch
from ignite.metrics.confusion_matrix import ConfusionMatrix
from kerosene.configs.configs import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from kerosene.nn.functional import js_div
from kerosene.training.trainers import ModelTrainer
from kerosene.training.trainers import Trainer
from kerosene.utils.tensors import flatten, to_onehot
from torch.utils.data import DataLoader, Dataset

from deepNormalize.inputs.datasets import SliceDataset
from deepNormalize.inputs.images import SliceType
from deepNormalize.metrics.metrics import mean_hausdorff_distance
from deepNormalize.training.sampler import Sampler
from deepNormalize.utils.constants import GENERATOR, SEGMENTER, IMAGE_TARGET, DATASET_ID, ABIDE_ID, \
    NON_AUGMENTED_INPUTS, AUGMENTED_INPUTS, NON_AUGMENTED_TARGETS, AUGMENTED_TARGETS
from deepNormalize.utils.constants import ISEG_ID, MRBRAINS_ID
from deepNormalize.utils.image_slicer import ImageSlicer, SegmentationSlicer, LabelMapper
from deepNormalize.utils.utils import construct_double_histrogram, construct_single_histogram, \
    construct_triple_histrogram, construct_class_histogram, get_all_patches, rebuild_augmented_images, \
    save_augmented_rebuilt_images, rebuild_image, save_rebuilt_image
from deepNormalize.utils.utils import to_html, to_html_per_dataset, to_html_JS, to_html_time

pynvml.nvmlInit()


class DualUNetTrainer(Trainer):

    def __init__(self, training_config, model_trainers: List[ModelTrainer],
                 train_data_loader: DataLoader, valid_data_loader: DataLoader, test_data_loader: DataLoader,
                 reconstruction_datasets: List[Dataset], normalize_reconstructors: list, input_reconstructors: list,
                 segmentation_reconstructors: list, augmented_reconstructors: list, gt_reconstructors: list,
                 run_config: RunConfiguration, dataset_config: dict, save_folder: str):
        super(DualUNetTrainer, self).__init__("DualUNetTrainer", train_data_loader, valid_data_loader,
                                              test_data_loader, model_trainers, run_config)
        self._training_config = training_config
        self._run_config = run_config
        self._dataset_configs = dataset_config
        self._patience_segmentation = training_config.patience_segmentation
        self._slicer = ImageSlicer()
        self._seg_slicer = SegmentationSlicer()
        self._label_mapper = LabelMapper()
        self._reconstruction_datasets = reconstruction_datasets
        self._normalize_reconstructors = normalize_reconstructors
        self._input_reconstructors = input_reconstructors
        self._gt_reconstructors = gt_reconstructors
        self._segmentation_reconstructors = segmentation_reconstructors
        self._augmented_reconstructors = augmented_reconstructors
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
        self._js_div_inputs_gauge = AverageGauge()
        self._js_div_gen_gauge = AverageGauge()
        self._general_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._iSEG_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._MRBrainS_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._ABIDE_confusion_matrix_gauge = ConfusionMatrix(num_classes=4)
        self._previous_mean_dice = 0.0
        self._previous_per_dataset_table = ""
        self._start_time = time.time()
        self._save_folder = save_folder
        self._sampler = Sampler(0.3)
        self._is_sliced = True if isinstance(self._reconstruction_datasets[0], SliceDataset) else False
        print("Total number of parameters: {}".format(
            sum(p.numel() for p in self._model_trainers[GENERATOR].parameters()) +
            sum(p.numel() for p in self._model_trainers[SEGMENTER].parameters())))
        pynvml.nvmlInit()

    def _train_g(self, G: ModelTrainer, real, backward=True):
        G.zero_grad()

        gen_pred = torch.nn.functional.relu(G.forward(real))

        loss_G = G.compute_and_update_train_loss("MSELoss", gen_pred, real)

        metric = G.compute_metric("MeanSquaredError", gen_pred, real)
        G.update_train_metric("MeanSquaredError", metric / 32768)

        if backward:
            loss_G.backward()
            G.step()

        return gen_pred

    def _valid_g(self, G: ModelTrainer, real):
        gen_pred = torch.nn.functional.relu(G.forward(real))

        G.compute_and_update_valid_loss("MSELoss", gen_pred, real)

        metric = G.compute_metric("MeanSquaredError", gen_pred, real)
        G.update_valid_metric("MeanSquaredError", metric / 32768)

        return gen_pred

    def _test_g(self, G: ModelTrainer, real):
        gen_pred = torch.nn.functional.relu(G.forward(real))

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

    def train_step(self, inputs, target):
        inputs, target = self._sampler(inputs, target)

        if self._should_activate_autoencoder():
            gen_pred = self._train_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            seg_pred, _ = self._train_s(self._model_trainers[SEGMENTER], inputs[NON_AUGMENTED_INPUTS],
                                        target[NON_AUGMENTED_TARGETS][IMAGE_TARGET])

            if self.current_train_step % 500 == 0:
                self._update_image_plots(self.phase, inputs[NON_AUGMENTED_INPUTS].cpu().detach(),
                                         gen_pred.cpu().detach(),
                                         seg_pred.cpu().detach(),
                                         target[NON_AUGMENTED_TARGETS][IMAGE_TARGET].cpu().detach(),
                                         target[NON_AUGMENTED_TARGETS][DATASET_ID].cpu().detach())

        if self._should_activate_segmentation():
            gen_pred = self._train_g(self._model_trainers[GENERATOR], inputs[AUGMENTED_INPUTS], backward=False)

            seg_pred, loss_S = self._train_s(self._model_trainers[SEGMENTER], gen_pred,
                                             target[NON_AUGMENTED_TARGETS][IMAGE_TARGET], backward=False)

            loss_S.mean().backward()

            self._model_trainers[SEGMENTER].step()
            self._model_trainers[GENERATOR].step()

            if self.current_train_step % 500 == 0:
                self._update_image_plots(self.phase, inputs[AUGMENTED_INPUTS].cpu().detach(),
                                         gen_pred.cpu().detach(),
                                         seg_pred.cpu().detach(),
                                         target[AUGMENTED_TARGETS][IMAGE_TARGET].cpu().detach(),
                                         target[AUGMENTED_TARGETS][DATASET_ID].cpu().detach())

    def validate_step(self, inputs, target):
        if self._should_activate_autoencoder():
            gen_pred = self._valid_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            seg_pred, _ = self._valid_s(self._model_trainers[SEGMENTER], inputs[NON_AUGMENTED_INPUTS],
                                        target[IMAGE_TARGET])

        if self._should_activate_segmentation():
            gen_pred = self._valid_g(self._model_trainers[GENERATOR], inputs[AUGMENTED_INPUTS])

            seg_pred, _ = self._valid_s(self._model_trainers[SEGMENTER], gen_pred, target[IMAGE_TARGET])

        if self.current_valid_step % 100 == 0:
            self._update_image_plots(self.phase, inputs[NON_AUGMENTED_INPUTS].cpu().detach(),
                                     gen_pred.cpu().detach(),
                                     seg_pred.cpu().detach(),
                                     target[IMAGE_TARGET].cpu().detach(),
                                     target[DATASET_ID].cpu().detach())

    def test_step(self, inputs, target):
        if self._should_activate_autoencoder():
            gen_pred = self._test_g(self._model_trainers[GENERATOR], inputs[NON_AUGMENTED_INPUTS])

            seg_pred, _ = self._test_s(self._model_trainers[SEGMENTER], inputs[NON_AUGMENTED_INPUTS],
                                       target[IMAGE_TARGET], self._class_dice_gauge_on_patches)

        if self._should_activate_segmentation():
            gen_pred = self._test_g(self._model_trainers[GENERATOR], inputs[AUGMENTED_INPUTS])

            seg_pred, loss_S = self._test_s(self._model_trainers[SEGMENTER], gen_pred, target[IMAGE_TARGET],
                                            self._class_dice_gauge_on_patches)

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

            self._class_hausdorff_distance_gauge.update(
                mean_hausdorff_distance(
                    to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4),
                    to_onehot(torch.squeeze(target[IMAGE_TARGET], dim=1).long(), num_classes=4))[-3:])

            self._general_confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                          num_classes=4),
                torch.squeeze(target[IMAGE_TARGET].long(), dim=1)))

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

        if self.current_test_step % 100 == 0:
            self._update_histograms(inputs[NON_AUGMENTED_INPUTS], target, gen_pred)
            self._update_image_plots(self.phase, inputs[NON_AUGMENTED_INPUTS].cpu().detach(),
                                     gen_pred.cpu().detach(),
                                     seg_pred.cpu().detach(),
                                     target[IMAGE_TARGET].cpu().detach(),
                                     target[DATASET_ID].cpu().detach())

    def scheduler_step(self):
        self._model_trainers[GENERATOR].scheduler_step()
        self._model_trainers[SEGMENTER].scheduler_step()

    def on_epoch_begin(self):
        self._class_hausdorff_distance_gauge.reset()
        self._mean_hausdorff_distance_gauge.reset()
        self._per_dataset_hausdorff_distance_gauge.reset()
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

        if self._current_epoch == self._training_config.patience_segmentation:
            self._model_trainers[GENERATOR].optimizer_lr = 0.001

    def on_test_epoch_end(self):
        if self.epoch % 20 == 0:
            all_patches, ground_truth_patches = get_all_patches(self._reconstruction_datasets, self._is_sliced)

            img_input = rebuild_image(self._dataset_configs.keys(), all_patches, self._input_reconstructors)
            img_gt = rebuild_image(self._dataset_configs.keys(), ground_truth_patches, self._gt_reconstructors)
            img_norm = rebuild_image(self._dataset_configs.keys(), all_patches, self._normalize_reconstructors)
            img_seg = rebuild_image(self._dataset_configs.keys(), all_patches, self._segmentation_reconstructors)

            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_input, "Input")
            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_gt,
                               "Ground_Truth")
            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_norm,
                               "Normalized")
            save_rebuilt_image(self._current_epoch, self._save_folder, self._dataset_configs.keys(), img_seg,
                               "Segmented")
            if self._training_config.build_augmented_images:
                img_augmented, augmented_minus_inputs, norm_minus_augmented = rebuild_augmented_images(
                    self._dataset_configs.keys(), all_patches, img_input, img_norm, self._augmented_reconstructors)

                save_augmented_rebuilt_images(self._current_epoch, self._save_folder, self._dataset_configs.keys(),
                                              img_augmented, augmented_minus_inputs, norm_minus_augmented)

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

                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg[dataset]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt[dataset]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_images.update(np.array(metric["Dice"]))

            if "iSEG" in img_seg:
                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg["iSEG"]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt["iSEG"]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_iseg_images.update(np.array(metric["Dice"]))
            else:
                self._class_dice_gauge_on_reconstructed_iseg_images.update(np.array([0.0, 0.0, 0.0]))
            if "MRBrainS" in img_seg:
                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg["MRBrainS"]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt["MRBrainS"]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_mrbrains_images.update(np.array(metric["Dice"]))
            else:
                self._class_dice_gauge_on_reconstructed_mrbrains_images.update(np.array([0.0, 0.0, 0.0]))
            if "ABIDE" in img_seg:
                metric = self._model_trainers[SEGMENTER].compute_metrics(
                    to_onehot(torch.tensor(img_seg["ABIDE"]).unsqueeze(0).long(), num_classes=4),
                    torch.tensor(img_gt["ABIDE"]).unsqueeze(0).long())
                self._class_dice_gauge_on_reconstructed_abide_images.update(np.array(metric["Dice"]))
            else:
                self._class_dice_gauge_on_reconstructed_abide_images.update(np.array([0.0, 0.0, 0.0]))

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
