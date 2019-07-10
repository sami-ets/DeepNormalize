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

import torch
import torch.backends.cudnn as cudnn
import os

cudnn.benchmark = True
cudnn.enabled = True

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from samitorch.training.trainer import Trainer
from samitorch.utils.utils import to_onehot
from samitorch.metrics.gauges import RunningAverageGauge

from deepNormalize.utils.logger import Logger
from tests.models.model_helper_test import ModelHelper
from deepNormalize.training.model_trainer import ModelTrainer
from deepNormalize.adapters.ConfigAdapter import ConfigAdapter

X = 0
Y = 1


class DeepNormalizeTrainer(Trainer):
    REAL_LABEL = 1
    FAKE_LABEL = 0

    def __init__(self, config, callbacks):
        super(DeepNormalizeTrainer, self).__init__(config, callbacks)
        adapter = ConfigAdapter(config)

        preprocessor_config = adapter.adapt(model_position=0, criterion_position=0)
        segmenter_config = adapter.adapt(model_position=1, criterion_position=0)
        discriminator_config = adapter.adapt(model_position=2, criterion_position=1)

        self._preprocessor_trainer = ModelTrainer(preprocessor_config, callbacks, "Preprocessor")
        self._segmenter_trainer = ModelTrainer(segmenter_config, callbacks, "Segmenter")
        self._discriminator_trainer = ModelTrainer(discriminator_config, callbacks, "Discriminator")

        self._total_loss = RunningAverageGauge()
        self._D_X_n_loss = RunningAverageGauge()

        self._global_step = 0

        if self.config.running_config.local_rank == 0:
            self._logger = Logger(self.config.logger_config, "DeepNormalizeTrainer")

    def train(self):
        self._logger.info("Beginning training ...")
        for epoch in range(self.config.max_epoch):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch_num: int, **kwargs):
        for iteration, (batch_d0, batch_d1) in enumerate(zip(self.config.dataloader[0], self.config.dataloader[1]), 0):
            batch = self._concatenate_data(batch_d0, batch_d1)

            self._at_iteration_begin()

            current_batch = self._prepare_batch(batch={"X": batch[X], "y": batch[Y]}, input_device=torch.device('cpu'),
                                                output_device=self.config.running_config.device)

            # Generate normalized data and DO detach (so gradients ARE NOT calculated for generator and segmenter).
            X_n = self._preprocessor_trainer.predict_batch(current_batch, detach=True)

            self._discriminator_trainer.enable_gradients()
            self._preprocessor_trainer.disable_gradients()
            self._segmenter_trainer.disable_gradients()

            loss_D_X = self._train_discriminator(current_batch["X"], X_n, debug=self.config.debug)

            self._discriminator_trainer.disable_gradients()
            self._preprocessor_trainer.enable_gradients()
            self._segmenter_trainer.enable_gradients()

            # Train segmenter. Must do inference on Normalizer again since that part of the graph has been
            # previously detached.
            X_n = self._preprocessor_trainer.predict_batch(current_batch, detach=False)
            self._preprocessor_trainer.disable_gradients()
            loss_S_X_n, X_s = self._train_segmenter(X_n, current_batch["y"], debug=self.config.debug)

            # Try to fool the discriminator with Normalized data as "real" data.
            loss_D_X_n = self._evaluate_discriminator_error_on_normalized_data(X_n)

            self._preprocessor_trainer.enable_gradients()
            self._segmenter_trainer.disable_gradients()
            self._discriminator_trainer.disable_gradients()

            total_error = self._train_normalizer(loss_S_X_n, loss_D_X_n, debug=self.config.debug)

            # Re-enable all parts of the graph.
            self._at_iteration_end()

            if iteration % self.config.logger_config.frequency == 0:
                dsc = self._segmenter_trainer.config.metric.compute().cuda()
                acc = torch.Tensor([self._discriminator_trainer.config.metric.compute()]).cuda()

                # Average loss and accuracy across processes for logging
                if self.config.running_config.is_distributed:
                    loss_D_X = self._reduce_tensor(loss_D_X.data)
                    loss_S_X_n = self._reduce_tensor(loss_S_X_n.data)
                    loss_D_X_n = self._reduce_tensor(loss_D_X_n.data)
                    total_error = self._reduce_tensor(total_error.data)
                    dsc = self._reduce_tensor(dsc.data)
                    acc = self._reduce_tensor(acc.data)

                self._discriminator_trainer.training_loss.update(loss_D_X.item(), current_batch["X"].size(0))
                self._segmenter_trainer.training_loss.update(loss_S_X_n.item(), current_batch["X"].size(0))

                self._discriminator_trainer.training_metric.update(acc.item(), current_batch["X"].size(0))
                self._segmenter_trainer.training_metric.update(dsc.item(), current_batch["X"].size(0))

                self._total_loss.update(total_error.item(), current_batch["X"].size(0))
                self._D_X_n_loss.update(loss_D_X_n.item(), current_batch["X"].size(0))

                torch.cuda.synchronize()

                if self.config.running_config.local_rank == 0:
                    self._logger.log_images(current_batch["X"], current_batch["y"], X_n,
                                            torch.argmax(torch.nn.functional.softmax(X_s, dim=1), dim=1,
                                                         keepdim=True), step=self._global_step)

                    self._logger.log_stats("training", {
                        "alpha": 0,
                        "lambda": self.config.variables.lambda_,
                        "segmenter_loss": loss_S_X_n,
                        "discriminator_loss_D_X_n": loss_D_X_n,
                        "discriminator_loss_D_X": loss_D_X,
                        "discriminator_metric": acc.item(),
                        "learning_rate": self.config.optimizer[0].param_groups[0]['lr'],
                        "segmenter_metric": self._segmenter_trainer.training_metric.average,
                    }, iteration)

                    self._logger.info(
                        "Discriminator loss: {}, Discriminator loss on normalized data: {}, Total loss: {}, Segmenter loss: {}, Segmenter Dice Score: {}".format(
                            self._discriminator_trainer.training_loss.average,
                            self._D_X_n_loss.average,
                            self._total_loss.average,
                            self._segmenter_trainer.training_loss.average,
                            self._segmenter_trainer.training_metric.average))

                    self._global_step += 1

        self._at_epoch_end(epoch_num)

    def train_batch(self, data_dict: dict, fold=0, **kwargs):
        pass

    @staticmethod
    def _concatenate_data(data_d0, data_d1):
        return torch.cat([data_d0[0], data_d1[0]], dim=0), torch.cat([data_d0[1], data_d1[1]], dim=0)

    @staticmethod
    def _prepare_batch(batch: dict, input_device: torch.device, output_device: torch.device):
        X, y = batch["X"], batch["y"]
        return {"X": X.to(device=output_device), "y": y.squeeze(1).to(device=output_device).long()}

    def _validate_epoch(self, epoch_num: int, **kwargs):
        if self.config.running_config.local_rank == 0:
            self._logger.info("Validating...")

        with torch.no_grad():
            for i, (batch_d0, batch_d1) in enumerate(zip(self.config.dataloader[0].get_validation_dataloader(),
                                                         self.config.dataloader[1].get_validation_dataloader()), 0):
                batch = self._concatenate_data(batch_d0, batch_d1)
                current_batch = self._prepare_batch(batch={"X": batch[X], "y": batch[Y]},
                                                    input_device=torch.device('cpu'),
                                                    output_device=self.config.running_config.device)
                out = self._validate_batch(current_batch)

                self._segmenter_trainer.validation_loss.update(out["out_segmenter"]["loss"].item(),
                                                               current_batch["X"].size(0))

                self._segmenter_trainer.config.metric.update((out["out_segmenter"]["output"], current_batch["y"]))

                self._discriminator_trainer.validation_loss.update(out["out_discriminator"]["loss"].item(),
                                                                   current_batch["X"].size(0))

            self._segmenter_trainer.validation_metric.update(self._segmenter_trainer.config.metric.compute(),
                                                             current_batch["X"].size(0))
            self._discriminator_trainer.validation_metric.update(self._discriminator_trainer.config.metric.compute(),
                                                                 current_batch["X"].size(0))
            self._segmenter_trainer.config.metric.reset()
            self._discriminator_trainer.config.metric.reset()

            if self.config.running_config.local_rank == 0:
                self._logger.log_validation_stats("validation", {
                    "segmenter_loss": self._segmenter_trainer.validation_loss.average,
                    "segmenter_metric": self._segmenter_trainer.validation_metric.average,
                    "discriminator_loss_D_X_n": self._discriminator_trainer.validation_loss.average,
                    "discriminator_metric": self._discriminator_trainer.validation_metric.average,

                }, epoch_num)

                self._segmenter_trainer._logger.info(
                    "Validation Epoch: {}, Segmenter validation dice loss: {}, Segmenter validation dice score: {}".format(
                        epoch_num,
                        self._segmenter_trainer.validation_loss.average,
                        self._segmenter_trainer.validation_metric.average))
                self._discriminator_trainer._logger.info(
                    "Validation Epoch: {}, Discriminator D_X_n validation CE loss: {}, Discriminator accuracy: {}".format(
                        epoch_num,
                        self._discriminator_trainer.validation_loss.average,
                        self._discriminator_trainer.validation_metric.average))

    def _validate_batch(self, data_dict: dict):
        out_segmenter = self._validate_segmenter(data_dict)

        out_discriminator = self._validate_discriminator(data_dict)

        return {"out_segmenter": out_segmenter, "out_discriminator": out_discriminator}

    def _validate_segmenter(self, data_dict: dict):
        X_n = self._preprocessor_trainer.predict_batch(data_dict)
        X_s = self._segmenter_trainer.predict_batch({"X": X_n, "y": data_dict["y"]})

        loss_S_X_n = self._segmenter_trainer.config.criterion(torch.nn.functional.softmax(X_s, dim=1),
                                                              to_onehot(data_dict["y"], num_classes=4))

        if self.config.running_config.is_distributed:
            loss_S_X_n = self._reduce_tensor(loss_S_X_n.data)

        return {"loss": loss_S_X_n, "output": X_s}

    def _validate_discriminator(self, data_dict: dict):
        X_n = self._preprocessor_trainer.predict_batch(data_dict)

        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.FAKE_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)

        pred_X_n = self._discriminator_trainer.predict_batch({"X": X_n, "y": y})

        loss_D_X_n = self._discriminator_trainer.config.criterion(pred_X_n, y)

        if self.config.running_config.is_distributed:
            loss_D_X_n = self._reduce_tensor(loss_D_X_n.data)

        self._discriminator_trainer.config.metric.update((pred_X_n, y))

        return {"loss": loss_D_X_n, "output": pred_X_n}

    def finalize(self, *args, **kwargs):
        pass

    def _at_training_begin(self, *args, **kwargs):
        pass

    def _at_training_end(self, *args, **kwargs):
        pass

    def _at_epoch_begin(self, *args, **kwargs):
        pass

    def _at_epoch_end(self, epoch_num: int, *args, **kwargs):
        self._validate_epoch(epoch_num)

    def _at_iteration_begin(self):
        self._preprocessor_trainer.at_iteration_begin()
        self._segmenter_trainer.at_iteration_begin()

    def _at_iteration_end(self):
        self._preprocessor_trainer.at_iteration_end()
        self._segmenter_trainer.at_iteration_end()
        self._discriminator_trainer.at_iteration_end()

    @staticmethod
    def _reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
        rt /= int(os.environ['WORLD_SIZE'])
        return rt

    def _train_discriminator(self, X, X_n, debug=False):
        if debug:
            params_discriminator = [np for np in self._discriminator_trainer.config.model.named_parameters() if
                                    np[1].requires_grad]
            params_segmenter = [np for np in self._segmenter_trainer.config.model.named_parameters() if
                                np[1].requires_grad]
            params_normalizer = [np for np in self._preprocessor_trainer.config.model.named_parameters() if
                                 np[1].requires_grad]
            initial_params_discriminator = [(name, p.clone()) for (name, p) in params_discriminator]
            initial_params_segmenter = [(name, p.clone()) for (name, p) in params_segmenter]
            initial_params_normalizer = [(name, p.clone()) for (name, p) in params_normalizer]

            loss_D = self._train_discriminator_subroutine(X, X_n)

            # Analyze behavior of networks parameters.
            ModelHelper().assert_model_params_not_changed(initial_params_normalizer, params_normalizer)
            ModelHelper().assert_model_params_not_changed(initial_params_segmenter, params_segmenter)
            ModelHelper().assert_model_params_changed(initial_params_discriminator, params_discriminator)
            ModelHelper().assert_model_grads_are_none(
                self._preprocessor_trainer.config.model.named_parameters())
            ModelHelper().assert_model_grads_are_not_none(
                self._segmenter_trainer.config.model.named_parameters())
            ModelHelper().assert_model_grads_are_none(
                self._discriminator_trainer.config.model.named_parameters())

        else:
            loss_D = self._train_discriminator_subroutine(X, X_n)

        self._discriminator_trainer.config.optimizer.zero_grad()

        return loss_D

    def _train_discriminator_subroutine(self, X, X_n):
        pred_X = self._discriminator_trainer.predict_batch({"X": X})

        y = torch.Tensor().new_full(size=(X.size(0),), fill_value=self.REAL_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_D_X = self._discriminator_trainer.evaluate_loss({"X": pred_X, "y": y})

        with amp.scale_loss(loss_D_X, self._discriminator_trainer.config.optimizer):
            loss_D_X.backward()

        pred_X_n = self._discriminator_trainer.predict_batch({"X": X_n})
        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.FAKE_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_D_X_n = self._discriminator_trainer.evaluate_loss({"X": pred_X_n, "y": y})

        with amp.scale_loss(loss_D_X_n, self._discriminator_trainer.config.optimizer):
            loss_D_X_n.backward()

        self._discriminator_trainer.config.optimizer.step()
        self._discriminator_trainer.config.metric.update((pred_X_n, y))

        return loss_D_X_n

    def _evaluate_discriminator_error_on_normalized_data(self, X_n):
        pred_X_n = self._discriminator_trainer.predict_batch({"X": X_n})
        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.REAL_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_D_X_n = self._discriminator_trainer.config.criterion(pred_X_n, y)

        return loss_D_X_n

    def _train_segmenter(self, X_n, y, debug=False):
        if debug:
            params_discriminator = [np for np in self._discriminator_trainer.config.model.named_parameters() if
                                    np[1].requires_grad]
            params_segmenter = [np for np in self._segmenter_trainer.config.model.named_parameters() if
                                np[1].requires_grad]
            params_normalizer = [np for np in self._preprocessor_trainer.config.model.named_parameters() if
                                 np[1].requires_grad]
            initial_params_discriminator = [(name, p.clone()) for (name, p) in params_discriminator]
            initial_params_segmenter = [(name, p.clone()) for (name, p) in params_segmenter]
            initial_params_normalizer = [(name, p.clone()) for (name, p) in params_normalizer]

            loss_S_X_n, X_s = self._train_segmenter_subroutine(X_n, y)

            # Analyze behavior of networks parameters.
            ModelHelper().assert_model_params_not_changed(initial_params_discriminator, params_discriminator)
            ModelHelper().assert_model_params_changed(initial_params_segmenter, params_segmenter)
            ModelHelper().assert_model_params_not_changed(initial_params_normalizer, params_normalizer)
            ModelHelper().assert_model_grads_are_none(
                self._preprocessor_trainer.config.model.named_parameters())
            ModelHelper().assert_model_grads_are_not_none(
                self._segmenter_trainer.config.model.named_parameters())
            ModelHelper().assert_model_grads_are_none(
                self._discriminator_trainer.config.model.named_parameters())

        else:
            loss_S_X_n, X_s = self._train_segmenter_subroutine(X_n, y)

        return loss_S_X_n, X_s

    def _train_segmenter_subroutine(self, X_n, y):
        X_s = self._segmenter_trainer.predict_batch({"X": X_n})
        loss_S_X_s = self._segmenter_trainer.config.criterion(torch.nn.functional.softmax(X_s, dim=1),
                                                              to_onehot(y, num_classes=4))
        with amp.scale_loss(loss_S_X_s, self._segmenter_trainer.config.optimizer):
            loss_S_X_s.backward(retain_graph=True)  # Retain graph in VRAM for future error addition.
        self._segmenter_trainer.config.optimizer.step()

        self._segmenter_trainer.config.metric.update((X_s, y))

        return loss_S_X_s, X_s

    def _train_normalizer(self, loss_S_X_n, loss_D_X_n, debug=False):
        if debug:
            params_discriminator = [np for np in self._discriminator_trainer.config.model.named_parameters() if
                                    np[1].requires_grad]
            params_segmenter = [np for np in self._segmenter_trainer.config.model.named_parameters() if
                                np[1].requires_grad]
            params_normalizer = [np for np in self._preprocessor_trainer.config.model.named_parameters() if
                                 np[1].requires_grad]
            initial_params_discriminator = [(name, p.clone()) for (name, p) in params_discriminator]
            initial_params_segmenter = [(name, p.clone()) for (name, p) in params_segmenter]
            initial_params_normalizer = [(name, p.clone()) for (name, p) in params_normalizer]

            total_error = self._train_normalizer_subroutine(loss_S_X_n, loss_D_X_n)

            # Analyze behavior of networks parameters.
            ModelHelper().assert_model_params_not_changed(initial_params_discriminator, params_discriminator)
            ModelHelper().assert_model_params_not_changed(initial_params_segmenter, params_segmenter)
            ModelHelper().assert_model_params_changed(initial_params_normalizer, params_normalizer)
            ModelHelper().assert_model_grads_are_none(
                self._preprocessor_trainer.config.model.named_parameters())
            ModelHelper().assert_model_grads_are_not_none(
                self._segmenter_trainer.config.model.named_parameters())
            ModelHelper().assert_model_grads_are_none(
                self._discriminator_trainer.config.model.named_parameters())

        else:
            total_error = self._train_normalizer_subroutine(loss_S_X_n, loss_D_X_n)

        return total_error

    def _train_normalizer_subroutine(self, loss_S_X_s, loss_D_X_n):
        # Compute total error on Normalizer and apply gradients on Normalizer.
        total_error = loss_S_X_s + self.config.variables.lambda_ * loss_D_X_n
        with amp.scale_loss(total_error, self._preprocessor_trainer.config.optimizer):
            total_error.backward()
        self._preprocessor_trainer.config.optimizer.step()

        return total_error

    def _setup(self, *args, **kwargs):
        pass
