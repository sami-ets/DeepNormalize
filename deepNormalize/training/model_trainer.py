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

cudnn.benchmark = True
cudnn.enabled = True

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from samitorch.metrics.gauges import RunningAverageGauge
from samitorch.training.trainer import Trainer
from samitorch.utils.utils import to_onehot

from deepNormalize.utils.logger import Logger

# Constant definitions for readability.
X = 0
Y = 1


class ModelTrainer(Trainer):

    def __init__(self, config, callbacks, class_name):
        super(ModelTrainer, self).__init__(config, callbacks)

        self.training_metric = RunningAverageGauge()
        self.training_loss = RunningAverageGauge()

        self.validation_metric = RunningAverageGauge()
        self.validation_loss = RunningAverageGauge()

        self._setup()
        self._at_training_begin()

        self._global_step = 0

        if self.config.running_config.local_rank == 0:
            self._logger = Logger(self.config.logger_config, class_name)

    def train(self):
        pass

    def _train_epoch(self, epoch_num: int, **kwargs):
        pass

        # for iteration, batch in enumerate(self.config.dataloader, 0):
        #     self._at_iteration_begin()
        #
        #     current_batch = self._prepare_batch(batch={"X": batch[X], "y": batch[Y]}, input_device=torch.device('cpu'),
        #                                         output_device=self.config.running_config.device)
        #
        #     out = self._train_batch(current_batch)
        #
        #     self.config.metric.update((out["output"], current_batch["y"]))
        #
        #     if iteration % self.config.logger_config.frequency == 0:
        #         metric = self.config.metric.compute().cuda()
        #
        #         if self.config.running_config.is_distributed:
        #             out["loss"] = self._reduce_tensor(out["loss"].data)
        #             metric = self._reduce_tensor(metric.data)
        #
        #         self._training_metric.update(metric.item(), current_batch["X"].size(0))
        #         self._training_loss.update(out["loss"].item(), current_batch["X"].size(0))
        #
        #         if self.config.running_config.local_rank == 0:
        #             self._logger.log_images(batch[X], batch[Y],
        #                                     torch.argmax(torch.nn.functional.softmax(out["output"], dim=1), dim=1,
        #                                                  keepdim=True), self._global_step)
        #
        #             self._log({"loss": self._training_loss.average,
        #                        "metric": self._training_metric.average,
        #                        "lr": self.config.optimizer.param_groups[0]["lr"],
        #                        "epoch_num": epoch_num,
        #                        "iteration": iteration}, training=True)
        #
        #             self._global_step += 1
        #
        # self._at_epoch_end(epoch_num)

    def train_batch(self, data_dict: dict):
        X_t = self.config.model(data_dict["X"])

        loss_T_X = self.config.criterion(torch.nn.functional.softmax(X_t, dim=1),
                                         to_onehot(data_dict["y"], num_classes=4))

        with amp.scale_loss(loss_T_X, self.config.optimizer):
            loss_T_X.backward()

        self.config.optimizer.step()

        return {"loss": loss_T_X, "output": X_t}

    def predict_batch(self, data_dict: dict, detach: bool = False):
        if detach:
            return self.config.model(data_dict["X"]).detach()
        else:
            return self.config.model(data_dict["X"])

    def evaluate_loss(self, data_dict: dict):
        return self.config.criterion(data_dict["X"], data_dict["y"])

    def _validate_epoch(self, epoch_num: int, **kwargs):
        pass
        # if self.config.running_config.local_rank == 0:
        #     self._logger.info("Validating...")
        #
        # with torch.no_grad():
        #     for i, batch in enumerate(self.config.dataloader.get_validation_dataloader()):
        #         current_batch = self._prepare_batch(batch={"X": batch[X], "y": batch[Y]},
        #                                             input_device=torch.device('cpu'),
        #                                             output_device=self.config.running_config.device)
        #         out = self._validate_batch(current_batch)
        #
        #         self._validation_loss.update(out["loss"].item(), current_batch["X"].size(0))
        #
        #         self.config.metric.update((out["output"], current_batch["y"]))
        #
        #     self._validation_metric.update(self._compute_metric(), current_batch["X"].size(0))
        #
        #     if self.config.running_config.local_rank == 0:
        #         self._log({"loss": self._validation_loss.average,
        #                    "metric": self._validation_metric.average,
        #                    "epoch_num": epoch_num}, training=False)

    def _validate_batch(self, data_dict: dict):
        X_s = self.config.model(data_dict["X"])

        loss_S_X = self.config.criterion(torch.nn.functional.softmax(X_s, dim=1),
                                         to_onehot(data_dict["y"], num_classes=4))

        if self.config.running_config.is_distributed:
            loss_S_X = self._reduce_tensor(loss_S_X.data)

        return {"loss": loss_S_X, "output": X_s}

    @staticmethod
    def _prepare_batch(batch: dict, input_device: torch.device, output_device: torch.device):
        X, y = batch["X"], batch["y"]
        return {"X": X.to(device=output_device), "y": y.squeeze(1).to(device=output_device).long()}

    def _at_training_begin(self, *args, **kwargs):
        self._initialize_model(self.config.model)

    def _at_training_end(self, *args, **kwargs):
        pass

    def _at_epoch_begin(self, *args, **kwargs):
        pass

    def _at_epoch_end(self, epoch_num: int, *args, **kwargs):
        self._validate_epoch(epoch_num)

    def at_iteration_begin(self):
        self.config.model.train()
        self.config.optimizer.zero_grad()

    def at_iteration_end(self):
        self.enable_gradients()

    def _at_validation_begin(self):
        self.config.model.eval()

    def _finalize(self, *args, **kwargs):
        pass

    def _compute_metric(self):
        dsc = self.config.metric.compute().item()
        self.config.metric.reset()
        return dsc

    def _log(self, data_dict: dict, training: bool = True):
        if training:
            self._logger.log_stats("training", {
                "loss": data_dict["loss"],
                "metric": data_dict["metric"],
                "learning_rate": data_dict["lr"],
            }, data_dict["epoch_num"])

            self._logger.info(
                "Epoch: {} Step {}, Segmenter dice loss: {}, Segmenter dice score: {}".format(
                    data_dict["epoch_num"],
                    data_dict["iteration"],
                    data_dict["loss"],
                    data_dict["metric"]))
        else:
            self._logger.log_validation_stats("validation", {
                "loss": data_dict["loss"],
                "metric": data_dict["metric"],
            }, data_dict["epoch_num"])

            self._logger.info(
                "Validation Epoch: {}, Segmenter validation dice loss: {}, Segmenter validation dice score: {}".format(
                    data_dict["epoch_num"],
                    data_dict["loss"],
                    data_dict["metric"]))

    def disable_gradients(self):
        for p in self.config.model.parameters(): p.requires_grad = False

    def enable_gradients(self):
        for p in self.config.model.parameters(): p.requires_grad = True

    def _reduce_tensor(self, tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= int(self.config.running_config.world_size)
        return rt

    @staticmethod
    def _initialize_model(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _setup(self, *args, **kwargs):

        assert torch.backends.cudnn.enabled, "Amp requires CuDNN backend to be enabled."

        if self.config.running_config.sync_batch_norm:
            import apex
            print("Using apex synced BN.")
            self.config.model = apex.parallel.convert_syncbn_model(self.config.model)

        # Transfer models on CUDA device.
        self.config.model.cuda()

        # Scale learning rate based on global batch size
        if self.config.running_config.loss_scale is not "dynamic":
            self.config.optimizer.param_groups[0]["lr"] = self.config.optimizer.param_groups[0]["lr"] * float(
                self.config.dataloader.batch_size * self.config.running_config.world_size) / float(
                self.config.running_config.loss_scale)

        # Initialize Amp.
        self.config.model, self.config.optimizer = amp.initialize(self.config.model, self.config.optimizer,
                                                                  opt_level=self.config.running_config.opt_level,
                                                                  keep_batchnorm_fp32=self.config.running_config.keep_batch_norm_fp32,
                                                                  loss_scale=self.config.running_config.loss_scale)

        torch.optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, factor=0.1, patience=3, verbose=True)

        # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
        # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
        # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
        # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
        if self.config.running_config.is_distributed:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # model = DDP(model)
            # delay_allreduce delays all communication to the end of the backward pass.
            self.config.model = DDP(self.config.model, delay_allreduce=True)
            # self.config.model = [DDP(model, delay_allreduce=True) for model in self.config.model]

        self.config.criterion = self.config.criterion.cuda()
