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

from samitorch.metrics.gauges import RunningAverageGauge
from samitorch.training.trainer import Trainer
from samitorch.utils.utils import to_onehot

from deepNormalize.utils.logger import Logger


class DeepNormalizeTrainer(Trainer):
    NORMALIZER = 0
    SEGMENTER = 1
    DISCRIMINATOR = 2
    REAL_LABEL = 1
    FAKE_LABEL = 0
    DICE_LOSS = 0
    CROSS_ENTROPY = 1

    def __init__(self, config, callbacks):
        super(DeepNormalizeTrainer, self).__init__(config, callbacks)

        self._segmenter_training_dice_score = RunningAverageGauge()
        self._segmenter_training_dice_loss = RunningAverageGauge()
        self._discriminator_training_accuracy = RunningAverageGauge()
        self._discriminator_training_loss = RunningAverageGauge()
        self._normalizer_training_loss = RunningAverageGauge()
        self._total_training_loss = RunningAverageGauge()

        self._segmenter_validation_dice_score = RunningAverageGauge()
        self._segmenter_validation_dice_loss = RunningAverageGauge()
        self._discriminator_validation_accuracy = RunningAverageGauge()
        self._discriminator_validation_loss = RunningAverageGauge()
        self._normalizer_validation_loss = RunningAverageGauge()
        self._total_validation_loss = RunningAverageGauge()

        self._at_training_begin()
        self._logger = Logger(self.config.logger_config)

    def train(self):
        for iteration in range(self.config.max_epoch):
            for i, sample in enumerate(self.config.dataloader):
                X, y = self.prepare_batch(batch=sample, input_device=torch.device('cpu'),
                                          output_device=self.config.running_config.device)

                self._at_iteration_begin()

                # Generate normalized data and DO detach (so gradients ARE NOT calculated for generator).
                X_n = self.config.model[self.NORMALIZER](X).detach()
                self._disable_gradients(self.config.model[self.NORMALIZER])
                self._disable_gradients(self.config.model[self.SEGMENTER])

                # Train the discriminator part of the graph.
                self._train_discriminator(X, X_n)

                self._disable_gradients(self.config.model[self.DISCRIMINATOR])
                self._enable_gradients(self.config.model[self.NORMALIZER])
                self._enable_gradients(self.config.model[self.SEGMENTER])

                # Train segmenter. Must do inference on Normalizer again since that part of the graph has been
                # previously detached.
                X_n = self.config.model[self.NORMALIZER](X)
                X_s = self.config.model[self.SEGMENTER](X_n)
                loss_S_X_s = self._train_segmenter(X_s, y)

                # Try to fool the discriminator with Normalized data as "real" data.
                loss_D_X_n = self._evaluate_discriminator_error_on_normalized_data(X_n)

                self._enable_gradients(self.config.model[self.NORMALIZER])
                self._disable_gradients(self.config.model[self.SEGMENTER])

                # Compute total error on Normalizer and apply gradients on Normalizer.
                total_error = loss_S_X_s + self.config.variables.lambda_ * loss_D_X_n
                total_error.backward()
                self.config.optimizer[self.NORMALIZER].step()

                # Re-enable all parts of the graph.
                [self._enable_gradients(model) for model in self.config.model]

                self._logger.log_images(X, y, X_n, torch.argmax(torch.nn.functional.softmax(X_s, dim=1), dim=1, keepdim=True), iteration)
                self._logger.log_stats("training", {
                    "alpha": 0,
                    "lambda": self.config.variables.lambda_,
                    "segmenter_loss": loss_S_X_s,
                    "discriminator_loss": loss_D_X_n,
                    "learning_rate": self.config._optimizer[0].param_groups[0]['lr']
                }, iteration)

                print("debug")

    def train_epoch(self, epoch_num: int, **kwargs):
        pass

    def train_batch(self, data_dict: dict, fold=0, **kwargs):
        pass

    @staticmethod
    def prepare_batch(batch: tuple, input_device: torch.device, output_device: torch.device):
        X, y = batch
        return X.to(device=output_device), y.squeeze(1).to(device=output_device).long()

    def validate_epoch(self, epoch_num: int, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def _setup(self, *args, **kwargs):
        pass

    def _at_training_begin(self, *args, **kwargs):

        distributed = False

        if 'WORLD_SIZE' in os.environ:
            distributed = int(os.environ['WORLD_SIZE']) > 1

        self._gpu = 0
        self._world_size = 1

        if distributed:
            self._gpu = self.config.running_config.local_rank
            torch.cuda.set_device(self._gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            self._world_size = torch.distributed.get_world_size()

        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

        if self.config.running_config.sync_batch_norm:
            import apex
            print("using apex synced BN")
            self._config.models = [apex.parallel.convert_syncbn_model(model) for model in self._config.model]

        # Transfer models on CUDA device.
        [model.cuda() for model in self.config.model]

        # Scale learning rate based on global batch size
        for optim in self.config.optimizer:
            optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"] * float(
                self.config.dataloader.batch_size * self._world_size) / 256.

        # Initialize Amp.
        for i, (model, optimizer) in enumerate(zip(self.config.model, self.config.optimizer)):
            self.config.model[i], self.config.optimizer[i] = amp.initialize(model, optimizer,
                                                                            opt_level=self.config.running_config.opt_level,
                                                                            keep_batchnorm_fp32=self.config.running_config.keep_batch_norm_fp32,
                                                                            loss_scale=self.config.running_config.loss_scale)

        # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
        # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
        # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
        # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
        if distributed:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # model = DDP(model)
            # delay_allreduce delays all communication to the end of the backward pass.
            [DDP(model, delay_allreduce=True) for model in self.config.model]

        [criterion.cuda() for criterion in self.config.criterion]

    def _at_training_end(self, *args, **kwargs):
        pass

    def _at_epoch_begin(self, *args, **kwargs):
        pass

    def _at_epoch_end(self, *args, **kwargs):
        pass

    def _at_iteration_begin(self):
        [model.train() for model in self.config.model]
        [optimizer.zero_grad() for optimizer in self.config.optimizer]

    @staticmethod
    def _disable_gradients(model):
        for p in model.parameters(): p.requires_grad = False

    @staticmethod
    def _enable_gradients(model):
        for p in model.parameters(): p.requires_grad = True

    def _train_discriminator(self, X, X_n):
        pred_X = self.config.model[self.DISCRIMINATOR](X)
        y = torch.Tensor().new_full(size=(X.size(0),), fill_value=self.REAL_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_X = self.config.criterion[self.CROSS_ENTROPY](pred_X.squeeze(1), y)

        loss_X.backward()

        pred_X_n = self.config.model[self.DISCRIMINATOR](X_n)
        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.FAKE_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_X_n = self.config.criterion[self.CROSS_ENTROPY](pred_X_n, y)

        loss_X_n.backward()

        self.config.optimizer[self.DISCRIMINATOR].step()

    def _evaluate_discriminator_error_on_normalized_data(self, X_n):
        pred_X_n = self.config.model[self.DISCRIMINATOR](X_n)
        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.REAL_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_X_n = self.config.criterion[self.CROSS_ENTROPY](pred_X_n, y)

        return loss_X_n

    def _train_segmenter(self, X_s, y):
        loss_S_X_s = self.config.criterion[self.DICE_LOSS](torch.nn.functional.softmax(X_s, dim=1),
                                                           to_onehot(y, num_classes=4))
        loss_S_X_s.backward(retain_graph=True)  # Retain graph in VRAM for future error addition.
        self.config.optimizer[self.SEGMENTER].step()

        return loss_S_X_s
