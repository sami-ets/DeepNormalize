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
from tests.models.model_helper_test import ModelHelper


class DeepNormalizeTrainer(Trainer):
    NORMALIZER = 0
    SEGMENTER = 1
    DISCRIMINATOR = 2
    REAL_LABEL = 1
    FAKE_LABEL = 0
    DICE_LOSS = 0
    CROSS_ENTROPY = 1
    DICE_METRIC = 0
    DISCRIMINATOR_ACCURACY = 1

    def __init__(self, config, callbacks):
        super(DeepNormalizeTrainer, self).__init__(config, callbacks)

        self._segmenter_training_dice_score = RunningAverageGauge()
        self._segmenter_training_dice_loss = RunningAverageGauge()
        self._discriminator_training_accuracy = RunningAverageGauge()
        self._discriminator_training_loss = RunningAverageGauge()
        self._discriminator_on_normalized_data_training_training_loss = RunningAverageGauge()
        self._normalizer_training_loss = RunningAverageGauge()
        self._total_training_loss = RunningAverageGauge()

        self._segmenter_validation_dice_score = RunningAverageGauge()
        self._segmenter_validation_dice_loss = RunningAverageGauge()
        self._discriminator_validation_accuracy = RunningAverageGauge()
        self._discriminator_validation_loss = RunningAverageGauge()
        self._normalizer_validation_loss = RunningAverageGauge()
        self._total_validation_loss = RunningAverageGauge()

        self._setup()
        self._distributed = None

        if self.config.running_config.local_rank == 0:
            self._logger = Logger(self.config.logger_config)

    def train(self):
        for iteration in range(self.config.max_epoch):
            for i, sample in enumerate(self.config.dataloader, 0):
                X, y = self.prepare_batch(batch=sample, input_device=torch.device('cpu'),
                                          output_device=self.config.running_config.device)

                self._at_iteration_begin()

                # Generate normalized data and DO detach (so gradients ARE NOT calculated for generator and segmenter).
                X_n = self.config.model[self.NORMALIZER](X).detach()
                self._enable_gradients(self.config.model[self.DISCRIMINATOR])
                self._disable_gradients(self.config.model[self.NORMALIZER])
                self._disable_gradients(self.config.model[self.SEGMENTER])

                loss_D = self._train_discriminator(X, X_n, debug=self.config.debug)

                self._disable_gradients(self.config.model[self.DISCRIMINATOR])
                self._enable_gradients(self.config.model[self.NORMALIZER])
                self._enable_gradients(self.config.model[self.SEGMENTER])

                # Train segmenter. Must do inference on Normalizer again since that part of the graph has been
                # previously detached.
                X_n = self.config.model[self.NORMALIZER](X)
                self._disable_gradients(self.config.model[self.NORMALIZER])
                X_s = self.config.model[self.SEGMENTER](X_n)
                loss_S_X_n = self._train_segmenter(X_s, y, debug=self.config.debug)
                self.config.metric[self.DICE_METRIC].update((X_s, y))
                self._segmenter_training_dice_score.update(self.config.metric[self.DICE_METRIC].compute())

                # Try to fool the discriminator with Normalized data as "real" data.
                loss_D_X_n = self._evaluate_discriminator_error_on_normalized_data(X_n)

                self._enable_gradients(self.config.model[self.NORMALIZER])
                self._disable_gradients(self.config.model[self.SEGMENTER])
                self._disable_gradients(self.config.model[self.DISCRIMINATOR])

                total_error = self._train_normalizer(loss_S_X_n, loss_D_X_n, debug=self.config.debug)

                # Re-enable all parts of the graph.
                [self._enable_gradients(model) for model in self.config.model]

                if i % self.config.logger_config.log_after_iterations == 0:
                    # Average loss and accuracy across processes for logging
                    if self._distributed:
                        loss_D = self._reduce_tensor(loss_D.data)
                        loss_S_X_n = self._reduce_tensor(loss_S_X_n.data)
                        loss_D_X_n = self._reduce_tensor(loss_D_X_n.data)

                    self._discriminator_training_loss.update(loss_D.item())
                    self._segmenter_training_dice_loss.update(loss_S_X_n.item())
                    self._discriminator_on_normalized_data_training_training_loss.update(loss_D_X_n.item())
                    self._total_training_loss.update(total_error.item())

                    torch.cuda.synchronize()

                    if self.config.running_config.local_rank == 0:
                        self._logger.log_images(X, y, X_n,
                                                torch.argmax(torch.nn.functional.softmax(X_s, dim=1), dim=1,
                                                             keepdim=True),
                                                iteration)

                        self._logger.log_stats("training", {
                            "alpha": 0,
                            "lambda": self.config.variables.lambda_,
                            "segmenter_loss": loss_S_X_n,
                            "discriminator_loss_D_X_n": loss_D_X_n,
                            "discriminator_loss_D": loss_D,
                            "learning_rate": self.config.optimizer[0].param_groups[0]['lr'],
                            "dice_score": self._segmenter_training_dice_score.average,
                        }, iteration)

                        self._logger.info(
                            "Discriminator loss: {}, Discriminator loss on normalized data: {}, Normalizer loss: {}, Total loss: {}".format(
                                self._discriminator_training_loss.average,
                                self._discriminator_on_normalized_data_training_training_loss.average,
                                self._normalizer_training_loss.average,
                                self._total_training_loss.average))

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
        self._distributed = False

        if 'WORLD_SIZE' in os.environ:
            self._distributed = int(os.environ['WORLD_SIZE']) > 1

        self._gpu = 0
        self._world_size = 1

        if self._distributed:
            self._gpu = self.config.running_config.local_rank
            torch.cuda.set_device(self._gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            self._world_size = torch.distributed.get_world_size()

        assert torch.backends.cudnn.enabled, "Amp requires CuDNN backend to be enabled."

        if self.config.running_config.sync_batch_norm:
            import apex
            print("Using apex synced BN.")
            self._config.models = [apex.parallel.convert_syncbn_model(model) for model in self._config.model]

        # Transfer models on CUDA device.
        [model.cuda() for model in self.config.model]

        # Scale learning rate based on global batch size
        for optim in self.config.optimizer:
            optim.param_groups[0]["lr"] = optim.param_groups[0]["lr"] * float(
                self.config.dataloader.batch_size * self._world_size) / 64.

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
        if self._distributed:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # model = DDP(model)
            # delay_allreduce delays all communication to the end of the backward pass.
            # for i, model in enumerate(self.config.model):
            #     model_ = DDP(model, delay_allreduce=True)
            #     self.config.model[i] = model_
            self.config.model = [DDP(model, delay_allreduce=True) for model in self.config.model]

        # for i, criterion in enumerate(self.config.criterion):
        #     criterion_ = criterion.cuda()
        #     self.config.criterion[i] = criterion_
        self.config.criterion = [criterion.cuda() for criterion in self.config.criterion]

    def _at_training_begin(self, *args, **kwargs):
        pass

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

    @staticmethod
    def _reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
        rt /= int(os.environ['WORLD_SIZE'])
        return rt

    def _train_discriminator(self, X, X_n, debug=False):
        if debug:
            params_discriminator = [np for np in self.config.model[self.DISCRIMINATOR].named_parameters() if
                                    np[1].requires_grad]
            params_segmenter = [np for np in self.config.model[self.SEGMENTER].named_parameters() if
                                np[1].requires_grad]
            params_normalizer = [np for np in self.config.model[self.NORMALIZER].named_parameters() if
                                 np[1].requires_grad]
            initial_params_discriminator = [(name, p.clone()) for (name, p) in params_discriminator]
            initial_params_segmenter = [(name, p.clone()) for (name, p) in params_segmenter]
            initial_params_normalizer = [(name, p.clone()) for (name, p) in params_normalizer]

            loss_D = self._train_discriminator_subroutine(X, X_n)

            # Analyze behavior of networks parameters.
            ModelHelper().assert_model_params_not_changed(initial_params_normalizer, params_normalizer)
            ModelHelper().assert_model_params_not_changed(initial_params_segmenter, params_segmenter)
            ModelHelper().assert_model_params_changed(initial_params_discriminator, params_discriminator)
            ModelHelper().assert_model_grads_are_none(self.config.model[self.NORMALIZER].named_parameters())
            ModelHelper().assert_model_grads_are_none(self.config.model[self.SEGMENTER].named_parameters())
            ModelHelper().assert_model_grads_are_not_none(self.config.model[self.DISCRIMINATOR].named_parameters())

        else:
            loss_D = self._train_discriminator_subroutine(X, X_n)

        self.config.optimizer[self.DISCRIMINATOR].zero_grad()

        return loss_D

    def _train_discriminator_subroutine(self, X, X_n):
        pred_X = self.config.model[self.DISCRIMINATOR](X)
        y = torch.Tensor().new_full(size=(X.size(0),), fill_value=self.REAL_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_X = self.config.criterion[self.CROSS_ENTROPY](pred_X.squeeze(1), y)

        with amp.scale_loss(loss_X, self.config.optimizer[self.DISCRIMINATOR]):
            loss_X.backward()

        pred_X_n = self.config.model[self.DISCRIMINATOR](X_n)
        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.FAKE_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_X_n = self.config.criterion[self.CROSS_ENTROPY](pred_X_n, y)

        with amp.scale_loss(loss_X_n, self.config.optimizer[self.DISCRIMINATOR]):
            loss_X_n.backward()

        self.config.optimizer[self.DISCRIMINATOR].step()

        return loss_X_n

    def _evaluate_discriminator_error_on_normalized_data(self, X_n):
        pred_X_n = self.config.model[self.DISCRIMINATOR](X_n)
        y = torch.Tensor().new_full(size=(X_n.size(0),), fill_value=self.REAL_LABEL, dtype=torch.long,
                                    device=self.config.running_config.device)
        loss_D_X_n = self.config.criterion[self.CROSS_ENTROPY](pred_X_n, y)

        return loss_D_X_n

    def _train_segmenter(self, X_s, y, debug=False):
        if debug:
            params_discriminator = [np for np in self.config.model[self.DISCRIMINATOR].named_parameters() if
                                    np[1].requires_grad]
            params_segmenter = [np for np in self.config.model[self.SEGMENTER].named_parameters() if
                                np[1].requires_grad]
            params_normalizer = [np for np in self.config.model[self.NORMALIZER].named_parameters() if
                                 np[1].requires_grad]
            initial_params_discriminator = [(name, p.clone()) for (name, p) in params_discriminator]
            initial_params_segmenter = [(name, p.clone()) for (name, p) in params_segmenter]
            initial_params_normalizer = [(name, p.clone()) for (name, p) in params_normalizer]

            loss_S_X_n = self._train_segmenter_subroutine(X_s, y)

            # Analyze behavior of networks parameters.
            ModelHelper().assert_model_params_not_changed(initial_params_discriminator, params_discriminator)
            ModelHelper().assert_model_params_changed(initial_params_segmenter, params_segmenter)
            ModelHelper().assert_model_params_not_changed(initial_params_normalizer, params_normalizer)
            ModelHelper().assert_model_grads_are_none(self.config.model[self.NORMALIZER].named_parameters())
            ModelHelper().assert_model_grads_are_not_none(self.config.model[self.SEGMENTER].named_parameters())
            ModelHelper().assert_model_grads_are_none(self.config.model[self.DISCRIMINATOR].named_parameters())

        else:
            loss_S_X_n = self._train_segmenter_subroutine(X_s, y)

        return loss_S_X_n

    def _train_segmenter_subroutine(self, X_s, y):
        loss_S_X_s = self.config.criterion[self.DICE_LOSS](torch.nn.functional.softmax(X_s, dim=1),
                                                           to_onehot(y, num_classes=4))
        with amp.scale_loss(loss_S_X_s, self.config.optimizer[self.SEGMENTER]):
            loss_S_X_s.backward(retain_graph=True)  # Retain graph in VRAM for future error addition.
        self.config.optimizer[self.SEGMENTER].step()

        return loss_S_X_s

    def _train_normalizer(self, loss_S_X_n, loss_D_X_n, debug=False):
        if debug:
            params_discriminator = [np for np in self.config.model[self.DISCRIMINATOR].named_parameters() if
                                    np[1].requires_grad]
            params_segmenter = [np for np in self.config.model[self.SEGMENTER].named_parameters() if
                                np[1].requires_grad]
            params_normalizer = [np for np in self.config.model[self.NORMALIZER].named_parameters() if
                                 np[1].requires_grad]
            initial_params_discriminator = [(name, p.clone()) for (name, p) in params_discriminator]
            initial_params_segmenter = [(name, p.clone()) for (name, p) in params_segmenter]
            initial_params_normalizer = [(name, p.clone()) for (name, p) in params_normalizer]

            total_error = self._train_normalizer_subroutine(loss_S_X_n, loss_D_X_n)

            # Analyze behavior of networks parameters.
            ModelHelper().assert_model_params_not_changed(initial_params_discriminator, params_discriminator)
            ModelHelper().assert_model_params_not_changed(initial_params_segmenter, params_segmenter)
            ModelHelper().assert_model_params_changed(initial_params_normalizer, params_normalizer)

            ModelHelper().assert_model_grads_are_not_none(self.config.model[self.NORMALIZER].named_parameters())
            ModelHelper().assert_model_grads_are_not_none(self.config.model[self.SEGMENTER].named_parameters())
            ModelHelper().assert_model_grads_are_none(self.config.model[self.DISCRIMINATOR].named_parameters())
        else:
            total_error = self._train_normalizer_subroutine(loss_S_X_n, loss_D_X_n)

        return total_error

    def _train_normalizer_subroutine(self, loss_S_X_s, loss_D_X_n):
        # Compute total error on Normalizer and apply gradients on Normalizer.
        total_error = loss_S_X_s + self.config.variables.lambda_ * loss_D_X_n
        with amp.scale_loss(total_error, self.config.optimizer[self.NORMALIZER]):
            total_error.backward()
        self.config.optimizer[self.NORMALIZER].step()

        return total_error
