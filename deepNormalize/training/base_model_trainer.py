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
from samitorch.training.model_trainer import ModelTrainer
from samitorch.utils.model_io import load

cudnn.benchmark = True
cudnn.enabled = True

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class DeepNormalizeModelTrainer(ModelTrainer):

    def __init__(self, config, callbacks, class_name):
        super(DeepNormalizeModelTrainer, self).__init__(config, callbacks, class_name)

    def _setup(self):

        assert torch.backends.cudnn.enabled, "Amp requires CuDNN backend to be enabled."

        if self._config.running_config.sync_batch_norm:
            import apex
            print("Using apex synced BN.")
            self._config.model = apex.parallel.convert_syncbn_model(self._config.model)

        # Transfer models on CUDA device.
        self._config.model.cuda(self._config.running_config.local_rank)

        # Scale learning rate based on global batch size
        if self._config.running_config.loss_scale is not "dynamic":
            self._config.optimizer.param_groups[0]["lr"] = self._config.optimizer.param_groups[0]["lr"] * float(
                (self._config.dataloader[0].batch_size * len(
                    self._config.dataloader)) * self._config.running_config.world_size) / float(
                self._config.running_config.loss_scale)

        # Initialize Amp.
        self._config.model, self._config.optimizer = amp.initialize(self._config.model, self._config.optimizer,
                                                                    opt_level=self._config.running_config.opt_level,
                                                                    keep_batchnorm_fp32=self._config.running_config.keep_batch_norm_fp32,
                                                                    loss_scale=self._config.running_config.loss_scale,
                                                                    num_losses=4)

        torch.optim.lr_scheduler.ReduceLROnPlateau(self._config.optimizer, factor=0.1, patience=3, verbose=True)

        # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
        # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
        # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
        # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
        if self._config.running_config.is_distributed:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # model = DDP(model)
            # delay_allreduce delays all communication to the end of the backward pass.
            self._config.model = DDP(self._config.model, delay_allreduce=True)

        self._config.criterion = self._config.criterion.cuda(self._config.running_config.local_rank)

    def restore_from_checkpoint(self, checkpoint_path):
        checkpoint = load(checkpoint_path,
                          map_location=lambda storage, loc: storage.cuda(self._config.running_config.local_rank))
        self._config.model.load_state_dict(checkpoint["state_dict"])

        return checkpoint["epoch"]
