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
from samitorch.logger.plots import AccuracyPlot, LossPlot, ParameterPlot
from samitorch.metrics.gauges import RunningAverageGauge
import torch
import torch.backends.cudnn as cudnn
import abc

from samitorch.inputs.batch import PatchBatch
from samitorch.utils.model_io import save, load

cudnn.benchmark = True
cudnn.enabled = True

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class ModelTrainer(object):

    def __init__(self, config, callbacks, class_name, visdom, running_config):
        self._config = config
        self._callbacks = callbacks

        self._visdom = visdom
        self._running_config = running_config

        self.training_metric_gauge = RunningAverageGauge()
        self.training_loss_gauge = RunningAverageGauge()
        self.validation_metric_gauge = RunningAverageGauge()
        self.validation_loss_gauge = RunningAverageGauge()

        self._training_metric_plot = AccuracyPlot(self._visdom,
                                                  "{} Training metric lr={} momentum={}".format(
                                                      class_name,
                                                      config.optimizer.param_groups[0]["lr"],
                                                      config.optimizer.param_groups[0]["momentum"]))
        self._validation_metric_plot = AccuracyPlot(self._visdom,
                                                    "{} Validation metric lr={} momentum={}".format(
                                                        class_name,
                                                        config.optimizer.param_groups[0]["lr"],
                                                        config.optimizer.param_groups[0]["momentum"]))
        self._training_loss_plot = LossPlot(self._visdom, "{} Training loss".format(class_name))
        self._validation_loss_plot = LossPlot(self._visdom, "{} Validation loss".format(class_name))
        self._learning_rate_plot = ParameterPlot(self._visdom, "{} Learning rate".format(class_name),
                                                 "learning rate")

        self._setup()

        self._global_step = torch.Tensor().new_zeros((1,), dtype=torch.int64, device='cpu')
        self._epoch = torch.Tensor().new_zeros((1,), dtype=torch.int64, device='cpu')

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def config(self):
        return self._config

    def predict(self, batch: PatchBatch, detach: bool = False):
        transformed_batch = PatchBatch.from_batch(batch).to_device(self._running_config.device)
        if detach:
            transformed_batch.x = self._config.model(batch.x).detach()
        else:
            transformed_batch.x = self._config.model(batch.x)
        return transformed_batch

    def compute_metric(self):
        metric = self._config.metric.compute()
        self._config.metric.reset()
        return metric

    def evaluate_loss(self, X, y):
        return self._config.criterion(X, y)

    def update_lr(self, new_lr):
        self._config.optimizer.param_groups["lr"] = new_lr

    def update_metric_gauge(self, metric, n_data, phase="training"):
        if phase == "training":
            self.training_metric_gauge.update(metric, n_data)
        else:
            self.validation_metric_gauge.update(metric, n_data)

    def update_loss_gauge(self, loss, n_data, phase="training"):
        if phase == "training":
            self.training_loss_gauge.update(loss, n_data)
        else:
            self.validation_loss_gauge.update(loss, n_data)

    def update_metric_plot(self, step, value, phase="training"):
        if phase == "training":
            self._training_metric_plot.append(step, value)
        else:
            self._validation_metric_plot.append(step, value)

    def update_loss_plot(self, step, value, phase="training"):
        if phase == "training":
            self._training_loss_plot.append(step, value)
        else:
            self._validation_loss_plot.append(step, value)

    def update_learning_rate_plot(self, epoch, value):
        self._learning_rate_plot.append(epoch, value)

    def update_metric(self, pred, y):
        self._config.metric.update((pred, y))

    def step(self):
        self._config.optimizer.step()

    def at_training_begin(self):
        self._initialize_model(self._config.model)

    def at_training_end(self, epoch_num: int):
        save("model.pickle", self._config.model, epoch_num, self._config.optimizer)

    def at_epoch_begin(self):
        self._config.model.train()

    def at_epoch_end(self):
        self._epoch += 1
        self.training_metric_gauge.reset()
        self.training_loss_gauge.reset()

    def at_iteration_begin(self):
        self._config.optimizer.zero_grad()

    def at_iteration_end(self):
        self._global_step += 1
        self.enable_gradients()

    def at_validation_begin(self):
        self._config.model.eval()

    def at_validation_end(self):
        self.validation_metric_gauge.reset()
        self.validation_loss_gauge.reset()

    @abc.abstractmethod
    def finalize(self):
        raise NotImplementedError()

    def disable_gradients(self):
        for p in self._config.model.parameters(): p.requires_grad = False

    def enable_gradients(self):
        for p in self._config.model.parameters(): p.requires_grad = True

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= int(self._running_config.world_size)
        return rt

    def reset_optimizer(self):
        self._config.optimizer.zero_grad()

    @staticmethod
    def _initialize_model(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @abc.abstractmethod
    def _setup(self):
        raise NotImplementedError()


class DeepNormalizeModelTrainer(ModelTrainer):

    def __init__(self, config, callbacks, class_name, visdom, running_config):
        super(DeepNormalizeModelTrainer, self).__init__(config, callbacks, class_name, visdom, running_config)

    def _setup(self):

        assert torch.backends.cudnn.enabled, "Amp requires CuDNN backend to be enabled."

        if self._running_config.sync_batch_norm:
            import apex
            print("Using apex synced BN.")
            self._config.model = apex.parallel.convert_syncbn_model(self._config.model)

        # Transfer models on CUDA device.
        self._config.model.cuda(self._running_config.local_rank)

        # Scale learning rate based on global batch size
        if self._running_config.loss_scale is not "dynamic":
            self._config.optimizer.param_groups[0]["lr"] = self._config.optimizer.param_groups[0]["lr"] * float(
                (self._config.dataloader[0].batch_size * len(
                    self._config.dataloader)) * self._running_config.world_size) / float(
                self._running_config.loss_scale)

        # Initialize Amp.
        self._config.model, self._config.optimizer = amp.initialize(self._config.model, self._config.optimizer,
                                                                    opt_level=self._running_config.opt_level,
                                                                    keep_batchnorm_fp32=self._running_config.keep_batch_norm_fp32,
                                                                    loss_scale=self._running_config.loss_scale,
                                                                    num_losses=4)

        torch.optim.lr_scheduler.ReduceLROnPlateau(self._config.optimizer, factor=0.1, patience=3, verbose=True)

        # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
        # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
        # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
        # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
        if self._running_config.is_distributed:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # model = DDP(model)
            # delay_allreduce delays all communication to the end of the backward pass.
            self._config.model = DDP(self._config.model, delay_allreduce=True)

        self._config.criterion = self._config.criterion.cuda(self._running_config.local_rank)

    def restore_from_checkpoint(self, checkpoint_path):
        checkpoint = load(checkpoint_path,
                          map_location=lambda storage, loc: storage.cuda(self._running_config.local_rank))
        self._config.model.load_state_dict(checkpoint["state_dict"])
        self._config.optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint["epoch"]
