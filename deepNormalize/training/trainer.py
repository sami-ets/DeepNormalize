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

import logging

import torch
import os
import abc
import numpy as np
from typing import List, Optional
from samitorch.callbacks.callbacks import Callback
from samitorch.inputs.batch import Batch
from samitorch.logger.plots import ImagesPlot

from deepNormalize.logger.image_slicer import AdaptedImageSlicer
from deepNormalize.inputs.images import SliceType

from deepNormalize.training.training_strategies import GANStrategy, AutoEncoderStrategy
from deepNormalize.utils.utils import concat_batches
from deepNormalize.training.generator_trainer import GeneratorTrainer
from deepNormalize.training.discriminator_trainer import DiscriminatorTrainer
from deepNormalize.training.segmenter_trainer import SegmenterTrainer
from deepNormalize.logger.plots import PiePlot

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this project.")


class Trainer(object):
    def __init__(self, config: list, callbacks: Optional[List[Callback]]):
        """Class initializer.

        Args:
            config (:obj:`samitorch.training.training_config.TrainingConfig`): A TrainingConfig containing training configuration.
            callbacks (:obj:`list` of :obj:`Callback`): A list of Callback objects to register.
        """
        assert config is not None, "Training must at least have a list of one configuration."
        self._config = config

        if callbacks is not None:
            self._callbacks = list()
            for cbck in callbacks:
                self._register_callback(cbck)

        self._global_step = torch.Tensor().new_zeros((1,), dtype=torch.int64, device='cpu')
        self._epoch = torch.Tensor().new_zeros((1,), dtype=torch.int64, device='cpu')

    @property
    def global_step(self):
        """int: The current epoch count."""
        return self._global_step

    @global_step.setter
    def global_step(self, global_step):
        self._global_step = global_step

    @property
    def epoch(self):
        """int: The current epoch count."""
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def config(self):
        """:obj:`list` of :obj:`TrainingConfig`: A list of registered training configuration, one per model."""
        return self._config

    @property
    def callbacks(self):
        """:obj:`list` of :obj:`Callback`: A list of registered callbacks."""
        return self._callbacks

    @abc.abstractmethod
    def train(self):
        """Main training loop.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _train_epoch(self, epoch_num: int, *args, **kwargs):
        """Train a model for one epoch.

        Args:
            epoch_num (int): current epoch number.
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _train_batch(self, batch: Batch, *args, **kwargs):
        """Function which handles prediction from batch, logging, loss calculation and optimizer step.

        This function is needed for the framework to provide a generic trainer function which works with all kind of
        networks and loss functions. The closure function must implement all steps from forwarding, over loss
        calculation, metric calculation, logging, and the actual backpropagation. It is called with an empty
        optimizer-dict to evaluate and should thus work with optional optimizers.

        Args:
            batch (:obj:`samitorch.inputs.batch.Batch`): Batch object containing the data.
            *args (dict): additional arguments.
            **kwargs (dict): additional keyword arguments.

        Returns:
            dict: Metric values (with same keys as input dict metrics).
            dict: Loss values (with same keys as input dict criterions).
            list: Arbitrary number of predictions.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @staticmethod
    def _prepare_batch(batch: Batch, input_device: torch.device, output_device: torch.device):
        """Converts a numpy batch of data and labels to suitable datatype and pushes them to correct devices

        Args
            batch (dict): dictionary containing the batch (must have keys 'data' and 'label'
            input_device (:obj:`torch.device`): device for network inputs
            output_device (:obj:`torch.device`): device for network outputs

        Returns:
            dict: dictionary containing all necessary data in right format and type and on the correct device

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_epoch(self, epoch_num: int, *args, **kwargs):
        """Run validation phase.

         Args:
            epoch_num (int): Current epoch number.
            kwargs: keyword arguments

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_batch(self, batch: Batch, *args, **kwargs):
        """ Run validation over a batch.

        Args:
            batch (:obj:`samitorch.inputs.batch.Batch`): An input Batch object,
            **kwargs: additional keywords arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _finalize(self, *args, **kwargs):
        """Finalize all operations (e.g. save checkpoint, finalize Data Loaders and close logger if required).

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: if not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _setup(self, *args, **kwargs):
        """Defines the actual Trainer Setup.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_training_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of the training.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_training_end(self, *args, **kwargs):
        """Defines the behaviour at the end of the training.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_epoch_begin(self, *args, **kwargs):
        """Defines the behaviour at beginnig of each epoch.

        Args:
            *args: positional arguments
            **kwargs (dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _at_epoch_end(self, *args, **kwargs):
        """
        Defines the behaviour at the end of each epoch.

        Args:
            *args: positional arguments
            **kwargs(dict): additional keyword arguments.

        Raises:
            NotImplementedError: If not overwritten by subclass.
        """

        raise NotImplementedError()

    @staticmethod
    def _is_better_val_scores(old_val_score: float, new_val_score: float, mode='highest'):
        """Check whether the new val score is better than the old one with respect to the optimization goal.

        Args:
            old_val_score (float): old validation score
            new_val_score (float): new validation score
            mode (str): String to specify whether a higher or lower validation score is optimal;
                must be in ['highest', 'lowest']

        Returns:
            bool: True if new score is better, False otherwise.
        """

        assert mode in ['highest', 'lowest'], "Invalid Comparison Mode"

        if mode == 'highest':
            return new_val_score > old_val_score
        elif mode == 'lowest':
            return new_val_score < old_val_score

    def _register_callback(self, callback: Callback):
        """Register Callback to Trainer.

        Args:
            callback (:class:`Callback`): the callback to register

        Raises:
            AssertionError: `callback` is not an instance of :class:`Callback` and has not both methods
            ['at_epoch_begin', 'at_epoch_end'].
        """

        assertion_str = "Given callback is not valid; Must be instance of " \
                        "Callback or provide functions " \
                        "'at_epoch_begin' and 'at_epoch_end'"

        assert \
            isinstance(callback, Callback) \
            or (hasattr(callback, "at_epoch_begin")
                and hasattr(callback, "at_epoch_end")), assertion_str

        self._callbacks.append(callback)


class DeepNormalizeTrainer(Trainer):
    LOGGER = logging.getLogger("DeepNormalizeTrainer")

    def __init__(self, config, callbacks, visdom, running_config, logger_config, pretraining_config, training_config,
                 dataloaders):
        super(DeepNormalizeTrainer, self).__init__(config, callbacks)
        self._visdom = visdom
        self._running_config = running_config
        self._logger_config = logger_config
        self._pretraining_config = pretraining_config
        self._training_config = training_config
        self._dataloaders = dataloaders

        # Create instance of every model trainer.
        self._discriminator_trainer = DiscriminatorTrainer(config[2], callbacks, "Discriminator", visdom,
                                                           self._running_config)
        self._generator_trainer = GeneratorTrainer(config[0], callbacks, self._discriminator_trainer,
                                                   "Generator", visdom, self._running_config)
        self._segmenter_trainer = SegmenterTrainer(config[1], callbacks, "Segmenter", visdom, self._running_config)

        # Various plots.
        self._input_images_plot = ImagesPlot(self._visdom, "Input Image")
        self._class_pie_plot = PiePlot(self._visdom,
                                       torch.Tensor().new_zeros(size=(3,)),
                                       ["ISEG", "MRBrainS", "Fake"],
                                       "Predicted classes")

        # Define training behavior.
        self._with_segmentation = False
        self._with_autoencoder = False

        if self._pretraining_config.pretrained:
            generator_epoch, discriminator_epoch, segmenter_epoch = 0, 0, 0
            if "generator" in self._pretraining_config.model_paths:
                generator_epoch = self._generator_trainer.restore_from_checkpoint(
                    self._pretraining_config.model_paths["generator"])
            if "discriminator" in self._pretraining_config.model_paths:
                discriminator_epoch = self._discriminator_trainer.restore_from_checkpoint(
                    self._pretraining_config.model_paths["discriminator"])
            if "segmenter" in self._pretraining_config.model_paths:
                segmenter_epoch = self._segmenter_trainer.restore_from_checkpoint(
                    self._pretraining_config.model_paths["segmenter"])
            # Get the highest epoch from checkpoints to restart training.
            epochs = np.array([generator_epoch, discriminator_epoch, segmenter_epoch])
            max_epoch = epochs.max()
            self._epoch = torch.Tensor().new([max_epoch])

            # Use a barrier() to make sure that all processes have finished reading the checkpoints.
            torch.distributed.barrier()

        self._slicer = AdaptedImageSlicer()

        self._gan_strategy = GANStrategy(self)
        self._autoencoder_strategy = AutoEncoderStrategy(self)

    @property
    def with_segmentation(self):
        return self._with_segmentation

    @with_segmentation.setter
    def with_segmentation(self, with_segmentation):
        self._with_segmentation = with_segmentation

    @property
    def with_autoencoder(self):
        return self._with_autoencoder

    @with_autoencoder.setter
    def with_autoencoder(self, with_autoencoder):
        self._with_autoencoder = with_autoencoder

    def train(self):
        for epoch in range(self._training_config.max_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)

    def _train_epoch(self, epoch_num: int):
        self._at_epoch_begin(epoch_num)

        for iteration, (batch_d0, batch_d1) in enumerate(zip(self._dataloaders[0], self._dataloaders[1]),
                                                         0):

            self._at_iteration_begin()

            # Make one big batch from both data loader.
            batch = concat_batches(batch_d0, batch_d1)

            batch = self._prepare_batch(batch=batch,
                                        input_device=torch.device('cpu'),
                                        output_device=self._running_config.device)

            loss_D, loss_D_G_X_as_X, custom_loss, loss_S_G_X, generated_batch, segmented_batch, mse_loss = self._train_batch(
                batch)

            if iteration % self._logger_config.frequency == 0:
                dsc = torch.Tensor().new_zeros(size=(1,)).cuda()
                acc = torch.Tensor().new_zeros(size=(1,)).cuda()

                self.update_image_plot(batch.x.cpu().data)

                if self._with_segmentation:
                    dsc = self._segmenter_trainer.compute_metric().cuda()

                if not self._with_autoencoder:
                    acc = torch.Tensor().new([self._discriminator_trainer.compute_metric()]).cuda()

                # Average loss and accuracy across processes for logging
                if self._running_config.is_distributed:
                    if self._with_segmentation:
                        loss_S_G_X = self._reduce_tensor(loss_S_G_X.data)
                        dsc = self._reduce_tensor(dsc.data)

                    loss_D_G_X_as_X = self._reduce_tensor(loss_D_G_X_as_X.data)
                    loss_D = self._reduce_tensor(loss_D.data)
                    mse_loss = self._reduce_tensor(mse_loss.data)
                    acc = self._reduce_tensor(acc.data)

                torch.cuda.synchronize()

                self._generator_trainer.update_image_plot(generated_batch.x.cpu().data)

                if self._running_config.local_rank == 0:

                    if self._with_autoencoder:
                        self._generator_trainer.update_loss_gauge(mse_loss.item(), batch.x.size(0))
                        self._generator_trainer.update_loss_plot(self.global_step, torch.Tensor().new(
                            [self._generator_trainer.training_loss_gauge.average]))

                    else:
                        self._generator_trainer.update_loss_gauge(loss_D_G_X_as_X.item(), batch.x.size(0))
                        self._generator_trainer.update_loss_plot(self.global_step, torch.Tensor().new(
                            [self._generator_trainer.training_loss_gauge.average]))
                        self._discriminator_trainer.update_loss_gauge(loss_D.item(), batch.x.size(0))
                        self._discriminator_trainer.update_loss_plot(self.global_step, torch.Tensor().new(
                            [self._discriminator_trainer.training_loss_gauge.average]))
                        self._discriminator_trainer.update_metric_gauge(acc.item(), batch.x.size(0))
                        self._discriminator_trainer.update_metric_plot(self.global_step, torch.Tensor().new(
                            [self._discriminator_trainer.training_metric_gauge.average]))

                        if self._with_segmentation:
                            self._segmenter_trainer.update_loss_gauge(loss_S_G_X.item(), batch.x.size(0))
                            self._segmenter_trainer.update_loss_plot(self.global_step, torch.Tensor().new(
                                [self._segmenter_trainer.training_loss_gauge.average]))
                            self._segmenter_trainer.update_metric_gauge(dsc.item(), batch.x.size(0))
                            self._segmenter_trainer.update_metric_plot(self.global_step, torch.Tensor().new(
                                [self._segmenter_trainer.training_metric_gauge.average]))
                            self._segmenter_trainer.update_image_plot(
                                torch.argmax(segmented_batch.x, dim=1, keepdim=True).float().cpu().data)

                    self.LOGGER.info(
                        "Epoch: {} Step: {} Generator loss: {}Discriminator loss: {} Segmenter loss: {} Segmenter Dice Score: {}".format(
                            epoch_num,
                            iteration,
                            self._generator_trainer.training_loss_gauge.average,
                            self._discriminator_trainer.training_loss_gauge.average,
                            self._segmenter_trainer.training_loss_gauge.average,
                            self._segmenter_trainer.training_metric_gauge.average))

            del batch
            del generated_batch
            del segmented_batch

            # Re-enable all parts of the graph.
            self._at_iteration_end()
        self._at_epoch_end()

    def _train_batch(self, batch: Batch):
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

    def _validate_epoch(self, epoch_num: int):
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
                    loss_D_G_X_as_X = self._generator_trainer.evaluate_discriminator_error_on_normalized_data(
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

    @staticmethod
    def _prepare_batch(batch: Batch, input_device: torch.device, output_device: torch.device):
        if not batch.device == input_device:
            raise ValueError("Data must be in CPU Memory but is on {} device".format(batch.device))
        return batch.to_device(output_device)

    def _at_epoch_begin(self, epoch_num):
        self._gan_strategy(self.epoch.item())
        self._autoencoder_strategy(self.epoch.item())
        self._generator_trainer.at_epoch_begin(self.epoch)
        self._discriminator_trainer.at_epoch_begin(self.epoch)
        self._segmenter_trainer.at_epoch_begin(self.epoch)

    def _at_epoch_end(self):
        self._generator_trainer.at_epoch_end()
        self._discriminator_trainer.at_epoch_end()
        self._segmenter_trainer.at_epoch_end()

    def _at_validation_begin(self):
        self._generator_trainer.at_validation_begin()
        self._discriminator_trainer.at_validation_begin()
        self._segmenter_trainer.at_validation_begin()

    def _at_validation_end(self):
        self._generator_trainer.update_loss_plot(self.epoch, torch.Tensor().new(
            [self._generator_trainer.validation_loss_gauge.average]), phase="validation")
        self._discriminator_trainer.update_loss_plot(self.epoch, torch.Tensor().new(
            [self._discriminator_trainer.validation_loss_gauge.average]), phase="validation")
        self._segmenter_trainer.update_loss_plot(self.epoch, torch.Tensor().new(
            [self._segmenter_trainer.validation_loss_gauge.average]), phase="validation")
        self._generator_trainer.validation_metric_gauge.reset()
        self._discriminator_trainer.validation_metric_gauge.reset()
        self._segmenter_trainer.validation_metric_gauge.reset()
        self._generator_trainer.validation_loss_gauge.reset()
        self._discriminator_trainer.validation_loss_gauge.reset()
        self._segmenter_trainer.validation_loss_gauge.reset()
        self._epoch += 1

    def _at_iteration_begin(self):
        self._generator_trainer.at_iteration_begin()
        self._segmenter_trainer.at_iteration_begin()
        self._discriminator_trainer.at_iteration_begin()

    def _at_iteration_end(self):
        self._generator_trainer.at_iteration_end()
        self._segmenter_trainer.at_iteration_end()
        self._discriminator_trainer.at_iteration_end()
        self._global_step += 1

    def _validate_batch(self, batch: Batch, **kwargs):
        pass

    def _finalize(self, *args, **kwargs):
        pass

    def _setup(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def _at_training_begin(self, *args, **kwargs):
        pass

    def _at_training_end(self, *args, **kwargs):
        pass

    @staticmethod
    def _reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= int(os.environ['WORLD_SIZE'])
        return rt

    def update_image_plot(self, image):
        image = torch.nn.functional.interpolate(image, scale_factor=5, mode="trilinear", align_corners=True)
        self._input_images_plot.update(self._slicer.get_slice(SliceType.AXIAL, image))

    def count(self, tensor, n_classes):
        count = torch.Tensor().new_zeros(size=(n_classes,))

        for i in range(n_classes):
            count[i] = torch.sum(tensor == i).int()

        return count
