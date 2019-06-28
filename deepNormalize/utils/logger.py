#  -*- coding: utf-8 -*-
#  Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import os
import torch
import numpy as np
import logging
import sys

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from deepNormalize.config.configurations import LoggerConfiguration


class Logger(object):

    def __init__(self, config: LoggerConfiguration):
        self._config = config
        self._log_path = self._create_folder()
        self._writer = SummaryWriter(self._log_path)
        self._logger = logging.getLogger("DeepNormalize")

        self._logger.setLevel(logging.INFO)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    def log_images(self, inputs: torch.Tensor, targets: torch.Tensor, normalized: torch.Tensor,
                   predictions: torch.Tensor, step: int) -> None:
        sources = {
            'inputs': inputs.detach().cpu().numpy(),
            'targets': targets.detach().cpu().numpy(),
            'normalized': normalized.detach().cpu().numpy(),
            'predictions': predictions.detach().cpu().numpy()
        }

        for name, batch in sources.items():
            for tag, image in self._transform_images(name, batch):
                self._writer.add_image(tag, image, global_step=step, dataformats='HW')

    def log_gradients(self, model: torch.nn.Module, step: int) -> None:
        for name, value in model.named_parameters():
            if value.grad is not None:
                self._writer.add_histogram(name + '/gradients', value.grad.data.cpu().numpy(), step)

    def log_model_parameters(self, model: torch.nn.Module, step: int) -> None:
        for name, value in model.named_parameters():
            self._writer.add_histogram(name, value.data.cpu().numpy(), step)

    def log_stats(self, phase, values: dict, step: int) -> None:
        tag_value = {
            f'{phase} Alpha': values["alpha"],
            f'{phase} Lambda': values["lambda"],
            f'{phase} Segmenter Dice Loss': values["segmenter_loss"],
            f'{phase} Discriminator D_X_n Cross Entropy Loss': values["discriminator_loss_D_X_n"],
            f'{phase} Discriminator D_X Cross Entropy Loss': values["discriminator_loss_D"],
            # f'{phase} Discriminator Accuracy': values["discriminator_accuracy"],
            f'{phase} Segmentation Dice': values["dice_score"],
            f'{phase} Learning rate': values["learning_rate"]
        }

        for tag, value in tag_value.items():
            self._writer.add_scalar(tag, value, step)

    def _transform_images(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize(img)))
        else:
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize(img)))

        return tagged_images

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + 1e-6)

    def _create_folder(self) -> str:
        log_path = os.path.join(self._config.path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(log_path)
        return log_path

    def info(self, message, *args):
        self._logger.info(message, *args)
