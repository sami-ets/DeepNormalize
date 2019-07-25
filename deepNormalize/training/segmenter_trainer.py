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

import torch

from samitorch.logger.plots import ImagesPlot
from samitorch.inputs.batch import Batch
from samitorch.utils.utils import to_onehot
from samitorch.training.training_strategies import LossCheckpointStrategy
from deepNormalize.inputs.images import SliceType
from deepNormalize.logger.image_slicer import SegmentationSlicer
from deepNormalize.training.base_model_trainer import DeepNormalizeModelTrainer

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class SegmenterTrainer(DeepNormalizeModelTrainer):

    def __init__(self, config, callbacks, class_name):
        super(SegmenterTrainer, self).__init__(config, callbacks, class_name)
        self._saving_strategy = LossCheckpointStrategy(self, "segmenter")
        self._segmented_images_plot = ImagesPlot(self._visdom, "Segmented Images")
        self._slicer = SegmentationSlicer()

    @property
    def saving_strategy(self):
        return self._saving_strategy

    def train_batch(self, batch: Batch):
        # Generate normalized data.
        segmented_batch = self.predict(batch)

        # Loss measures generator's ability to fool the discriminator.
        loss_S_G_X = self.evaluate_loss(segmented_batch.x,
                                        to_onehot(torch.squeeze(batch.y, dim=1).long(), num_classes=4))

        with amp.scale_loss(loss_S_G_X, self._config.optimizer, loss_id=3) as scaled_loss:
            scaled_loss.backward()

        self.step()

        return loss_S_G_X, segmented_batch

    def update_image_plot(self, image):
        image = torch.nn.functional.interpolate(image, scale_factor=5, mode="trilinear", align_corners=True)
        self._segmented_images_plot.update(self._slicer.get_colored_slice(SliceType.AXIAL, image))

    def validate_batch(self, batch: Batch):
        segmented_batch = self.predict(batch)

        loss_S_G_X = self.evaluate_loss(torch.nn.functional.softmax(segmented_batch.x, dim=1),
                                        to_onehot(torch.squeeze(batch.y, dim=1).long(), num_classes=4))

        if self._config.running_config.is_distributed:
            loss_S_G_X = self.reduce_tensor(loss_S_G_X.data)

        return loss_S_G_X, segmented_batch

    def at_epoch_begin(self, epoch_num: torch.Tensor):
        super(SegmenterTrainer, self).at_epoch_begin()
        self.update_learning_rate_plot(epoch_num,
                                       torch.Tensor().new([self._config.optimizer.param_groups[0]['lr']]).cpu())

    def at_epoch_end(self):
        super(SegmenterTrainer, self).at_epoch_end()
        self._saving_strategy(self.validation_loss_gauge.average)
