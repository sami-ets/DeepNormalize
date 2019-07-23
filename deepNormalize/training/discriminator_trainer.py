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
import numpy as np

from deepNormalize.training.base_model_trainer import DeepNormalizeModelTrainer

from samitorch.inputs.batch import Batch

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class DiscriminatorTrainer(DeepNormalizeModelTrainer):

    def __init__(self, config, callbacks, class_name):
        super(DiscriminatorTrainer, self).__init__(config, callbacks, class_name)

    def train_batch(self, batch: Batch, generated_batch: Batch):
        # Measure discriminator's ability to classify real from generated samples
        pred_X = self.predict(batch)
        loss_D_X = self.evaluate_loss(torch.nn.functional.log_softmax(pred_X.x, dim=1), batch.dataset_id.long())
        pred_X.to_device('cpu')

        pred_G_X = self.predict(generated_batch)
        choices = np.random.choice(a=pred_G_X.x.size(0), size=(int(pred_G_X.x.size(0)/2), ), replace=True)
        pred_G_X.x = pred_G_X.x[choices]
        fake_ids = torch.Tensor().new_full(size=(int(generated_batch.x.size(0) / 2),),
                                           fill_value=2,
                                           dtype=torch.int8,
                                           device=self._config.running_config.device)
        pred_G_X.dataset_id = fake_ids
        loss_D_G_X = self.evaluate_loss(torch.nn.functional.log_softmax(pred_G_X.x.detach(), dim=1),
                                        pred_G_X.dataset_id.long())
        pred_G_X.to_device('cpu')

        loss_D = ((loss_D_X + loss_D_G_X) / 2.0)

        with amp.scale_loss(loss_D, self._config.optimizer) as scaled_loss:
            scaled_loss.backward()

        self.step()

        return loss_D, pred_X, pred_G_X

    def validate_batch(self, batch: Batch, generated_batch: Batch):
        pred_X = self.predict(batch)
        loss_D_X = self.evaluate_loss(pred_X.x, batch.dataset_id.long())
        pred_X.to_device('cpu')

        pred_G_X = self.predict(generated_batch)
        choices = np.random.choice(a=pred_G_X.x.size(0), size=(int(pred_G_X.x.size(0) / 2),), replace=True)
        pred_G_X.x = pred_G_X.x[choices]
        fake_ids = torch.Tensor().new_full(size=(batch.x.size(0) / 2,),
                                           fill_value=2,
                                           dtype=torch.int8,
                                           device=self._config.running_config.device)
        pred_G_X.dataset_id = fake_ids
        loss_D_G_X = self.evaluate_loss(torch.nn.functional.softmax(pred_G_X.x.detach(), dim=1),
                                        pred_G_X.dataset_id.long())
        pred_G_X.to_device('cpu')

        loss_D = (loss_D_X + loss_D_G_X) / 2.0

        if self._config.running_config.is_distributed:
            loss_D = self.reduce_tensor(loss_D.data)

        return loss_D, pred_X, pred_G_X

    def at_epoch_begin(self, epoch_num: torch.Tensor):
        super(DiscriminatorTrainer, self).at_epoch_begin()
        self.update_learning_rate_plot(epoch_num,
                                       torch.Tensor().new([self._config.optimizer.param_groups[0]['lr']]).cpu())
