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

import abc

from samitorch.training.training_strategies import TrainingStrategy
from samitorch.utils.model_io import save
from samitorch.training.model_trainer import ModelTrainer
from datetime import datetime


class GANStrategy(TrainingStrategy):
    def __init__(self, trainer):
        super(GANStrategy, self).__init__(trainer)

    def __call__(self, epoch_num: int):
        if 0 <= epoch_num < 15:
            self._trainer.with_segmentation = False

        else:
            self._trainer.with_segmentation = True


class AutoEncoderStrategy(TrainingStrategy):
    def __init__(self, trainer):
        super(AutoEncoderStrategy, self).__init__(trainer)

    def __call__(self, epoch_num: int):
        if 0 <= epoch_num < 5:
            self._trainer.with_autoencoder = True

        else:
            self._trainer.with_autoencoder = False