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

import copy

from samitorch.configs.configurations import Configuration


class ConfigAdapter(object):

    def __init__(self, config: Configuration):
        self._config = config

    def adapt(self, model_position: int, criterion_position: int):
        config = copy.copy(self._config)
        config.criterion = self._config.criterion[criterion_position]
        config.metric = self._config.metric[criterion_position]
        config.model = self._config.model[model_position]
        config.optimizer = self._config.optimizer[model_position]

        return config
