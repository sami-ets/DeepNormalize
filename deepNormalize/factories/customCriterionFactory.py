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

from kerosene.nn.criterions import CriterionFactory

from deepNormalize.nn.criterion import MeanLoss


class CustomCriterionFactory(CriterionFactory):
    def __init__(self):
        super(CustomCriterionFactory, self).__init__()
        self.register("MeanLoss", MeanLoss)
