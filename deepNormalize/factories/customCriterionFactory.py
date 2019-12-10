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

from typing import Union

import numpy as np
import torch
from kerosene.configs.configs import RunConfiguration
from kerosene.nn.criterions import CriterionFactory, CriterionType

from deepNormalize.nn.criterions import MeanLoss


class CustomCriterionFactory(CriterionFactory):
    def __init__(self, run_config: RunConfiguration):
        super(CustomCriterionFactory, self).__init__()
        self._run_config = run_config
        self.register("MeanLoss", MeanLoss)

    def create(self, criterion_type: Union[str, CriterionType], params):
        if criterion_type == CriterionType.DiceLoss or criterion_type == "DiceLoss" or criterion_type == CriterionType.TverskyLoss or criterion_type == "TverskyLoss":
            if params["weight"] is not None:
                params["weight"] = torch.Tensor().new_tensor(np.array(params["weight"]),
                                                             dtype=torch.float,
                                                             device=self._run_config.device,
                                                             requires_grad=False)

        return self._criterion[str(criterion_type)](**params) if params is not None else self._criterion[
            str(criterion_type)]()
