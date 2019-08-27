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

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError


class Accuracy(Metric):

    def __init__(self, topk=(1,)):
        super(Accuracy, self).__init__()
        self._topk = topk
        self._num_correct = 0
        self._num_examples = 0
        self._batch_size = None

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0
        super(Accuracy, self).reset()

    def update(self, output):
        y_pred, y = output

        """Computes the precision@k for the specified values of k"""
        maxk = max(self._topk)
        self._batch_size = y.size(0)

        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))

        self._num_correct += torch.sum(correct)
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')

        res = []

        for k in self._topk:
            correct_k = self._num_correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / self._batch_size))
        return res
