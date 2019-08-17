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

import numpy as np
import copy

from samitorch.inputs.batch import PatchBatch


def split_dataset(dataset, split):
    validation_dataset = copy.copy(dataset)
    idx_full = np.arange(len(dataset))

    np.random.shuffle(idx_full)

    if isinstance(split, int):
        assert split > 0
        assert split < len(dataset), "Validation set size is configured to be larger than entire dataset."
        len_valid = split
    else:
        len_valid = int(len(dataset) * split)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))

    dataset.slices = dataset.slices[train_idx]
    dataset.num_patches = dataset.slices.shape[0]

    validation_dataset.slices = validation_dataset.slices[valid_idx]
    validation_dataset.num_patches = validation_dataset.slices.shape[0]

    return dataset, validation_dataset


def build_configurations(config, num):
    pass


def concat_batches(batch_0, batch_1):
    return PatchBatch(batch_0.samples + batch_1.samples)
