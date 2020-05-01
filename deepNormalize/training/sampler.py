import math
import numpy as np
import torch

from deepNormalize.utils.constants import AUGMENTED_INPUTS, NON_AUGMENTED_INPUTS, IMAGE_TARGET, DATASET_ID


class Sampler(object):

    def __init__(self, keep_augmented_prob: float):
        self._keep_augmented_prob = keep_augmented_prob

    def __call__(self, inputs, targets):
        augmented_choices = np.random.choice(np.arange(0, len(inputs[AUGMENTED_INPUTS])),
                                             math.ceil(len(inputs[AUGMENTED_INPUTS]) * self._keep_augmented_prob),
                                             replace=False)
        augmented_inputs = inputs[AUGMENTED_INPUTS][augmented_choices]
        augmented_targets = targets[IMAGE_TARGET][augmented_choices]
        augmented_target_ids = targets[DATASET_ID][augmented_choices]

        non_augmented_choices = np.setdiff1d(np.arange(0, len(inputs[NON_AUGMENTED_INPUTS])), augmented_choices)
        non_augmented_inputs = inputs[NON_AUGMENTED_INPUTS][non_augmented_choices]
        non_augmented_targets = targets[IMAGE_TARGET][non_augmented_choices]
        non_augmented_target_ids = targets[DATASET_ID][non_augmented_choices]

        new_inputs_ = torch.cat((augmented_inputs, non_augmented_inputs))
        new_image_targets = torch.cat((augmented_targets, non_augmented_targets))
        new_target_ids = torch.cat((augmented_target_ids, non_augmented_target_ids))

        new_targets_ = [new_image_targets, new_target_ids]

        new_inputs = [inputs[NON_AUGMENTED_INPUTS], new_inputs_]
        new_targets = [targets, new_targets_]

        return new_inputs, new_targets
