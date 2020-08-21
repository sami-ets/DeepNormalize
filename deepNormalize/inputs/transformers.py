import random

import numpy as np
import torch
from samitorch.inputs.sample import Sample


class AddBiasField(object):

    def __init__(self, exec_probability: float, alpha: float = None):
        self._exec_probability = exec_probability
        self._alpha = alpha

    def __call__(self, inputs):
        if random.uniform(0, 1) <= self._exec_probability:
            if isinstance(inputs, np.ndarray):

                if self._alpha is None:
                    self._alpha = np.random.normal(0, 1)

                x = np.linspace(1 - max(0.5, self._alpha), 1 + self._alpha, inputs.shape[1])
                y = np.linspace(1 - max(0.5, self._alpha), 1 + self._alpha, inputs.shape[2])
                z = np.linspace(1 - max(0.5, self._alpha), 1 + self._alpha, inputs.shape[3])
                [X, Y, Z] = np.meshgrid(x, y, z)
                bias = np.multiply(X, Y, Z).transpose(1, 0, 2)
                bias = np.expand_dims(bias, 0)

                inputs = inputs * bias

                return inputs
        else:
            return inputs
