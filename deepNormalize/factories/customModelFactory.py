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

from kerosene.models.models import ModelFactory

from deepNormalize.config.types import ModelType
from deepNormalize.models.resnet3d import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from deepNormalize.models.unet3d import Unet
from deepNormalize.models.dcgan3d import DCGAN


class CustomModelFactory(ModelFactory):
    def __init__(self):
        pass

    def create(self, model_type, params):
        if model_type == ModelType.UNET3D:
            return Unet(**params)
        elif model_type == ModelType.DCGAN:
            return DCGAN(**params)
        elif model_type == ModelType.RESNET_18:
            return ResNet18(params)
        elif model_type == ModelType.RESNET_34:
            return ResNet34(params)
        elif model_type == ModelType.RESNET_50:
            return ResNet50(params)
        elif model_type == ModelType.RESNET_101:
            return ResNet101(params)
        elif model_type == ModelType.RESNET_152:
            return ResNet152(params)
        else:
            raise NotImplementedError("The given model type ({}) is not implemented.".format(type))
