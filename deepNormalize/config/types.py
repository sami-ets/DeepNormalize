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

from enum import Enum


class ModelType(Enum):
    RESNET_18 = "ResNet18"
    RESNET_34 = "ResNet34"
    RESNET_50 = "ResNet50"
    RESNET_101 = "ResNet101"
    RESNET_152 = "ResNet152"
    UNET3D = "UNet3D"
    DCGAN = "DCGAN"

    ALL = [RESNET_18, RESNET_34, RESNET_50, RESNET_101, RESNET_152, UNET3D, DCGAN]

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        else:
            return False


