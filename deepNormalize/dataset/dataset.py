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

import os
import re

from torch.utils.data.dataset import Dataset


class iSEGDataset(Dataset):

    def __init__(self):
        pass

    def _get_files(self, path):
        subjects = list()
        keys = ["t1", "t2", "roi", "label"]

        for (dirpath, dirnames, filenames) in os.walk(path):
            if len(filenames) is not 0:
                # Filter files.
                t1 = list(filter(re.compile(r"^.*?T1.nii$").search, filenames))
                t2 = list(filter(re.compile(r"^.*?T2.nii$").search, filenames))
                roi = list(filter(re.compile(r"^.*?ROIT1.nii.gz$").search, filenames))
                seg_training = list(filter(re.compile(r"^.*?labels.nii$").search, filenames))

                t1 = [os.path.join(dirpath, ("{}".format(i))) for i in t1]
                t2 = [os.path.join(dirpath, ("{}".format(i))) for i in t2]
                roi = [os.path.join(dirpath, ("{}".format(i))) for i in roi]
                seg_training = [os.path.join(dirpath, ("{}".format(i))) for i in seg_training]

                subjects.append(dict((key, volume) for key, volume in zip(keys, [t1,
                                                                                 t2,
                                                                                 roi,
                                                                                 seg_training])))

        return subjects
