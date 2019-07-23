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


class PiePlot(object):

    def __init__(self, visdom, X, legend, title):
        self._title = title
        self._legend = legend
        self._visdom = visdom
        self._window = self._visdom.pie(X=X.cpu(),
                                        opts=dict(
                                            title=title,
                                            legend=legend))

    def update(self, sizes):
        self._visdom.pie(X=sizes.cpu(),
                         win=self._window,
                         opts=dict(
                             title=self._title,
                             legend=self._legend,
                             explode=[0, 0, 0.1]))
