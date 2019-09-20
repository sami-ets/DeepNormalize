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


def to_html(classe_names, metric_names, metric_values):
    arr_tuple = tuple(([metric_values[i] for i in range(len(metric_values))]))
    interleaved = np.vstack(arr_tuple).reshape((-1,), order='F')

    class_row = "".join(["<th colspan='{}'> {} </th>".format(str(len(metric_names)),
                                                             str(class_name)) for class_name in classe_names])

    metric_row = "".join(["<th> {} </th>".format(str(metric_name)) for metric_name in metric_names]) * len(classe_names)

    metric_values_row = "".join(["<td> {} </td>".format(str(metric_value)) for metric_value in interleaved])

    header = "<!DOCTYPE html> \
             <html> \
             <head> \
             <style> \
             table { \
               font-family: arial, sans-serif;\
               border-collapse: collapse;\
               width: 100%;\
             } \
             td, th { \
               border: 1px solid #dddddd; \
               text-align: left; \
               padding: 8px; \
             } \
             tr:nth-child(even) { \
               background-color: #dddddd; \
             } \
             </style> \
             </head> \
             <body> \
             <h2>Metric Table</h2> \
             <table> \
              <caption>Metrics</caption>"

    html = header + \
           "<tr>" + class_row + "</tr>" + \
           "<tr>" + metric_row + "</tr>" + \
           "<tr>" + metric_values_row + "</tr>" + \
           "</table></body></html>"

    return html
