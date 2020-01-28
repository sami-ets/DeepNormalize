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

import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


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
             tr:nth-child(odd) { \
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


def to_html_per_dataset(classe_names, metric_names, metric_values, datasets):
    arr_tuple_1 = tuple(([metric_values[0][i] for i in range(len(metric_values)-1)]))
    interleaved = np.vstack(arr_tuple_1).reshape((-1,), order='F')
    arr_tuple_2 = tuple(([metric_values[1][i] for i in range(len(metric_values)-1)]))
    interleaved_2 = np.vstack(arr_tuple_2).reshape((-1,), order='F')
    arr_tuple_3 = tuple(([metric_values[2][i] for i in range(len(metric_values)-1)]))
    interleaved_3 = np.vstack(arr_tuple_3).reshape((-1,), order='F')

    class_row = "".join(["<th colspan='{}'> {} </th>".format(str(len(metric_names)),
                                                             str(class_name)) for class_name in classe_names])

    metric_row = "".join(["<th> {} </th>".format(str(metric_name)) for metric_name in metric_names]) * len(classe_names)

    metric_values_row = "".join(["<td> {} </td>".format(str(metric_value)) for metric_value in interleaved])
    metric_values_row_2 = "".join(["<td> {} </td>".format(str(metric_value)) for metric_value in interleaved_2])
    metric_values_row_3 = "".join(["<td> {} </td>".format(str(metric_value)) for metric_value in interleaved_3])

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
             tr:nth-child(odd) { \
               background-color: #dddddd; \
             } \
             </style> \
             </head> \
             <body> \
             <h2>Per-Dataset Metric Table</h2> \
             <table> \
              <caption>Metrics</caption>"

    html = header + \
           "<tr>" + "<th> </th>" + class_row + "</tr>" + \
           "<tr>" + "<th> </th>" + metric_row + "</tr>" + \
           "<tr>" + "<td> {} </td>".format(datasets[0]) + metric_values_row + "</tr>" + \
           "<tr>" + "<td> {} </td>".format(datasets[1]) + metric_values_row_2 + "</tr>" + \
           "<tr>" + "<td> {} </td>".format(datasets[2]) + metric_values_row_3 + "</tr>" + \
           "</table></body></html>"

    return html


def to_html_JS(data_types, metric_names, metric_values):
    metric_name_row = "".join("<th> {} </th>".format(metric_names[0]))

    metric_values_row = "".join("<td> {} </td>".format(str(metric_values[0])))
    metric_values_row_2 = "".join("<td> {} </td>".format(str(metric_values[1])))

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
                tr:nth-child(odd) { \
                  background-color: #dddddd; \
                } \
                </style> \
                </head> \
                <body> \
                <h2>Jensen-Shannon Metric Table</h2> \
                <table> \
                 <caption>Metrics</caption>"

    html = header + \
           "<tr>" + "<th> </th>" + metric_name_row + "</tr>" + \
           "<tr>" + "<td> {} </td>".format(data_types[0]) + metric_values_row + "</tr>" + \
           "<tr>" + "<td> {} </td>".format(data_types[1]) + metric_values_row_2 + "</tr>" + \
           "</table></body></html>"

    return html


def to_html_time(time):
    days = str(time).split(',')[0].split(' ')[0] if "days" in str(time).split(',')[0] else str(0)
    time = str(time).split(':', 2)

    header = "<!DOCTYPE html> \
                   <html> \
                        <head> \
                            <style> \
                                * { \
                                    box-sizing: border-box; \
                                    margin: 0; \
                                    padding: 0; \
                                } \
                                body { \
                                } \
                                .container { \
                                    color: #333; \
                                    text-align: center; \
                                } \
                                h1 { \
                                    font-weight: normal; \
                                    color: #0 \
                                } \
                                li { \
                                    display: inline-block; \
                                    list-style-type: none; \
                                    padding: 1em; \
                                    color: #0 \
                                } \
                                li span { \
                                    display: block; \
                                } \
                                html, body { \
                                } \
                                body { \
                                    -webkit-box-align: center; \
                                    -ms-flex-align: center; \
                                    align-items: center; \
                                    display: -webkit-box; \
                                    display: -ms-flexbox; \
                                    display: flex; \
                                    font-family: -apple-system, \
                                    BlinkMacSystemFont, \
                                    'Segoe UI', \
                                    Roboto, \
                                    Oxygen-Sans, \
                                    Ubuntu, \
                                    Cantarell, \
                                    'Helvetica Neue', \
                                    sans-serif; \
                                } \
                                .container { \
                                    margin: 0 \
                                    auto; \
                                } \
                            </style> \
                        </head> "

    body = "<body> \
                            <div class='container'> \
                                <h1 id='head'> Runtime </h1> \
                                <ul> \
                                    <li><span id='days'> {} </span> Days </li> \
                                    <li><span id='hours'> {} </span> Hours </li> \
                                    <li><span id='minutes'> {} </span> Minutes </li> \
                                    <li><span id='seconds'> {} </span> Seconds </li> \
                                </ul> \
                            </div> \
                        </body> \
                </html>".format(days, time[0], time[1], time[2])

    return header + body
