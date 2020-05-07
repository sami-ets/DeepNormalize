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
import uuid

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
from samitorch.inputs.transformers import ToNifti1Image, NiftiToDisk
from torchvision.transforms import Compose

from deepNormalize.utils.constants import DATASET_ID, ABIDE_ID, ISEG_ID, MRBRAINS_ID, IMAGE_TARGET


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
    arr_tuple_1 = tuple(([metric_values[0][i] for i in range(len(metric_values) - 1)]))
    interleaved = np.vstack(arr_tuple_1).reshape((-1,), order='F')
    arr_tuple_2 = tuple(([metric_values[1][i] for i in range(len(metric_values) - 1)]))
    interleaved_2 = np.vstack(arr_tuple_2).reshape((-1,), order='F')
    arr_tuple_3 = tuple(([metric_values[2][i] for i in range(len(metric_values) - 1)]))
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


def construct_triple_histrogram(gen_pred_iseg, input_iseg, gen_pred_mrbrains, input_mrbrains, gen_pred_abide,
                                input_abide):
    fig1, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2,
                                                              figsize=(12, 10))

    _, bins, _ = ax1.hist(gen_pred_iseg[gen_pred_iseg > 0].flatten(), bins=128,
                          density=False, label="iSEG")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Generated iSEG Histogram")
    ax1.legend()

    _ = ax2.hist(gen_pred_mrbrains[gen_pred_mrbrains > 0].flatten(), bins=bins,
                 density=False, label="MRBrainS")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Generated MRBrainS Histogram")
    ax2.legend()

    _ = ax3.hist(gen_pred_abide[gen_pred_abide > 0].flatten(), bins=bins,
                 density=False, label="ABIDE")
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Generated ABIDE Histogram")
    ax3.legend()

    _, bins, _ = ax4.hist(input_iseg[input_iseg > 0].flatten(), bins=128,
                          density=False, label="iSEG")
    ax4.set_xlabel("Intensity")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Input iSEG Histogram")
    ax4.legend()
    _ = ax5.hist(input_mrbrains[input_mrbrains > 0].flatten(), bins=bins,
                 density=False, label="MRBrainS")
    ax5.set_xlabel("Intensity")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Input MRBrainS Histogram")
    ax5.legend()

    _ = ax6.hist(input_abide[input_abide > 0].flatten(), bins=bins,
                 density=False, label="ABIDE")
    ax6.set_xlabel("Intensity")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Input ABIDE Histogram")
    ax6.legend()

    fig1.tight_layout()
    id = str(uuid.uuid4())
    fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

    fig1.clf()
    plt.close(fig1)

    return "/tmp/histograms-{}.png".format(str(id))


def construct_double_histrogram(gen_pred_iseg, input_iseg, gen_pred_mrbrains, input_mrbrains):
    fig1, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                  figsize=(12, 10))

    _, bins, _ = ax1.hist(input_iseg[input_iseg > 0].flatten(), bins=128,
                          density=False, label="iSEG")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Input iSEG Histogram")
    ax1.legend()

    _ = ax3.hist(gen_pred_iseg[gen_pred_iseg > 0].flatten(), bins=bins,
                 density=False, label="iSEG")
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Generated iSEG Histogram")
    ax3.legend()

    _, bins, _ = ax2.hist(input_mrbrains[input_mrbrains > 0].flatten(), bins=128,
                          density=False, label="MRBrainS")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Input MRBrainS Histogram")
    ax2.legend()

    _ = ax4.hist(gen_pred_mrbrains[gen_pred_mrbrains > 0].flatten(), bins=bins,
                 density=False, label="MRBrainS")
    ax4.set_xlabel("Intensity")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Generated MRBrainS Histogram")
    ax4.legend()

    fig1.tight_layout()
    id = str(uuid.uuid4())
    fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

    fig1.clf()
    plt.close(fig1)

    return "/tmp/histograms-{}.png".format(str(id))


def construct_single_histogram(gen_pred, input):
    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                    figsize=(12, 10))

    _, bins, _ = ax1.hist(input[input > 0].flatten(), bins=128,
                          density=False, label="Input")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Input Histogram")
    ax1.legend()

    _ = ax2.hist(gen_pred[gen_pred > 0].flatten(), bins=bins,
                 density=False, label="iSEG")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Generated Histogram")
    ax2.legend()

    fig1.tight_layout()
    id = str(uuid.uuid4())
    fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

    fig1.clf()
    plt.close(fig1)

    return "/tmp/histograms-{}.png".format(str(id))


def construct_class_histogram(inputs, target, gen_pred):
    iseg_inputs = inputs[torch.where(target[DATASET_ID] == ISEG_ID)]
    iseg_targets = target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ISEG_ID)]
    iseg_gen_pred = gen_pred[torch.where(target[DATASET_ID] == ISEG_ID)]
    mrbrains_inputs = inputs[torch.where(target[DATASET_ID] == MRBRAINS_ID)]
    mrbrains_targets = target[IMAGE_TARGET][torch.where(target[DATASET_ID] == MRBRAINS_ID)]
    mrbrains_gen_pred = gen_pred[torch.where(target[DATASET_ID] == MRBRAINS_ID)]
    abide_inputs = inputs[torch.where(target[DATASET_ID] == ABIDE_ID)]
    abide_targets = target[IMAGE_TARGET][torch.where(target[DATASET_ID] == ABIDE_ID)]
    abide_gen_pred = gen_pred[torch.where(target[DATASET_ID] == ABIDE_ID)]

    fig1, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(nrows=4, ncols=2,
                                                                          figsize=(12, 10))

    _, bins, _ = ax1.hist(iseg_gen_pred[torch.where(iseg_targets == 0)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax1.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 0)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax1.hist(abide_gen_pred[torch.where(abide_targets == 0)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Generated Background Histogram")
    ax1.legend()

    _, bins, _ = ax2.hist(iseg_gen_pred[torch.where(iseg_targets == 1)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax2.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 1)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax2.hist(abide_gen_pred[torch.where(abide_targets == 1)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Generated CSF Histogram")
    ax2.legend()

    _, bins, _ = ax3.hist(iseg_gen_pred[torch.where(iseg_targets == 2)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax3.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 2)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax3.hist(abide_gen_pred[torch.where(abide_targets == 2)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax3.set_xlabel("Intensity")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Generated Gray Matter Histogram")
    ax3.legend()

    _, bins, _ = ax4.hist(iseg_gen_pred[torch.where(iseg_targets == 3)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax4.hist(mrbrains_gen_pred[torch.where(mrbrains_targets == 3)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax4.hist(abide_gen_pred[torch.where(abide_targets == 3)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax4.set_xlabel("Intensity")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Generated White Matter Histogram")
    ax4.legend()

    _, bins, _ = ax5.hist(iseg_inputs[torch.where(iseg_targets == 0)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax5.hist(mrbrains_inputs[torch.where(mrbrains_targets == 0)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax5.hist(abide_inputs[torch.where(abide_targets == 0)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax5.set_xlabel("Intensity")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Input Background Histogram")
    ax5.legend()

    _, bins, _ = ax6.hist(iseg_inputs[torch.where(iseg_targets == 1)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax6.hist(mrbrains_inputs[torch.where(mrbrains_targets == 1)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax6.hist(abide_inputs[torch.where(abide_targets == 1)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax6.set_xlabel("Intensity")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Input CSF Histogram")
    ax6.legend()

    _, bins, _ = ax7.hist(iseg_inputs[torch.where(iseg_targets == 2)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax7.hist(mrbrains_inputs[torch.where(mrbrains_targets == 2)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax7.hist(abide_inputs[torch.where(abide_targets == 2)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax7.set_xlabel("Intensity")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Input Gray Matter Histogram")
    ax7.legend()

    _, bins, _ = ax8.hist(iseg_inputs[torch.where(iseg_targets == 3)].cpu().detach().numpy(), bins=128,
                          density=False, label="iSEG")
    _ = ax8.hist(mrbrains_inputs[torch.where(mrbrains_targets == 3)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="MRBrainS")
    _ = ax8.hist(abide_inputs[torch.where(abide_targets == 3)].cpu().detach().numpy(), bins=bins,
                 alpha=0.75, density=False, label="ABIDE")
    ax8.set_xlabel("Intensity")
    ax8.set_ylabel("Frequency")
    ax8.set_title("Input White Matter Histogram")
    ax8.legend()
    fig1.tight_layout()
    id = str(uuid.uuid4())
    fig1.savefig("/tmp/histograms-{}.png".format(str(id)))

    fig1.clf()
    plt.close(fig1)
    return "/tmp/histograms-{}.png".format(str(id))


def count(tensor, n_classes):
    count = torch.Tensor().new_zeros(size=(n_classes,), device="cpu")
    for i in range(n_classes):
        count[i] = torch.sum(tensor == i).int()
    return count


def get_all_patches(reconstruction_datasets, is_sliced=False):
    if not is_sliced:
        all_patches = list(
            map(lambda dataset: natural_sort([sample.x for sample in dataset._samples]), reconstruction_datasets))
        ground_truth_patches = list(
            map(lambda dataset: natural_sort([sample.y for sample in dataset._samples]), reconstruction_datasets))
    else:
        all_patches = list(
            map(lambda dataset: [patch.slice for patch in dataset._patches], reconstruction_datasets))
        ground_truth_patches = list(
            map(lambda dataset: [patch.slice for patch in dataset._patches], reconstruction_datasets))

    return all_patches, ground_truth_patches


def rebuild_images(datasets, all_patches, ground_truth_patches, input_reconstructors, gt_reconstructors,
                   normalize_reconstructors, segmentation_reconstructors):
    img_input = {k: v for (k, v) in zip(datasets, list(
        map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
            input_reconstructors)))}
    img_gt = {k: v for (k, v) in zip(datasets, list(
        map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), ground_truth_patches,
            gt_reconstructors)))}
    img_norm = {k: v for (k, v) in zip(datasets, list(
        map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
            normalize_reconstructors)))}
    img_seg = {k: v for (k, v) in zip(datasets, list(
        map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
            segmentation_reconstructors)))}

    return img_input, img_gt, img_norm, img_seg


def rebuild_image(datasets, all_patches, reconstructor):
    return {k: v for (k, v) in zip(datasets, list(
        map(lambda patches, reconstructor: reconstructor.reconstruct_from_patches_3d(patches), all_patches,
            reconstructor)))}


def rebuild_augmented_images(img_augmented, img_input, img_gt, img_norm, img_seg):
    mask_gt = {}
    mask_seg = {}

    for dataset in img_input.keys():
        mask_gt[dataset] = np.zeros(img_gt[dataset].shape)
        mask_seg[dataset] = np.zeros(img_seg[dataset].shape)

    for dataset in img_gt.keys():
        mask_gt[dataset][img_gt[dataset] >= 1] = 1

    for dataset in img_seg.keys():
        mask_seg[dataset][img_seg[dataset] >= 1] = 1

    img_augmented = {k: v for (k, v) in zip(img_augmented.keys(), list(
        map(lambda image, mask: image * mask, img_augmented.values(), mask_seg.values())))}

    img_input = {k: v for (k, v) in zip(img_input.keys(), list(
        map(lambda image, mask: image * mask, img_input.values(), mask_gt.values())))}

    augmented_minus_inputs = {k: v for (k, v) in zip(img_augmented.keys(), list(
        map(lambda augmented, input: augmented - input, img_augmented.values(), img_input.values())))}

    normalized_minus_inputs = {k: v for (k, v) in zip(img_augmented.keys(), list(
        map(lambda normalized, input: normalized - input, img_norm.values(), img_input.values())))}

    return augmented_minus_inputs, normalized_minus_inputs


def save_rebuilt_image(current_epoch, save_folder, datasets, image, image_type):
    if not os.path.exists(os.path.join(save_folder, "reconstructed_images")):
        os.makedirs(os.path.join(save_folder, "reconstructed_images"))

    for dataset in datasets:
        transform_img = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_{}_{}_Image_{}.nii.gz".format(image_type, dataset,
                                                                                                    str(
                                                                                                        current_epoch))))])
        transform_img(image[dataset])


def save_rebuilt_images(current_epoch, save_folder, datasets, img_input, img_norm, img_seg, img_gt):
    if not os.path.exists(os.path.join(save_folder, "reconstructed_images")):
        os.makedirs(os.path.join(save_folder, "reconstructed_images"))

    for dataset in datasets:
        transform_img_norm = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Normalized_{}_Image_{}.nii.gz".format(
                                                           dataset, str(current_epoch))))])
        transform_img_seg = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Segmented_{}_Image_{}.nii.gz".format(
                                                           dataset, str(current_epoch))))])
        transform_img_gt = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Ground_Truth_{}_Image_{}.nii.gz".format(
                                                           dataset, str(current_epoch))))])
        transform_img_input = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Input_{}_Image.nii.gz".format(
                                                           dataset, str(current_epoch))))])

        transform_img_norm(img_norm[dataset])
        transform_img_seg(img_seg[dataset])
        transform_img_gt(img_gt[dataset])
        transform_img_input(img_input[dataset])


def save_augmented_rebuilt_images(current_epoch, save_folder, datasets, img_augmented, augmented_minus_inputs,
                                  norm_minus_augmented):
    if not os.path.exists(os.path.join(save_folder, "reconstructed_images")):
        os.makedirs(os.path.join(save_folder, "reconstructed_images"))

    for dataset in datasets:
        transform_img_augmented = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Augmented_{}_Image_{}.nii.gz".format(
                                                           dataset, str(current_epoch))))])
        transform_img_augmented_minus_inputs = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Augmented_minus_Inputs_{}_Image_{}.nii.gz".format(
                                                           dataset, str(current_epoch))))])
        transform_img_normalized_minus_augmented = Compose(
            [ToNifti1Image(), NiftiToDisk(os.path.join(save_folder, "reconstructed_images",
                                                       "Reconstructed_Normalized_minus_Augmented_{}_Image_{}.nii.gz".format(
                                                           dataset, str(current_epoch))))])

        transform_img_augmented(img_augmented[dataset])
        transform_img_augmented_minus_inputs(augmented_minus_inputs[dataset])
        transform_img_normalized_minus_augmented(norm_minus_augmented[dataset])
