import numpy as np
import pandas
import torch
from kerosene.nn.functional import js_div
from samitorch.inputs.transformers import ToNumpyArray, PadToShape, ApplyMask
from torchvision.transforms import transforms
import os
from deepNormalize.models.unet3d import Unet


def compute_probs(data, n=256):
    p = torch.Tensor().new_zeros(data.shape[0], n)

    for image in range(data.shape[0]):
        h = (torch.histc(data[image], n, min=0, max=1))
        p[image] = (h / (data.shape[1]))

    return p


def compute_js_divergence(samples):
    """
    Computes the JS Divergence using the support intersection between two different samples
    """
    return js_div(samples)


def compute_js_distance(js_div):
    return np.sqrt(js_div)


if __name__ == '__main__':

    iseg_csv = "/mnt/md0/Data/iSEG_scaled/Training/output_iseg_images.csv"
    mrbrains_csv = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/output_mrbrains_images.csv"
    abide_csv = "/mnt/md0/Data/ABIDE_scaled/output_abide_images.csv"
    iseg_csv = pandas.read_csv(iseg_csv)
    mrbrains_csv = pandas.read_csv(mrbrains_csv)
    abide_csv = pandas.read_csv(abide_csv).sample(75)

    c, d, h, w = 1, 256, 256, 192

    transform = transforms.Compose([ToNumpyArray(), PadToShape((c, d, h, w))])
    iseg_inputs = torch.tensor([transform(image) for image in iseg_csv["T1"]])
    mrbrains_inputs = torch.tensor([transform(image) for image in mrbrains_csv["T1"]])
    # abide_inputs = torch.tensor([transform(image) for image in abide_csv["T1"]])


    generated_iseg = transform("/mnt/md0/Research/DualUNet/Reconstructed_Normalized_iSEG_Image_80.nii.gz")
    generated_mrbrains = transform("/mnt/md0/Research/DualUNet/Reconstructed_Normalized_MRBrainS_Image_80.nii.gz")
    generated_abide = transform("/mnt/md0/Research/DualUNet/Reconstructed_Normalized_ABIDE_Image_80.nii.gz")
    segmentation_iseg = transform("/mnt/md0/Research/DualUNet/Reconstructed_Segmented_iSEG_Image_80.nii.gz")
    segmentation_mrbrains = transform("/mnt/md0/Research/DualUNet/Reconstructed_Segmented_MRBrainS_Image_80.nii.gz")
    segmentation_abide = transform("/mnt/md0/Research/DualUNet/Reconstructed_Segmented_ABIDE_Image_80.nii.gz")
    generated_iseg = torch.tensor(ApplyMask(segmentation_iseg)(generated_iseg))
    generated_mrbrains = torch.tensor(ApplyMask(segmentation_mrbrains)(generated_mrbrains))
    generated_abide = torch.tensor(ApplyMask(segmentation_abide)(generated_abide))

    train_samples = torch.cat([iseg_inputs, mrbrains_inputs, abide_inputs], dim=0).view(90, c * d * h * w)
    generated_samples = torch.cat([generated_iseg, generated_mrbrains, generated_abide], dim=0).view(3, c * d * h * w)

    p_inputs = compute_probs(train_samples)
    p_generated = compute_probs(generated_samples)

    js_divergence_inputs = compute_js_divergence(p_inputs).item()
    js_divergence_generated = compute_js_divergence(p_generated).item()

    print("JS Distance Inputs : " + str(compute_js_distance(js_divergence_inputs)))
    print("JS Distance Generated : " + str(compute_js_distance(js_divergence_generated)))

    # hist_iseg_inputs = torch.cat(
    #     [torch.histc(iseg_inputs[i].view(1, c * d * h * w), bins=10, min=0, max=1).unsqueeze(0)
    #      for i in range(iseg_inputs.shape[0])])
    # hist_mrbrains_inputs = torch.cat(
    #     [torch.histc(mrbrains_inputs[i].view(1, c * d * h * w), bins=10, min=0, max=1).unsqueeze(0)
    #      for i in range(mrbrains_inputs.shape[0])])
    #
    # hist_inputs = torch.cat([hist_iseg_inputs, hist_mrbrains_inputs], dim=0)
    #
    # hist_inputs = hist_inputs / hist_inputs.sum(dim=1).unsqueeze(1)
    # # hist_inputs = torch.nn.Softmax(dim=2)(hist_inputs.unsqueeze(0))
    #
    # hist_gen = torch.cat([
    #     torch.histc(generated_iseg.view(1, c * d * h * w), bins=10, min=0, max=1).unsqueeze(0),
    #     torch.histc(generated_mrbrains.view(1, c * d * h * w), bins=10, min=0, max=1).unsqueeze(0)])
    # hist_gen = hist_gen / hist_gen.sum(dim=1).unsqueeze(1)
    # # hist_gen = torch.nn.Softmax(dim=2)(hist_gen.unsqueeze(0))
    #
    # print("JS Div Inputs : " + str((np.sqrt(js_div(hist_inputs.unsqueeze(0)).item()))))
    # print("JS Div generated : " + str((np.sqrt(js_div(hist_gen.unsqueeze(0)).item()))))
