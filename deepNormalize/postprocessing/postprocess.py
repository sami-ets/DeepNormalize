import torch
import pandas
import numpy as np
from samitorch.inputs.transformers import ToNumpyArray, PadToShape, ApplyMask
from kerosene.nn.functional import js_div
from torchvision.transforms import transforms
from deepNormalize.models.unet3d import Unet

if __name__ == '__main__':
    model = Unet(1, 1, True, True)

    iseg_csv = "/mnt/md0/Data/iSEG_scaled/Training/output_iseg_images.csv"
    mrbrains_csv = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/output_mrbrains_images.csv"

    iseg_csv = pandas.read_csv(iseg_csv)
    mrbrains_csv = pandas.read_csv(mrbrains_csv)

    transform = transforms.Compose([ToNumpyArray(), PadToShape((1, 256, 256, 192))])
    iseg_inputs = torch.tensor([transform(image) for image in iseg_csv["T1"]]).cuda()
    mrbrains_inputs = torch.tensor([transform(image) for image in mrbrains_csv["T1"]]).cuda()
    generated_iseg = transform("/mnt/md0/Research/Reconstructed_Normalized_iSEG_Image_80.nii.gz")
    generated_mrbrains = transform("/mnt/md0/Research/Reconstructed_Normalized_MRBrainS_Image_80.nii.gz")
    # generated_abide = transform("/mnt/md0/Research/Reconstructed_Normalized_ABIDE_Image_80.nii.gz")
    segmentation_iseg = transform("/mnt/md0/Research/Reconstructed_Segmented_iSEG_Image_80.nii.gz")
    segmentation_mrbrains = transform("/mnt/md0/Research/Reconstructed_Segmented_MRBrainS_Image_80.nii.gz")
    # segmentation_abide = transform("/mnt/md0/Research/Reconstructed_Segmented_ABIDE_Image_80.nii.gz")

    c, d, h, w = iseg_inputs.shape[1], iseg_inputs.shape[2], iseg_inputs.shape[3], iseg_inputs.shape[4]

    generated_iseg = torch.tensor(ApplyMask(segmentation_iseg)(generated_iseg)).cuda()
    generated_mrbrains = torch.tensor(ApplyMask(segmentation_mrbrains)(generated_mrbrains)).cuda()
    # generated_abide = torch.tensor(ApplyMask(segmentation_abide)(generated_abide)).cuda()

    hist_iseg_inputs = torch.cat(
        [torch.histc(iseg_inputs[i].view(1, c * d * h * w), bins=256, min=0, max=1).unsqueeze(0)
         for i in range(iseg_inputs.shape[0])])
    hist_mrbrains_inputs = torch.cat(
        [torch.histc(mrbrains_inputs[i].view(1, c * d * h * w), bins=256, min=0, max=1).unsqueeze(0)
         for i in range(mrbrains_inputs.shape[0])])

    hist_inputs = torch.cat([hist_iseg_inputs, hist_mrbrains_inputs], dim=0)

    hist_inputs = hist_inputs / hist_inputs.sum(dim=1).unsqueeze(1)
    hist_inputs = torch.nn.Softmax(dim=2)(hist_inputs.unsqueeze(0))

    hist_gen = torch.cat([
        torch.histc(generated_iseg.view(1, c * d * h * w), bins=256, min=0, max=1).unsqueeze(0),
        torch.histc(generated_mrbrains.view(1, c * d * h * w), bins=256, min=0, max=1).unsqueeze(0)])
    hist_gen = hist_gen / hist_gen.sum(dim=1).unsqueeze(1)
    hist_gen = torch.nn.Softmax(dim=2)(hist_gen.unsqueeze(0))

    print("JS Div Inputs : " + str((np.sqrt(js_div(hist_inputs).item()))))
    print("JS Div generated : " + str((np.sqrt(js_div(hist_gen).item()))))
