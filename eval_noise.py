from typing import Tuple, List

import numpy as np
import torch
from samitorch.inputs.transformers import ToNifti1Image, NiftiToDisk, PadToShape, ToNumpyArray
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose

from deepNormalize.inputs.datasets import SingleNDArrayDataset
from deepNormalize.models.unet3d import Unet
from deepNormalize.utils.constants import EPSILON, ISEG_ID, MRBRAINS_ID, ABIDE_ID, GENERATOR
from deepNormalize.utils.slices import SliceBuilder

PATCH = 0
SLICE = 1
SEGMENTER = 1

DEVICE = "cuda:0"


def custom_collate(batch):
    patches = [item[0].unsqueeze(0) for item in batch]
    slices = [(item[1][0], item[1][1], item[1][2], item[1][3]) for item in batch]
    return [torch.cat(patches, dim=0), slices]


class ImageReconstructor(object):
    def __init__(self, images: List[str], patch_size: Tuple[int, int, int, int],
                 reconstructed_image_size: Tuple[int, int, int, int], step: Tuple[int, int, int, int],
                 batch_size: int = 5, models: List[torch.nn.Module] = None, normalize: bool = False,
                 is_ground_truth: bool = False, normalize_and_segment: bool = False, is_multimodal=False, alpha=0.0,
                 prob_bias=0.0, snr=0.0, prob_noise=0.0):
        self._patch_size = patch_size
        self._reconstructed_image_size = reconstructed_image_size
        self._step = step
        self._models = models
        self._do_normalize = normalize
        self._is_ground_truth = is_ground_truth
        self._do_normalize_and_segment = normalize_and_segment
        self._is_multimodal = is_multimodal
        self._batch_size = batch_size
        self._alpha = alpha
        self._snr = snr
        self._prob_bias = prob_bias
        self._prob_noise = prob_noise

        transformed_images = []
        for image in images:
            transform = Compose([ToNumpyArray(), PadToShape(target_shape=self._reconstructed_image_size)])
            transformed_images.append(transform(image))

        self._images = transformed_images

        self._overlap_maps = list(
            map(lambda image: SliceBuilder(image, self._patch_size, self._step).build_overlap_map(),
                self._images))

    @staticmethod
    def custom_collate(batch):
        patches = [item[0].unsqueeze(0) for item in batch]
        slices = [(item[1][0], item[1][1], item[1][2], item[1][3]) for item in batch]
        return [torch.cat(patches, dim=0), slices]

    @staticmethod
    def _normalize(img):
        return (img - np.min(img)) / (np.ptp(img) + EPSILON)

    def reconstruct_from_patches_3d(self):
        datasets = list(map(lambda image: SingleNDArrayDataset(image,
                                                               patch_size=self._patch_size,
                                                               step=self._step,
                                                               prob_bias=self._prob_bias,
                                                               prob_noise=self._prob_noise,
                                                               alpha=self._alpha,
                                                               snr=self._snr),
                            self._images))

        data_loaders = list(map(
            lambda dataset: DataLoader(dataset, batch_size=self._batch_size, num_workers=0, drop_last=False,
                                       shuffle=False,
                                       pin_memory=False, collate_fn=self.custom_collate), datasets))

        if len(datasets) == 2:
            reconstructed_image = [np.zeros(datasets[0].image_shape), np.zeros(datasets[1].image_shape)]

            for idx, (iseg_inputs, mrbrains_inputs) in enumerate(zip(data_loaders[ISEG_ID], data_loaders[MRBRAINS_ID])):
                inputs = torch.cat((iseg_inputs[PATCH], mrbrains_inputs[PATCH]))
                slices = [iseg_inputs[SLICE], mrbrains_inputs[SLICE]]

                if self._do_normalize:
                    patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs.to(DEVICE))))

                elif self._do_normalize_and_segment:
                    normalized_patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs.to(DEVICE))))
                    patches = torch.argmax(
                        torch.nn.functional.softmax(self._models[SEGMENTER](normalized_patches), dim=1), dim=1,
                        keepdim=True)
                else:
                    patches = inputs

                for pred_patch, slice in zip(patches[0:self._batch_size], slices[ISEG_ID]):
                    reconstructed_image[ISEG_ID][slice] = reconstructed_image[ISEG_ID][slice] + \
                                                          pred_patch.data.cpu().numpy()

                for pred_patch, slice in zip(patches[self._batch_size:self._batch_size * 2], slices[MRBRAINS_ID]):
                    reconstructed_image[MRBRAINS_ID][slice] = reconstructed_image[MRBRAINS_ID][slice] + \
                                                              pred_patch.data.cpu().numpy()

            if self._do_normalize_and_segment or self._is_ground_truth:
                reconstructed_image[ISEG_ID] = np.clip(
                    np.round(reconstructed_image[ISEG_ID] * self._overlap_maps[ISEG_ID]), a_min=0, a_max=3)
                reconstructed_image[MRBRAINS_ID] = np.clip(
                    np.round(reconstructed_image[MRBRAINS_ID] * self._overlap_maps[MRBRAINS_ID]), a_min=0, a_max=3)
            else:
                reconstructed_image[ISEG_ID] = reconstructed_image[ISEG_ID] * self._overlap_maps[ISEG_ID]
                reconstructed_image[MRBRAINS_ID] = reconstructed_image[MRBRAINS_ID] * self._overlap_maps[MRBRAINS_ID]

            transforms_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                "reconstructed_iseg_image_generated_noise_{}_alpha_{}.nii.gz".format(self._snr, self._alpha))])
            transforms_(reconstructed_image[ISEG_ID])
            transforms_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                "reconstructed_mrbrains_image_generated_noise_{}_alpha_{}.nii.gz".format(self._snr, self._alpha))])
            transforms_(reconstructed_image[MRBRAINS_ID])

        if len(datasets) == 3:
            reconstructed_image = [np.zeros(datasets[0].image_shape), np.zeros(datasets[1].image_shape),
                                   np.zeros(datasets[2].image_shape)]

            for idx, (iseg_inputs, mrbrains_inputs, abide_inputs) in enumerate(
                    zip(data_loaders[ISEG_ID], data_loaders[MRBRAINS_ID], data_loaders[ABIDE_ID])):
                inputs = torch.cat((iseg_inputs[PATCH], mrbrains_inputs[PATCH], abide_inputs[PATCH]))
                slices = [iseg_inputs[SLICE], mrbrains_inputs[SLICE], abide_inputs[SLICE]]

                if self._do_normalize:
                    patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs)))

                elif self._do_normalize_and_segment:
                    normalized_patches = torch.nn.functional.sigmoid((self._models[GENERATOR](inputs)))
                    patches = torch.argmax(
                        torch.nn.functional.softmax(self._models[SEGMENTER](normalized_patches), dim=1), dim=1,
                        keepdim=True)
                else:
                    patches = inputs

                for pred_patch, slice in zip(patches[0:self._batch_size], slices[ISEG_ID]):
                    reconstructed_image[ISEG_ID][slice] = reconstructed_image[ISEG_ID][slice] + \
                                                          pred_patch.data.cpu().numpy()

                for pred_patch, slice in zip(patches[self._batch_size:self._batch_size * 2], slices[MRBRAINS_ID]):
                    reconstructed_image[MRBRAINS_ID][slice] = reconstructed_image[MRBRAINS_ID][slice] + \
                                                              pred_patch.data.cpu().numpy()

                for pred_patch, slice in zip(patches[self._batch_size * 2:self._batch_size * 3], slices[ABIDE_ID]):
                    reconstructed_image[MRBRAINS_ID][slice] = reconstructed_image[ABIDE_ID][slice] + \
                                                              pred_patch.data.cpu().numpy()

            reconstructed_image[ISEG_ID] = reconstructed_image[ISEG_ID] * self._overlap_maps[ISEG_ID]
            reconstructed_image[MRBRAINS_ID] = reconstructed_image[MRBRAINS_ID] * self._overlap_maps[MRBRAINS_ID]
            reconstructed_image[ABIDE_ID] = reconstructed_image[ABIDE_ID] * self._overlap_maps[ABIDE_ID]

            if self._do_normalize_and_segment:
                reconstructed_image[ISEG_ID] = np.clip(np.round(reconstructed_image[ISEG_ID]), a_min=0, a_max=3)
                reconstructed_image[MRBRAINS_ID] = np.clip(np.round(reconstructed_image[MRBRAINS_ID]), a_min=0, a_max=3)
                reconstructed_image[ABIDE_ID] = np.clip(np.round(reconstructed_image[ABIDE_ID]), a_min=0, a_max=3)

            transforms_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                "reconstructed_iseg_image_generated_noise_{}_alpha_{}.nii.gz".format(self._snr, self._alpha))])
            transforms_(reconstructed_image[ISEG_ID])
            transforms_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                "reconstructed_mrbrains_image_generated_noise_{}_alpha_{}.nii.gz".format(self._snr, self._alpha))])
            transforms_(reconstructed_image[MRBRAINS_ID])
            transforms_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                "reconstructed_abide_image_generated_noise_{}_alpha_{}.nii.gz".format(self._snr, self._alpha))])
            transforms_(reconstructed_image[ABIDE_ID])


if __name__ == "__main__":
    checkpoint = torch.load("/mnt/md0/models/data_augmentation/Generator/Generator.tar",
                            map_location=torch.device(DEVICE))
    generator = Unet(1, 1, True, True)
    generator.load_state_dict(checkpoint["model_state_dict"])
    models = [generator.to(DEVICE).eval()]

    MRBrainS_path = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/2/T1/T1.nii.gz"
    iSEG_Path = "/mnt/md0/Data/iSEG_scaled/Training/9/T1/T1.nii.gz"

    snrs = [20.0, 40.0, 60.0, 80.0, 100.0]

    for snr in snrs:
        reconstructor = ImageReconstructor([iSEG_Path, MRBrainS_path],
                                           patch_size=(1, 32, 32, 32),
                                           reconstructed_image_size=(1, 256, 256, 192), step=(1, 16, 16, 16),
                                           batch_size=5, models=models, normalize=False,
                                           alpha=0.0, prob_bias=0.0, snr=0.0, prob_noise=0.0)
        reconstructor.reconstruct_from_patches_3d()
