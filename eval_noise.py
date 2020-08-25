import numpy as np
import torch
from torch.utils.data import DataLoader

from deepNormalize.inputs.datasets import SingleImageDataset
from deepNormalize.models.dcgan3d import DCGAN
from deepNormalize.models.unet3d import Unet
from deepNormalize.utils.slices import SliceBuilder
from samitorch.inputs.transformers import ToNifti1Image, NiftiToDisk

from torchvision.transforms import transforms

PATCH = 0
SLICE = 1

PATCH_SIZE = (1, 32, 32, 32)
STEP = (1, 8, 8, 8)
BATCH_SIZE = 15
ALPHA = 0.5
PROB_BIAS = 1.0
PROB_NOISE = 0.0

NOISE_LEVELS = [0]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def custom_collate(batch):
    patches = [item[0].unsqueeze(0) for item in batch]
    slices = [(item[1][0], item[1][1], item[1][2], item[1][3]) for item in batch]
    return [torch.cat(patches, dim=0), slices]


if __name__ == "__main__":

    for snr in NOISE_LEVELS:
        for alpha in ALPHAS:
            dataset = SingleImageDataset("/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/", subject="2",
                                         patch_size=PATCH_SIZE,
                                         step=STEP,
                                         prob_bias=PROB_BIAS,
                                         prob_noise=PROB_NOISE,
                                         alpha=alpha,
                                         snr=snr)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False, shuffle=False,
                                     pin_memory=False,
                                     collate_fn=custom_collate)

            c, d, h, w = dataset.image_shape
            reconstructed_image = np.zeros((1, d, h, w))

            overlap_map = SliceBuilder(dataset._image, patch_size=PATCH_SIZE, step=STEP).build_overlap_map()

            generator = Unet(1, 1, True, True)
            discriminator = DCGAN(1, 3)

            checkpoint = torch.load("/mnt/md0/models/data_augmentation/Generator/Generator.tar")
            generator.load_state_dict(checkpoint["model_state_dict"])
            generator.cuda().eval()
            # checkpoint = torch.load("/mnt/md0/models/data_augmentation/Discriminator/Discriminator.tar")
            # discriminator.load_state_dict(checkpoint["model_state_dict"])
            # discriminator.cuda().eval()

            for idx, patch_slice in enumerate(data_loader):
                patches, slices = patch_slice[PATCH].cuda(), patch_slice[SLICE]
                pred_patches = torch.nn.functional.sigmoid(generator(patches))

                for pred_patch, slice in zip(pred_patches, slices):
                    reconstructed_image[slice] = reconstructed_image[slice] + pred_patch.data.cpu().numpy()

                print("{}/{} patches processed".format(idx * BATCH_SIZE, len(dataset)))

            reconstructed_image = reconstructed_image * overlap_map

            transforms_ = transforms.Compose([ToNifti1Image(), NiftiToDisk(
                "reconstructed_mrbrains_image_generated_noise_{}_alpha_{}.nii.gz".format(snr, alpha))])
            transforms_(reconstructed_image)
