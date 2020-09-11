import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn
import torch
from kerosene.utils.tensors import to_onehot
from samitorch.inputs.transformers import ToNumpyArray, PadToShape
from torch.utils.data import DataLoader, Dataset
import visdom

from deepNormalize.models.dcgan3d import DCGAN
from deepNormalize.models.resnet3d import ResNet18
from deepNormalize.models.unet3d import Unet
from deepNormalize.utils.slices import SliceBuilder

PATCH = 0
SLICE = 1

PATCH_SIZE = (1, 32, 32, 32)
STEP = (1, 8, 8, 8)

BATCH_SIZE = 5

DEVICE = "cuda:1"


class SingleImageDataset(Dataset):
    def __init__(self, root_path: str, subject: str, patch_size, step):
        self._image_path = os.path.join(root_path, subject, "T1", "T1.nii.gz")
        self._image = PadToShape(target_shape=[1, 256, 256, 192])(ToNumpyArray()(self._image_path))
        self._slices = SliceBuilder(self._image, patch_size=patch_size, step=step).build_slices(
            keep_centered_on_foreground=True)
        self._image_max = self._image.max()
        self._patch_size = patch_size
        self._step = step

    def __getitem__(self, index):
        try:
            image = torch.tensor([(self._image[self._slices[index]])], dtype=torch.float32,
                                 requires_grad=False).squeeze(0)
            return image
        except Exception as e:
            pass

    def __len__(self):
        return len(self._slices)

    @property
    def image_shape(self):
        return self._image.shape


if __name__ == "__main__":

    vis = visdom.Visdom(env="confusion_matrix")

    DATA_SET_ROOT_PATH_MR_BRAINS = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/"
    DATA_SET_ROOT_PATH_ISEG = "/mnt/md0/Data/iSEG_scaled/Training/"

    # MR_BRAIN_SUBJECT = str(2)
    # ISEG_SUBJECT = str(9)
    MR_BRAIN_SUBJECT = str(3)
    ISEG_SUBJECT = str(6)

    mrbrains_dataset = SingleImageDataset(root_path=DATA_SET_ROOT_PATH_MR_BRAINS, subject=MR_BRAIN_SUBJECT,
                                          patch_size=PATCH_SIZE, step=STEP)
    iseg_dataset = SingleImageDataset(DATA_SET_ROOT_PATH_ISEG, subject=ISEG_SUBJECT, patch_size=PATCH_SIZE, step=STEP)

    mrbrains_dataloader = DataLoader(mrbrains_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False,
                                     shuffle=False,
                                     pin_memory=False)
    iseg_data_loader = DataLoader(iseg_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False, shuffle=False,
                                  pin_memory=False)

    # TODO restore your model from checkpoint
    generator = Unet(1, 1, True, True)
    discriminator = ResNet18({"in_channels": 1,
                              "out_channels": 3,
                              "num_groups": None,
                              "conv_groups": 1,
                              "width_per_group": 64,
                              "padding": [1, 1, 1, 1, 1, 1],
                              "activation": "ReLU",
                              "zero_init_residual": False,
                              "replace_stride_with_dilation": None,
                              "gaussian_filter": False})

    checkpoint = torch.load("/mnt/md0/models/Generator/Generator_147.tar", map_location="cpu")
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.to(DEVICE)
    checkpoint = torch.load("/mnt/md0/models/Discriminator/Discriminator_147.tar", map_location="cpu")
    discriminator.load_state_dict(checkpoint["model_state_dict"])
    discriminator.to(DEVICE)

    iseg_pred = torch.zeros(3, ).to(DEVICE)
    iseg_pred_real = torch.zeros(3, ).to(DEVICE)
    mrbrains_pred = torch.zeros(3, ).to(DEVICE)
    mrbrains_pred_real = torch.zeros(3, ).to(DEVICE)

    iseg_count = 0
    mrbrain_count = 0

    for idx, (iseg_inputs, mrbrains_inputs) in enumerate(zip(iseg_data_loader, mrbrains_dataloader)):
        inputs = torch.cat((iseg_inputs, mrbrains_inputs))

        disc_pred_on_real = discriminator.forward(inputs.to(DEVICE))[0]
        normalised_patches = torch.nn.functional.sigmoid(generator(inputs.to(DEVICE)))
        disc_pred_on_gen = discriminator.forward(normalised_patches)[0]

        mrbrains_pred = mrbrains_pred + to_onehot(
            torch.argmax(torch.softmax(disc_pred_on_gen[BATCH_SIZE:2 * BATCH_SIZE], dim=1), dim=1),
            num_classes=3).sum(dim=0)

        mrbrain_count += BATCH_SIZE

        mrbrains_pred_real += to_onehot(
            torch.argmax(torch.softmax(disc_pred_on_real[BATCH_SIZE:2 * BATCH_SIZE], dim=1), dim=1),
            num_classes=3).sum(dim=0)

        iseg_pred = iseg_pred + to_onehot(torch.argmax(torch.softmax(disc_pred_on_gen[0:BATCH_SIZE], dim=1), dim=1),
                                          num_classes=3).sum(dim=0)

        iseg_pred_real += to_onehot(torch.argmax(torch.softmax(disc_pred_on_real[0:BATCH_SIZE], dim=1), dim=1),
                                    num_classes=3).sum(dim=0)

        iseg_count += BATCH_SIZE

    total = iseg_count + mrbrain_count

    # data = np.vstack((iseg_pred_real.cpu().numpy(),
    #                   mrbrains_pred_real.cpu().numpy(),
    #                   iseg_pred.cpu().numpy(),
    #                   mrbrains_pred.cpu().numpy()))
    #
    # sns = seaborn.heatmap(data, annot=True, yticklabels=["iSEG", "MRBrainS", "Gen iSEG", "Gen MRBrainS"],
    #                       xticklabels=["iSEG", "MRBrainS", "Generated"], cmap="viridis")
    #
    # plt.show()
    #
    data_lr = np.fliplr(np.vstack((iseg_pred_real.cpu().numpy(),
                                   mrbrains_pred_real.cpu().numpy(),
                                   iseg_pred.cpu().numpy(),
                                   mrbrains_pred.cpu().numpy())))
    #
    # sns = seaborn.heatmap(data_lr, annot=True, yticklabels=["iSEG", "MRBrainS", "Gen iSEG", "Gen MRBrainS"],
    #                       xticklabels=["Generated", "MRBRainS", "iSEG"], cmap="viridis")
    #
    # plt.show()

    data_ud = np.flipud(np.fliplr(np.vstack((iseg_pred_real.cpu().numpy(),
                                             mrbrains_pred_real.cpu().numpy(),
                                             iseg_pred.cpu().numpy(),
                                             mrbrains_pred.cpu().numpy()))))

    sns = seaborn.heatmap(data_ud, annot=True, yticklabels=["Gen MRBrainS", "Gen iSEG", "MRBrainS", "iSEG"],
                          xticklabels=["Generated", "MRBRainS", "iSEG"], cmap="viridis")

    plt.show()

    data_ud_norm = np.flipud(np.fliplr(np.vstack((iseg_pred_real.cpu().numpy() / total,
                                                  mrbrains_pred_real.cpu().numpy() / total,
                                                  iseg_pred.cpu().numpy() / total,
                                                  mrbrains_pred.cpu().numpy() / total))))

    sns = seaborn.heatmap(data_ud_norm, annot=True, yticklabels=["Gen MRBrainS", "Gen iSEG", "MRBrainS", "iSEG"],
                          xticklabels=["Generated", "MRBRainS", "iSEG"], cmap="RdBu_r")

    plt.show()

    vis.heatmap(data_lr, opts={
        "columnnames": ["Generated", "MRBrainS", "ISEG"],
        "rownames": ["iSEG", "MRBrainS", "Gen iSEG", "Gen MRBrainS"],
        "title": "Actual Confusion Matrix"})
