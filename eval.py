import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn
import torch
from kerosene.utils.tensors import to_onehot
from samitorch.inputs.transformers import PadToPatchShape, ToNumpyArray
from torch.utils.data import DataLoader, Dataset

from deepNormalize.models.dcgan3d import DCGAN
from deepNormalize.models.unet3d import Unet
from slices import SliceBuilder

PATCH = 0
SLICE = 1

PATCH_SIZE = (1, 32, 32, 32)
STEP = (1, 32, 32, 32)

BATCH_SIZE = 10
class SingleImageDataset(Dataset):
    def __init__(self, root_path: str, subject: str, patch_size, step):
        self._image_path = os.path.join(root_path, subject, "T1", "T1.nii.gz")
        self._image = PadToPatchShape(patch_size=patch_size, step=step)(ToNumpyArray()(self._image_path))
        self._slices = SliceBuilder(self._image.shape, patch_size=patch_size, step=step).build_slices()
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
    DATA_SET_ROOT_PATH_MR_BRAINS = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/"
    DATA_SET_ROOT_PATH_ISEG = "/mnt/md0/Data/iSEG_scaled/Training/"

    MR_BRAIN_SUBJECT = str(2)
    ISEG_SUBJECT = str(9)
    # MR_BRAIN_SUBJECT = str(3)
    # ISEG_SUBJECT = str(6)

    mrbrains_dataset = SingleImageDataset(root_path=DATA_SET_ROOT_PATH_MR_BRAINS, subject=MR_BRAIN_SUBJECT,
                                          patch_size=PATCH_SIZE, step=STEP)
    iseg_dataset = SingleImageDataset(DATA_SET_ROOT_PATH_ISEG, subject=ISEG_SUBJECT, patch_size=PATCH_SIZE, step=STEP)

    mrbrain_data_loader = DataLoader(mrbrains_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False,
                                     shuffle=False,
                                     pin_memory=False)
    iseg_data_loader = DataLoader(iseg_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False, shuffle=False,
                                  pin_memory=False)

    # TODO restore your model from checkpoint
    generator = Unet(1, 1, True, True)
    discriminator = DCGAN(1, 3)

    checkpoint = torch.load("/mnt/md0/models/Generator/Generator.tar")
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.to("cuda:1").eval()
    checkpoint = torch.load("/mnt/md0/models/Discriminator/Discriminator.tar")
    discriminator.load_state_dict(checkpoint["model_state_dict"])
    discriminator.to("cuda:1").eval()

    iseg_pred = torch.zeros(3, ).to("cuda:1")
    iseg_pred_real = torch.zeros(3, ).to("cuda:1")
    mrbrains_pred = torch.zeros(3, ).to("cuda:1")
    mrbrains_pred_real = torch.zeros(3, ).to("cuda:1")

    iseg_count = 0
    mrbrain_count = 0

    for idx, inputs in enumerate(mrbrain_data_loader):
        normalised_patches = torch.nn.functional.sigmoid(generator(inputs.to("cuda:1")))

        disc_pred = discriminator.forward(normalised_patches)[0]

        mrbrains_pred = mrbrains_pred + to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                                  num_classes=3).sum(dim=0)

        disc_pred = discriminator.forward(inputs.to("cuda:1"))[0]

        mrbrains_pred_real += to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                        num_classes=3).sum(dim=0)

        mrbrain_count += disc_pred.shape[0]

    for idx, inputs in enumerate(iseg_data_loader):
        normalised_patches = torch.nn.functional.sigmoid(generator(inputs.to("cuda:1")))

        disc_pred = discriminator.forward(normalised_patches)[0]

        iseg_pred = iseg_pred + to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                          num_classes=3).sum(dim=0)

        disc_pred = discriminator.forward(inputs.to("cuda:1"))[0]

        iseg_pred_real += to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                    num_classes=3).sum(dim=0)

        iseg_count += disc_pred.shape[0]

    print(str(iseg_pred.cpu().numpy() / iseg_count), str(mrbrains_pred.cpu().numpy() / mrbrain_count))
    print(str(iseg_pred_real.cpu().numpy() / iseg_count), str(mrbrains_pred_real.cpu().numpy() / mrbrain_count))

    data = (np.vstack((mrbrains_pred.cpu().numpy()/mrbrains_pred.sum().cpu().numpy(), iseg_pred.cpu().numpy()/iseg_pred.sum().cpu().numpy(), mrbrains_pred_real.cpu().numpy()/mrbrains_pred_real.sum().cpu().numpy(),
                      iseg_pred_real.cpu().numpy()/iseg_pred_real.sum().cpu().numpy())))
           # / (mrbrains_pred+iseg_pred+mrbrains_pred_real+iseg_pred_real).sum().cpu().numpy()
    sns = seaborn.heatmap(data, annot=True, yticklabels=["Gen MRBrainS", "Gen iSEG", "MRBrainS", "iSEG"], xticklabels=["Generated", "MRBRainS", "iSEG"], cmap="RdBu_r")
    plt.show()
