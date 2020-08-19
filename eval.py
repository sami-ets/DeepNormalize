import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from kerosene.utils.tensors import to_onehot
from torch.utils.data import DataLoader

from deepNormalize.inputs.datasets import SingleImageDataset
from deepNormalize.models.dcgan3d import DCGAN
from deepNormalize.models.unet3d import Unet

PATCH = 0
SLICE = 1

PATCH_SIZE = (1, 32, 32, 32)
STEP = (1, 32, 32, 32)

BATCH_SIZE = 10

if __name__ == "__main__":
    DATA_SET_ROOT_PATH_MR_BRAINS = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/"
    DATA_SET_ROOT_PATH_ISEG = "/mnt/md0/Data/iSEG_scaled/Training/"

    MR_BRAIN_SUBJECT = str(2)
    ISEG_SUBJECT = str(9)
    # MR_BRAIN_SUBJECT = str(3)
    # ISEG_SUBJECT = str(6)

    mr_brains_dataset = SingleImageDataset(root_path=DATA_SET_ROOT_PATH_MR_BRAINS, subject=MR_BRAIN_SUBJECT,
                                           patch_size=PATCH_SIZE, step=STEP)
    iseg_dataset = SingleImageDataset(DATA_SET_ROOT_PATH_ISEG, subject=ISEG_SUBJECT, patch_size=PATCH_SIZE, step=STEP)

    mr_brain_data_loader = DataLoader(mr_brains_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False,
                                      shuffle=False,
                                      pin_memory=False)
    iseg_data_loader = DataLoader(iseg_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False, shuffle=False,
                                  pin_memory=False)

    # TODO restore your model from checkpoint
    generator = Unet(1, 1, True, True)
    discriminator = DCGAN(1, 3)

    checkpoint = torch.load("/mnt/md0/models/Generator/Generator.tar")
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.cuda()
    checkpoint = torch.load("/mnt/md0/models/Discriminator/Discriminator.tar")
    discriminator.load_state_dict(checkpoint["model_state_dict"])
    discriminator.cuda()

    iseg_pred = torch.zeros(3, ).cuda()
    iseg_pred_real = torch.zeros(3, ).cuda()
    mrbrains_pred = torch.zeros(3, ).cuda()
    mrbrains_pred_real = torch.zeros(3, ).cuda()

    iseg_count = 0
    mrbrain_count = 0

    for idx, inputs in enumerate(mr_brain_data_loader):
        normalised_patches = generator(inputs.cuda())

        disc_pred = discriminator.forward(normalised_patches)[0]

        mrbrains_pred = mrbrains_pred + to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                                  num_classes=3).sum(dim=0)

        disc_pred = discriminator.forward(inputs.cuda())[0]

        mrbrains_pred_real += to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                        num_classes=3).sum(dim=0)

        mrbrain_count += disc_pred.shape[0]

    for idx, inputs in enumerate(iseg_data_loader):
        normalised_patches = generator(inputs.cuda())

        disc_pred = discriminator.forward(normalised_patches)[0]

        iseg_pred = iseg_pred + to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                          num_classes=3).sum(dim=0)

        disc_pred = discriminator.forward(inputs.cuda())[0]

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
