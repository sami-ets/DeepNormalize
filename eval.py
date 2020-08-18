import torch
from kerosene.utils.tensors import to_onehot
from torch.utils.data import DataLoader

from deepNormalize.inputs.datasets import SingleImageDataset
from deepNormalize.models.dcgan3d import DCGAN
from deepNormalize.models.resnet3d import ResNet3D
from deepNormalize.models.unet3d import Unet

PATCH = 0
SLICE = 1

PATCH_SIZE = (1, 32, 32, 32)
STEP = (1, 8, 8, 8)

BATCH_SIZE = 10

if __name__ == "__main__":
    DATA_SET_ROOT_PATH_MR_BRAINS = "/mnt/md0/Data/MRBrainS_scaled/DataNii/TrainingData/"
    DATA_SET_ROOT_PATH_ISEG = "/mnt/md0/Data/iSEG_scaled/Training/"

    # MR_BRAIN_SUBJECT = str(2)
    # ISEG_SUBJECT = str(9)
    MR_BRAIN_SUBJECT = str(3)
    ISEG_SUBJECT = str(6)

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
    checkpoint = torch.load("/mnt/md0/models/Discriminator/Discriminator2.tar")
    discriminator.load_state_dict(checkpoint["model_state_dict"])
    discriminator.cuda()

    iseg_pred = torch.zeros(3, ).cuda()
    mr_brain_pred = torch.zeros(3, ).cuda()

    iseg_count = 0
    mrbrain_count = 0

    for idx, inputs in enumerate(mr_brain_data_loader):
        normalised_patches = generator(inputs.cuda())

        disc_pred = discriminator.forward(normalised_patches)[0]

        mr_brain_pred = mr_brain_pred + to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                                  num_classes=3).sum(dim=0)

        mrbrain_count += disc_pred.shape[0]

    for idx, inputs in enumerate(iseg_data_loader):
        normalised_patches = generator(inputs.cuda())

        disc_pred = discriminator.forward(normalised_patches)[0]

        iseg_pred = iseg_pred + to_onehot(torch.argmax(torch.softmax(disc_pred, dim=1), dim=1),
                                          num_classes=3).sum(dim=0)

        iseg_count += disc_pred.shape[0]

    print(str(iseg_pred.cpu().numpy() / iseg_count), str(mr_brain_pred.cpu().numpy() / mrbrain_count))
