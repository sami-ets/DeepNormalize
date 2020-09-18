import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from samitorch.inputs.transformers import ToNumpyArray
from torchvision.transforms import Compose
import os
import re
import seaborn as sns
plt.rcParams.update({'font.size': 22})

import matplotlib.ticker as ticker


class MRBrainSImageExtraction(object):

    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        images_np = list()
        seg_map_np = list()

        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            images = list(filter(re.compile(r"^T1\.nii").search, files))

            for file in images:
                images_np.append(self._transforms(os.path.join(root, file)))

        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            labels = list(filter(re.compile(r"^LabelsForTesting.*\.nii").search, files))

            for file in labels:
                seg_map_np.append(self._transforms(os.path.join(root, file)))

        images = np.array(images_np).astype(np.float32)
        segmentation = np.array(seg_map_np).astype(np.float32)

        return images, segmentation


class ISEGImageExtraction(object):

    def __init__(self, root_dir: str):
        self._root_dir = root_dir
        self._transforms = Compose([ToNumpyArray()])

    def run(self):
        images_np = list()
        seg_np = list()

        for root, dirs, files in os.walk(os.path.join(self._root_dir)):
            images = list(filter(re.compile(r".*T1\.nii").search, files))
            for file in images:
                images_np.append(self._transforms(os.path.join(root, file)))

        for root, dirs, files in os.walk(os.path.join(self._root_dir, "label")):
            for file in files:
                seg_np.append(self._transforms(os.path.join(root, file)))

        images = np.array(images_np).astype(np.float32)
        seg_np = np.array(seg_np).astype(np.float32)

        return images, seg_np


if __name__ == '__main__':
    mrbrains_path = "/mnt/md0/Data/Direct/MRBrainS/DataNii/TrainingData/"
    iseg_path = "/mnt/md0/Data/Direct/iSEG/Training/"
    mrbrains_images, mrbrains_seg = MRBrainSImageExtraction(mrbrains_path).run()
    iseg_images, iseg_seg = ISEGImageExtraction(iseg_path).run()
    scale_y = 1e3
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(19,5))
    ax0, ax1 = axes.flatten()

    ax0.hist(mrbrains_images[np.where(mrbrains_seg == 2)], bins=64, range=[0, 800], alpha=0.5, color="green",
             label="Gray Matter")
    ax0.hist(mrbrains_images[np.where(mrbrains_seg == 3)], bins=64, range=[0, 800], alpha=0.5, color="red",
             label="White Matter")
    ax0.hist(mrbrains_images[np.where(mrbrains_seg == 1)], bins=64, range=[0, 800], alpha=0.5, color="blue",
             label="CSF")
    ax0.legend(prop={'size': 22})
    ticks = ax0.get_yticks().tolist()
    ticks = [int(i/1000) for i in ticks]
    ticks = [str(ticks[0])] + [str(tick) + "K" for tick in ticks[1:]]
    ax0.set_yticklabels(ticks)
    ax0.set_ylabel('Voxel count')
    ax0.set_xlabel('Intensities')
    ax0.set_title("MRBrainS Intensity Distribution per Class")

    ax1.hist(iseg_images[np.where(iseg_seg == 2)], bins=64, range=[0, 800], alpha=0.5, color="green",
             label="Gray Matter")
    ax1.hist(iseg_images[np.where(iseg_seg == 3)], bins=64, range=[0, 800], alpha=0.5, color="red",
             label="White Matter")
    ax1.hist(iseg_images[np.where(iseg_seg == 1)], bins=64, range=[0, 800], alpha=0.5, color="blue", label="CSF")
    ax1.set_title("iSEG Intensity Distribution per Class")
    ticks = ax1.get_yticks().tolist()
    ticks = [int(i / 1000) for i in ticks]
    ticks = [str(ticks[0])] + [str(tick) + "K" for tick in ticks[1:]]
    ax1.set_yticklabels(ticks)
    ax1.set_ylabel('Voxel count')
    ax1.set_xlabel('Intensities')
    ax1.legend(prop={'size': 22})

    fig.tight_layout()

    plt.savefig("journal_intensity_distributions.png", dpi=450)
    plt.show()