import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats
import numpy as np

from samitorch.inputs.transformers import *


def to_graph_data(path: str, original: str, title: str, style: str, dataset: str):
    image = ToNDTensor()(ToNumpyArray()(path)).squeeze(0)
    original = ToNDTensor()(ToNumpyArray()(original)).squeeze(0)
    image = (image - image.mean()) / image.std()
    image[torch.where(original == 0)] = 0

    if dataset == "MrBrains":
        image_flatten = image[:, :, 160].transpose(0, 1)
    else:
        image_flatten = image[:, :, 150].transpose(0, 1)

    xs = []
    ys = []

    for x in range(0, image_flatten.shape[0]):
        values = image_flatten[x][torch.where(image_flatten[x] != 0)].numpy()
        ys.extend(values)
        xs.extend([x] * len(values))

    return np.array(xs) - np.array(xs).min(), ys, [title] * len(ys), [style] * len(ys)


if __name__ == "__main__":
    ROOT_PATH = "C:\\Users\\Benoit\\Desktop\\ETS\\Maitrise\\dev\\normalized\\bias field\\{}\\"
    ORIGINAL_IMAGE = "original.nii.gz"
    NOISY = "noisy_{}.nii.gz"
    DENOISE = "denoise_{}.nii.gz"

    ALPHA = [0.5, 0.9]
    DATASET = ["MrBrains", "ISeg"]

    for dataset in DATASET:
        xs = []
        ys = []
        zs = []
        styles = []
        legend = ["Original"]

        x, y, z, style = to_graph_data(os.path.join(ROOT_PATH.format(dataset), ORIGINAL_IMAGE),
                                       os.path.join(ROOT_PATH.format(dataset), ORIGINAL_IMAGE), "Original", "Real", dataset)
        r, _ = stats.pearsonr(x, y)
        print("Original {}".format(r))
        xs.extend(x)
        ys.extend(y)
        zs.extend(z)
        styles.extend(style)

        for alpha in ALPHA:
            x, y, z, style = to_graph_data(os.path.join(ROOT_PATH.format(dataset), NOISY.format(int(alpha * 10))),
                                           os.path.join(ROOT_PATH.format(dataset), ORIGINAL_IMAGE),
                                       "Original + Bias Field (\u03B1={})".format(alpha), "Real", dataset)
            r, _ = stats.pearsonr(x, y)
            print("Original + Bias Field (\u03B1={}) {}".format(alpha, r))
            xs.extend(x)
            ys.extend(y)
            zs.extend(z)
            styles.extend(style)
            legend.append("Original + Bias Field (\u03B1={})".format(alpha))

            x, y, z, style = to_graph_data(os.path.join(ROOT_PATH.format(dataset), DENOISE.format(int(alpha * 10))),
                                           os.path.join(ROOT_PATH.format(dataset), ORIGINAL_IMAGE),
                                       "Original + Bias Field (\u03B1={})".format(alpha), "Generated", dataset)
            r, _ = stats.pearsonr(x, y)
            print("Denoised (\u03B1={}) {}".format(alpha, r))
            xs.extend(x)
            ys.extend(y)
            zs.extend(z)
            styles.extend(style)
            legend.append("Normalized (\u03B1={})".format(alpha))

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        styles = np.array(styles)

        data = {"Voxel Location": xs, "Voxel Intensity": ys, "Experiments": zs, "Style": styles}
        ax = sns.lineplot(x="Voxel Location", y="Voxel Intensity", data=pd.DataFrame.from_dict(data),
                          hue="Experiments", legend='brief', ci=None, style="Style")
        # plt.ylim(0, 1.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(legend, title=None)
        plt.tight_layout()
        plt.show()
