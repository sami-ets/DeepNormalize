import numpy as np
from kerosene.utils.tensors import flatten
from scipy.spatial.distance import directed_hausdorff


def mean_hausdorff_distance(seg_pred, target):
    distances = np.zeros((4,))
    for channel in range(seg_pred.size(1)):
        distances[channel] = max(
            directed_hausdorff(
                flatten(seg_pred[:, channel, ...]).cpu().detach().numpy(),
                flatten(target[:, channel, ...]).cpu().detach().numpy())[0],
            directed_hausdorff(
                flatten(target[:, channel, ...]).cpu().detach().numpy(),
                flatten(seg_pred[:, channel, ...]).cpu().detach().numpy())[0])
    return distances
