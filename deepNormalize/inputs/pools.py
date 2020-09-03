from typing import Tuple

import random
import torch


class ImagePool(object):
    def __init__(self, size: int = 360):
        self._size = size
        self._inputs = []
        self._targets = []

    @property
    def size(self):
        return self._size

    @property
    def nb_images(self):
        return len(self._inputs)

    @property
    def is_full(self):
        return self.nb_images >= self.size

    def query(self, images: Tuple[torch.Tensor, torch.Tensor]):
        xs, ys, dataset_ids = images[0], images[1][0], images[1][1]

        xs = xs.cpu().data
        ys = ys.cpu().data
        dataset_ids = dataset_ids.cpu().data

        return_images = []

        if self._size == 0:
            return xs, [ys, dataset_ids]

        for x, y, dataset_id in zip(xs, ys, dataset_ids):

            if not self.is_full:
                self._inputs.append(x)
                self._targets.append([y, dataset_id])
                return_images.append((x, [y, dataset_id]))

            else:
                if random.uniform(0, 1) > 0.5:
                    random_id = random.randint(0, self.size - 1)

                    x_tmp = self._inputs[random_id].clone()
                    y_tmp = self._targets[random_id][0].clone()
                    dataset_id_tmp = self._targets[random_id][1].clone()

                    self._inputs[random_id] = x
                    self._targets[random_id] = [y, dataset_id]

                    return_images.append((x_tmp, [y_tmp, dataset_id_tmp]))
                else:
                    return_images.append((x, [y, dataset_id]))

        x = torch.stack([return_image[0] for return_image in return_images], 0).to(images[0].device)
        y = torch.stack([return_image[1][0] for return_image in return_images], 0).to(images[0].device)
        dataset_id = torch.stack([return_image[1][1] for return_image in return_images], 0).to(images[0].device)

        return x, [y, dataset_id]
