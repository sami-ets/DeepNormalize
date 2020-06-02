import torch
from torch import nn


class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, gaussian_filter:bool = False):
        super(DCGAN, self).__init__()
        self._has_gaussian_filter = gaussian_filter

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)), nn.Conv3d(in_filters, out_filters, 3, 2),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout3d(0.25)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        gaussian_weights = torch.distributions.normal.Normal(1, 1).sample((1, 1, 5, 5, 5))
        gaussian_weights = gaussian_weights / torch.sum(gaussian_weights)

        if self._has_gaussian_filter:
            self._gaussian_filter = torch.nn.Conv3d(in_channels, in_channels, kernel_size=5, stride=1, padding=0,
                                                bias=False)
            self._gaussian_filter.weight.data = gaussian_weights

        self._layer_1 = nn.Sequential(*discriminator_block(in_channels, 16, bn=False))

        self._layer_2 = nn.Sequential(*discriminator_block(16, 32))

        self._layer_3 = nn.Sequential(*discriminator_block(32, 64))

        self._layer_4 = nn.Sequential(*discriminator_block(64, 128))

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, out_channels), nn.Sigmoid())

    def forward(self, img):

        if self._has_gaussian_filter:
            with torch.no_grad():
                img = self._gaussian_filter(img)

        out_1 = self._layer_1(img)
        out_2 = self._layer_2(out_1)
        out_3 = self._layer_3(out_2)
        out_4 = self._layer_4(out_3)
        out_5 = out_4.view(out_4.shape[0], -1)
        validity = self.adv_layer(out_5)

        return validity, out_1, out_2, out_3, out_4
