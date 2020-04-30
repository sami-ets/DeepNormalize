from torch import nn

class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)), nn.Conv3d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout3d(0.25)]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, out_channels), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity