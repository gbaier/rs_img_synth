import torch
import torch.nn as nn
import torch.nn.functional as TF

from . import common


class UNet(nn.Module):
    """ U-Net implemenation

    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015) https://arxiv.org/abs/1505.04597

    Parameters
    ----------
    in_channels : int
        number of input channels
    n_classes : int
        number of output channels
    n_downsample : int
        depth of the network
    """

    def __init__(self, in_channels, n_classes, n_downsample=4):
        super().__init__()

        # input and output channels for the downsampling path
        out_channels = [64 * (2 ** i) for i in range(n_downsample)]
        in_channels = [in_channels] + out_channels[:-1]

        self.down_path = nn.ModuleList(
            [
                self._make_unet_conv_block(ich, och)
                for ich, och in zip(in_channels, out_channels)
            ]
        )

        # input channels of the upsampling path
        in_channels = [64 * (2 ** i) for i in range(n_downsample, 0, -1)]
        self.body = self._make_unet_conv_block(out_channels[-1], in_channels[0])

        self.upsamplers = nn.ModuleList(
            [self._make_upsampler(nch, nch // 2) for nch in in_channels]
        )

        self.up_path = nn.ModuleList(
            [self._make_unet_conv_block(nch, nch // 2) for nch in in_channels]
        )

        self.last = nn.Conv2d(64, n_classes, kernel_size=1)

        self._return_intermed = False

    @staticmethod
    def _make_unet_conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    @staticmethod
    def _make_upsampler(in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    @property
    def return_intermed(self):
        return self._return_intermed

    @return_intermed.setter
    def return_intermed(self, value):
        self._return_intermed = value

    def forward(self, x):
        blocks = []
        for down in self.down_path:
            # UNet conv block increases the number of channels
            x = down(x)
            blocks.append(x)
            # Downsampling, by mass pooling
            x = TF.max_pool2d(x, 2)

        x = self.body(x)

        for upsampler, up, block in zip(
            self.upsamplers, self.up_path, reversed(blocks)
        ):
            # upsample and reduce number of channels
            x = upsampler(x)
            x = torch.cat([x, block], dim=1)
            # UNet conv block reduces the number of channels again
            x = up(x)

        x = self.last(x)

        if self.return_intermed:
            return common.NNOutput(x, blocks)

        return x
