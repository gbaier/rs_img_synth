import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

from . import common


class PatchGAN(nn.Module):
    """ PatchGAN discriminator used by Pix2Pix HD and SPADE

    Original code taken from [1] but with lots of modifications.

    [1] https://github.com/NVlabs/SPADE/blob/master/models/networks/discriminator.py.

    """

    def __init__(self, input_nc, n_layers=4, init_nc=64):
        super().__init__()

        self.n_layers = n_layers

        self.kernel_size = 4
        self.init_nc = init_nc
        self._return_intermed = False

        self.init_conv = nn.Sequential(
            nn.Conv2d(
                input_nc, self.init_nc, self.kernel_size, stride=2, padding=self.padding
            ),
            nn.LeakyReLU(0.2),
        )

        # number of channels
        ncs = [self.init_nc * 2 ** n_layer for n_layer in range(n_layers - 1)]
        # every layer downsamples except for last
        strides = (len(ncs) - 1) * [2] + [1]

        for idx, (nc, stride) in enumerate(zip(ncs, strides)):
            self.add_module(self.layername(idx), self.layer(nc, stride))

        self.final_conv = nn.Conv2d(
            2 * ncs[-1], 1, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )

    @property
    def return_intermed(self):
        return self._return_intermed

    @return_intermed.setter
    def return_intermed(self, value):
        self._return_intermed = value

    @property
    def padding(self):
        return self.kernel_size // 2

    def layer(self, input_nc, stride):
        return nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    input_nc,
                    2 * input_nc,
                    self.kernel_size,
                    stride=stride,
                    padding=self.padding,
                )
            ),
            nn.InstanceNorm2d(2 * input_nc),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def layername(idx):
        return "conv_{}".format(idx)

    @property
    def intermed_layers(self):
        # The last layer is not considered a feature layer for computing feature loss,
        # since it already contributed to the generator loss.
        return [self.init_conv] + [
            getattr(self, self.layername(idx)) for idx in range(self.n_layers - 1)
        ]

    def forward(self, gen_input, real_or_fake):
        """ a conditioned discriminator's forward method

        Parameters
        ----------

        gen_input : dict of torch.Tensor
            input the generator received, i.e., the condition variable
        real_or_fake: torch.Tensor
            either the real sample or the fake one created by the generator

        """

        x = torch.cat([gi for gi in gen_input.values()] + [real_or_fake], dim=1)

        xs = []

        for layer in self.intermed_layers:
            x = layer(x)
            if self.return_intermed:
                xs.append(x)

        if self.final_conv is not None:
            x = self.final_conv(x)
        x = [x]  # make consistent with multiscale discriminator

        return common.NNOutput(final=x, features=xs)


class Multiscale(nn.Module):
    """ Multiscale discriminator

    Parameters
    ----------
    discriminators : list[nn.Module]
        list of discriminators, each will operate at a different scale

    """

    def __init__(self, discriminators):
        super().__init__()

        self.n_scales = len(discriminators)
        self._return_intermed = False

        for idx, disc in enumerate(discriminators):
            self.add_module(self.disc_name(idx), disc)

    @property
    def return_intermed(self):
        return self._return_intermed

    @return_intermed.setter
    def return_intermed(self, value):
        self._return_intermed = value
        for idx in range(self.n_scales):
            disc_name = self.disc_name(idx)
            getattr(self, disc_name).return_intermed = value

    @staticmethod
    def disc_name(idx):
        return "disc_{}".format(idx)

    def forward(self, gen_input, real_or_fake):
        """ a conditioned discriminator's forward method

        Parameters
        ----------

        gen_input : dict of torch.Tensor
            input the generator received, i.e., the condition variable
        real_or_fake: torch.Tensor
            either the real sample or the fake one created by the generator

        """

        xs = {}

        # cycle through all scales
        for idx in range(self.n_scales):
            disc_name = self.disc_name(idx)
            disc = getattr(self, disc_name)
            xs[disc_name] = disc(gen_input, real_or_fake)

            # downsample input
            gen_input = {
                k: torch.nn.functional.avg_pool2d(v, 2) for k, v in gen_input.items()
            }
            real_or_fake = torch.nn.functional.avg_pool2d(real_or_fake, 2)

        # concatenate output of discriminators
        final = [final for nno in xs.values() for final in nno.final]
        # concatenate and flatten features of different discriminators
        features = [feat for nno in xs.values() for feat in nno.features]

        return common.NNOutput(final, features)
