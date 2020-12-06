import torch.nn as nn
import torch.nn.functional
import torch.nn.utils.spectral_norm as spectral_norm

from . import arch


##############################
#                            #
# Generator buildings blocks #
#                            #
##############################


class Encoder(nn.Module):
    def __init__(
        self,
        input_nc,
        init_nc=64,
        init_kernel_size=7,
        n_downsample=4,
        downsampler=arch.Downsampler,
    ):
        super().__init__()

        self.input_nc = input_nc

        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(init_kernel_size // 2),
            spectral_norm(nn.Conv2d(input_nc, init_nc, init_kernel_size, padding=0)),
            nn.ReLU(),
        )

        self.n_downsample = n_downsample

        # nn.sequential does not support multiple inputs
        for i in range(n_downsample):
            self.add_module("down{}".format(i), downsampler(init_nc * 2 ** i))

    def forward(self, x, segmap=None):
        # no normalization layer in first convolution
        x = self.init_conv(x)

        for i in range(self.n_downsample):
            down = getattr(self, "down{}".format(i))
            if segmap is not None:
                x = down(x, segmap)
            else:
                x = down(x)

        return x


class Body(nn.Module):
    def __init__(self, input_nc, n_stages, res_block=arch.ResnetBasicBlock):

        super().__init__()

        self.n_stages = n_stages

        for i in range(n_stages):
            self.add_module("body{}".format(i), res_block(input_nc, input_nc))

    def forward(self, x, segmap=None):
        """ expects output either from Encoder or an inital convolution layer"""

        for i in range(self.n_stages):
            rnb = getattr(self, "body{}".format(i))
            if segmap is not None:
                x = rnb(x, segmap)
            else:
                x = rnb(x)

        return x


class Decoder(nn.Module):
    """ Decoder part for an image synthesis generator.

    This decoder follow the philosophy of [1] and [2] by pairing ResNet-blocks with simple upsampling.
    Upsampling is done using nearest neighbour to prevent checkerboard artifacts.
    Decoder must be used together with Body, since both [1] and [2] have some layers without upsampling.

    [1] Brock et. al. "Large scale GAN training for high fidelity natural image synthesis", ICLR 2019
    [2] Park et. al. "Semantic Image Synthesis with Spatially-Adaptive Normalization.", CVPR 2019
    [3] Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.00003

    """

    def __init__(
        self,
        input_nc,
        output_nc=3,
        n_up_stages=4,
        res_block=arch.ResnetBasicBlock,
        final_kernel_size=3,
        return_intermed=False,
    ):

        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_up_stages = n_up_stages
        self._return_intermed = return_intermed

        for i in range(n_up_stages):
            in_channels = self.input_nc // (2 ** i)
            out_channels = in_channels // 2
            self.add_module("up{}".format(i), res_block(in_channels, out_channels))

        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(final_kernel_size // 2),
            spectral_norm(
                nn.Conv2d(out_channels, output_nc, final_kernel_size, padding=0)
            ),
            nn.Tanh(),
        )

    def input_shape(self, output_shape):
        """ given the output spatial dimension compute the dimension the input has to have """
        return tuple((x // 2 ** self.n_up_stages) for x in output_shape)

    @property
    def return_intermed(self):
        return self._return_intermed

    @return_intermed.setter
    def return_intermed(self, value):
        self._return_intermed = value

    def forward(self, x, segmap=None):
        """ expects output from Body """

        xs = []

        for i in range(self.n_up_stages):
            # simple nearest neighbour upsampling
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            rnb = getattr(self, "up{}".format(i))
            if segmap is not None:
                x = rnb(x, segmap)
            else:
                x = rnb(x)

            if self.return_intermed:
                xs.append(x)
        if self.return_intermed:
            return xs

        return self.final_conv(x)


#############################
#                           #
# Generator implementations #
#                           #
#############################


class ResnetEncoderDecoder(nn.Module):
    """ Encoder-Decoder architecture heavily inspired by [1], [2] and [3] for style transfer also used in Pix2Pix and Pix2PixHD

    The default parameterization corresponds to the global generator of the Pix2Pix HD [2].

    There are slight differences between [1] and [2]. [2] uses reflection padding for all Resnet blocks, [1] notes that zero-padding let to
    artifacts and only pads in at the initial convolution layer, i.e. each Resnet block reduces the spatial dimensions.

    [1] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. “Perceptual Losses for Real-Time Style Transfer and Super-Resolution.”, ECCV 2016
        https://arxiv.org/abs/1603.08155 and supplementary material https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    [2] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro.
        "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs", CVPR, 2018 https://arxiv.org/abs/1711.11585

    [3] Park et. al. "Semantic Image Synthesis with Spatially-Adaptive Normalization.", CVPR 2019, https://nvlabs.github.io/SPADE/


    """

    def __init__(
        self,
        input_nc,
        init_nc=64,
        output_nc=3,
        init_kernel_size=7,
        n_downsample=4,
        n_resnet_blocks=9,
    ):
        super().__init__()

        self.encoder = Encoder(input_nc, init_nc, init_kernel_size, n_downsample)

        body_nc = init_nc * 2 ** n_downsample

        self.body = Body(body_nc, n_resnet_blocks)

        self.decoder = Decoder(body_nc, output_nc, n_downsample)

    def forward(self, gen_input):
        x = torch.cat([gi for gi in gen_input.values()], dim=1)

        x = self.encoder(x)
        x = self.body(x)
        x = self.decoder(x)

        return x

    @property
    def input_nc(self):
        return self.encoder.input_nc


class SPADEResnetEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_nc,
        label_nc,
        init_nc=64,
        output_nc=3,
        init_kernel_size=7,
        n_downsample=4,
        n_resnet_blocks=9,
    ):
        super().__init__()

        get_spade_block = lambda inc, outc: arch.SPADEResnetBlock(inc, outc, label_nc)
        get_spade_downsampler = lambda inc: arch.SpadeDownsampler(inc, label_nc)

        self.encoder = Encoder(
            input_nc, init_nc, init_kernel_size, n_downsample, get_spade_downsampler
        )

        body_nc = init_nc * 2 ** n_downsample

        self.body = Body(body_nc, n_resnet_blocks, get_spade_block)

        self.decoder = Decoder(body_nc, output_nc, n_downsample, get_spade_block)

    def forward(self, gen_input):
        # get segmentation map
        segmap = gen_input["seg"]

        # concatenate all other inputs
        x = torch.cat([v for k, v in gen_input.items() if k != "seg"], dim=1)

        x = self.encoder(x, segmap)
        x = self.body(x, segmap)
        x = self.decoder(x, segmap)

        return x


class SPADEGenerator(nn.Module):
    def __init__(
        self, label_nc, init_nc=256, output_nc=3, n_mid_stages=2, n_up_stages=4
    ):
        super().__init__()

        self.label_nc = label_nc

        # iniital embedding of segmentation map
        init_kernel_size = 3
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(init_kernel_size // 2),
            spectral_norm(nn.Conv2d(label_nc, init_nc, init_kernel_size, padding=0)),
            nn.ReLU(),
        )

        get_spade_block = lambda inc, outc: arch.SPADEResnetBlock(inc, outc, label_nc)

        self.head = Body(init_nc, n_mid_stages, get_spade_block)

        self.decoder = Decoder(init_nc, output_nc, n_up_stages, get_spade_block)

    def forward(self, gen_input):
        segmap = gen_input["seg"]
        # Input dimension deterime final output dimension
        shape_ds = self.decoder.input_shape(segmap.shape[-2:])
        segmap_ds = torch.nn.functional.interpolate(segmap, size=shape_ds)

        x = self.init_conv(segmap_ds)

        x = self.head(x, segmap)
        x = self.decoder(x, segmap)

        return x

    @property
    def input_nc(self):
        return self.label_nc
