import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F


#########################
#                       #
# Basic building blocks #
#                       #
#########################


class ResnetBasicBlock(nn.Module):
    """ ResNet block """

    def __init__(self, inplanes, planes, kernel_size=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.kernel_size = kernel_size

        if inplanes != planes:
            self.shortcut = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        else:
            self.shortcut = None

        self.pad = nn.ReflectionPad2d(self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = spectral_norm(nn.Conv2d(inplanes, planes, self.kernel_size))
        self.conv2 = spectral_norm(nn.Conv2d(planes, planes, self.kernel_size))

        if norm_layer is not None:
            self.norm1 = norm_layer(inplanes)
            self.norm2 = norm_layer(planes)
        else:
            self.norm1 = self.norm2 = lambda x: x

    def forward(self, x):
        """ Ordering of operations differs in ResNet blocks depending on the publication.

        [1] : conv -> norm -> relu
        [2] and [3] : norm -> relu -> conv

        We follow [2] and [3] as they specifically target image synthesis.

        [1] He et. al. "Deep residual learning for image recognition, CVPR 2016
        [2] Brock et. al. "Large scale GAN training for high fidelity natural
                           image synthesis", ICLR 2019
        [3] Park et. al. "Semantic Image Synthesis with Spatially-Adaptive
                          Normalization.", CVPR 2019

        """

        identity = x

        out = self.pad(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.pad(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.shortcut:
            identity = self.shortcut(identity)
        out += identity

        return out


class Downsampler(nn.Module):
    """ Typical downsampler by strided convolution, norm layer and ReLU """

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(input_nc, 2 * input_nc, kernel_size=3, stride=2, padding=1)
        )
        self.norm = norm_layer(2 * input_nc)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


##########################
#                        #
# SPADE buildings blocks #
#                        #
##########################


class SPADE(nn.Module):
    """ SPADE normalization layer

    Code taken and modified from
    https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py

    SPADE consists of two steps. First, it normalizes the activations using
    your favorite normalization method, such as Batch Norm or Instance Norm.
    Second, it applies scale and bias to the normalized output, conditioned on
    the segmentation map.

    Parameters
    ----------
    num_features : int
        The number of channels of the normalized activations,
        i.e., SPADE's output dimension
    label_nc: int
        The number of channels of the input semantic map,
        i.e., SPADE's  input dimension
    norm_layer: torch.nn.BatchNorm2d or InstanceNorm2d.
        Which normalization method to use together with SPADE.
        Generators often use batch or instance normalization.

    """

    def __init__(self, num_features, label_nc, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.norm = norm_layer(num_features)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        conv_opts = {
            "kernel_size": 3,
            "padding": 1,
        }

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, **conv_opts), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, num_features, **conv_opts)
        self.mlp_beta = nn.Conv2d(nhidden, num_features, **conv_opts)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class DataSegmapTuple:

    data: torch.tensor
    segmap: torch.tensor

    def __init__(self, data, segmap):
        self.data = data
        self.segmap = segmap

    def __add__(self, other):
        return DataSegmapTuple(self.data + other.data, self.segmap)


def pass_segmap(method):
    """ SPADE normalization requires the segmentation map as additional input

    By wrapping all forward methods with this wrapper SPADE blocks can be used
    just like regular nn.Modules, i.e.

    x = block_1(x)
    x = block_2(x)

    return x

    where the forward methods of block_1 and block_2 was wrapped using this function.

    ToDo:
    checkout register_forward_hook and register_forward_pre_hook

    """

    def wrapper(dst: DataSegmapTuple):
        # treat SPADE modules differently
        if isinstance(method.__self__, SPADE):
            segmap_ds = F.interpolate(
                dst.segmap, size=dst.data.size()[2:], mode="nearest"
            )
            x = method(dst.data, segmap_ds)
        else:
            x = method(dst.data)
        return DataSegmapTuple(
            x, dst.segmap
        )  # return new features and segmap of original size

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


class SPADEResnetBlock(ResnetBasicBlock):
    def __init__(self, inplanes, planes, label_nc, kernel_size=3):
        super().__init__(
            inplanes,
            planes,
            kernel_size,
            lambda num_features: SPADE(num_features, label_nc),
        )

        for name, child in self.named_children():
            # wraps all contained modules to pass the segmentation map
            # together with the feature tensor
            child.forward = pass_segmap(child.forward)
            setattr(self, name, child)

    def forward(self, x, segmap):
        # Call the base class's forward method but pass in a
        # DataSegmapTuple instead of just a tensor. Since all
        # child modules were modified with pass_segmap they can
        # process the DataSegmapTuple
        dst = super().forward(DataSegmapTuple(x, segmap))

        return dst.data


class SpadeDownsampler(Downsampler):
    def __init__(self, input_nc, label_nc):
        super().__init__(input_nc, lambda num_features: SPADE(num_features, label_nc))

        for name, child in self.named_children():
            # wraps all contained modules to pass the segmentation map
            # together with the feature tensor
            child.forward = pass_segmap(child.forward)
            setattr(self, name, child)

    def forward(self, x, segmap):
        # Call the base class's forward method but pass in a
        # DataSegmapTuple instead of just a tensor. Since all
        # child modules were modified with pass_segmap they can
        # process the DataSegmapTuple
        dst = super().forward(DataSegmapTuple(x, segmap))

        return dst.data
