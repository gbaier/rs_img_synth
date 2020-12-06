import argparse
import pathlib

import yaml
import torch
import torch.nn.functional as TF
import tqdm

import numpy as np
from scipy import linalg

import options.gan
import datasets.nrw
import datasets.dfc
import models.unet
import utils


##########################
#                        #
# Comannd line arguments #
#                        #
##########################

parser = argparse.ArgumentParser(
    description="compare real and fake images using a segmentation network"
)
parser.add_argument("generator", help="pt file of the generator model")
parser.add_argument("segmentor", help="pt file of the segmentation model")
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("This scripts expects CUDA to be available")

# Get the directory of the generator and load its configuration.
# From the configuration we also get the dataset, etc.
GEN_DIR = pathlib.Path(args.generator).absolute().parents[0]
with open(GEN_DIR / "config.yml", "r") as stream:
    CONFIG = yaml.load(stream)


#########
#       #
# U-Net #
#       #
#########

# Create a U-Net model and load weights
dset_class = getattr(datasets, CONFIG["dataset"]["name"])
n_labels = dset_class.N_LABELS
# Number of channels from the generator's output defines
# The U-Net's number of input channels
input_nc = dset_class.N_CHANNELS[CONFIG["dataset"]["output"]]

seg_model = models.unet.UNet(input_nc, n_labels).to(device)
seg_model.load_state_dict(torch.load(args.segmentor).state_dict())
seg_model.to(device).eval()
# return intermediate features to compute FID
seg_model.return_intermed = True

#############
#           #
# Generator #
#           #
#############

gen_model = options.gan.get_generator(CONFIG)
# remove distributed wrapping, i.e. module. from keynames
state_dict = utils.unwrap_state_dict(torch.load(args.generator))
gen_model.load_state_dict(state_dict)
gen_model = gen_model.to(device).eval()

################
#              #
# Dataset prep #
#              #
################

_, test_transforms = options.common.get_transforms(CONFIG)
dataset_test = options.common.get_dataset(
    CONFIG, split="test", transforms=test_transforms
)
# dataset_test = torch.utils.data.Subset(dataset_test, list(range(16)))

BATCH_SIZE = 8

test_dataloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=BATCH_SIZE, num_workers=4
)


def process_intermed_features(intermed_features):
    """ processes intermediate features before computing FID

    Applies global average pooling to the features of all layers and
    concatenates the resulting feature maps to a single vector.

    Parameters
    ----------

    intermed_features: list of pytorch tensors
        each element of the list contains a batch of intermediate features from a different layer

    """

    # global average pooling of spatial dimensions
    pooled = [TF.avg_pool2d(t, t.shape[-2:]).squeeze() for t in intermed_features]

    # concatenate features of different layers
    concat = torch.cat(pooled, dim=1)

    return concat.tolist()


# Taken from https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# Taken from https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
def calculate_activation_statistics(features):
    """Calculation of the statistics used by the FID.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    features = np.array(features)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


###########
#         #
# Testing #
#         #
###########

with torch.no_grad():
    real = []
    fake = []
    for idx, sample in tqdm.tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        sample = {k: v.to(device) for k, v in sample.items()}

        # generator fake images
        gen_input = {dt: sample[dt] for dt in CONFIG["dataset"]["input"]}
        gen_output = gen_model(gen_input)

        # get features from real and fake images
        seg_real, features_real = seg_model(sample[CONFIG["dataset"]["output"]])
        seg_fake, features_fake = seg_model(gen_output)

        real += process_intermed_features(features_real)
        fake += process_intermed_features(features_fake)

    mu_real, sigma_real = calculate_activation_statistics(real)
    mu_fake, sigma_fake = calculate_activation_statistics(fake)

    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print("FID: {:5.4f}".format(fid_value))
