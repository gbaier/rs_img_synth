import argparse
import pathlib

import yaml
import torch
import torchvision.transforms.functional as TF
import matplotlib.colors
from PIL import Image
import numpy as np

import options.gan
import datasets.nrw
from utils import unwrap_state_dict


from IPython import embed

def invert_colormap(img, cmap, norm):
    img_invert = np.zeros(img.shape[:2], dtype=np.int32)
    for color, idx in zip(cmap.colors, range(int(norm.vmin), int(norm.vmax)+1)):
        # conversion from hex to rgb and rescaling
        color_rgb = matplotlib.colors.to_rgb(color)
        red, green, blue = (255*x for x in color_rgb)
        red_mask = img[:, :, 0] == red
        green_mask = img[:, :, 1] == green
        blue_mask = img[:, :, 2] == blue

        mask = np.logical_and(red_mask, green_mask)
        mask = np.logical_and(mask, blue_mask)

        img_invert[mask] = idx
    return img_invert


###################
#                 #
# Parse arguments #
#                 #
###################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--seg", type=str, help="segmentation map"
)
parser.add_argument(
    "--dem", type=str, help="digitial elevation model"
)
parser.add_argument("model", type=str)
parser.add_argument("output", type=str)
args = parser.parse_args()

########################
#                      #
# Get config and model #
#                      #
########################

OUT_DIR = pathlib.Path(args.model).absolute().parents[0]

# loading config
with open(OUT_DIR / "config.yml", "r") as stream:
    CONFIG = yaml.load(stream)
print("config: {}".format(CONFIG))

if torch.cuda.device_count() >= 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("loading model {}".format(args.model))
model = options.gan.get_generator(CONFIG)
# remove distributed wrapping, i.e. module. from keynames
state_dict = unwrap_state_dict(torch.load(args.model))
model.load_state_dict(state_dict)
model.eval()
model.to(device)

##############
#            #
# Load image #
#            #
##############

def seg2tensor(seg):
    seg = np.array(Image.open(seg))
    seg_inv = invert_colormap(seg, datasets.nrw.lcov_cmap, datasets.nrw.lcov_norm)
    seg_inv_one_hot = torch.nn.functional.one_hot(TF.to_tensor(seg_inv).long(), 11).squeeze().permute(2, 0, 1).float()
    return seg_inv_one_hot.unsqueeze(0)

def dem2tensor(dem):
    dem = np.array(Image.open(dem))
    return TF.to_tensor(dem).unsqueeze(0)

sample = {}
if args.seg:
    sample["seg"] = seg2tensor(args.seg)
if args.dem:
    sample["dem"] = dem2tensor(args.dem)

with torch.no_grad():
    fake_rgb = model({k: v.to(device) for k, v in sample.items()})


def sar2rgb(sar):
    return np.squeeze(np.clip(255*sar, 0, 255).astype(np.uint8))

# for SAR
# fake_rgb = sar2rgb(fake_rgb.squeeze().cpu().numpy())

# for RGB
fake_rgb = (fake_rgb.squeeze().cpu().numpy() * 255).astype(np.uint8)
fake_rgb = np.moveaxis(fake_rgb, 0, 2)

result = Image.fromarray(fake_rgb)
result.save(args.output)
