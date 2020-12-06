import argparse
import pathlib

import torch
import torch.nn
import torchvision
import numpy as np
import tqdm
import yaml

import datasets.nrw
import datasets.dfc
import options.gan as options
from utils import unwrap_state_dict


###############################################
#                                             #
# Parsing and checking command line arguments #
#                                             #
###############################################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model", type=str)
args = parser.parse_args()

print("loading model {}".format(args.model))

# infer output directory from model path
OUT_DIR = pathlib.Path(args.model).absolute().parents[0]

# loading config
with open(OUT_DIR / "config.yml", "r") as stream:
    CONFIG = yaml.load(stream)
print("config: {}".format(CONFIG))

train_transforms, test_transforms = options.common.get_transforms(CONFIG)

dataset = options.common.get_dataset(CONFIG, split='test', transforms=test_transforms)


###########
#         #
# Testing #
#         #
###########

if torch.cuda.device_count() >= 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = options.get_generator(CONFIG)
# remove distributed wrapping, i.e. module. from keynames
state_dict = unwrap_state_dict(torch.load(args.model))
model.load_state_dict(state_dict)
model.eval()
model.to(device)


############
#          #
# Plotting #
#          #
############

BATCH_SIZE = 8

test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

with torch.no_grad():
    for idx, sample in tqdm.tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        imgs = []

        gen_input = {dt: sample[dt] for dt in CONFIG["dataset"]["input"]}

        fake = model({k: v.to(device) for k, v in gen_input.items()}).cpu()
        real = sample[CONFIG["dataset"]["output"]]

        if CONFIG["dataset"]["output"] == "sar":
            fake = [
                np.moveaxis(dataset.sar2rgb(np.moveaxis(x.numpy(), 0, -1)), -1, 0)
                for x in fake.clone().detach()
            ]
            fake = torch.tensor(fake).float()

            real = [
                np.moveaxis(dataset.sar2rgb(np.moveaxis(x.numpy(), 0, -1)), -1, 0)
                for x in real.clone().detach()
            ]
            real = torch.tensor(real).float()

        if "dem" in gen_input:
            depth_as_rgb = [
                np.moveaxis(dataset.depth2rgb(x.squeeze().numpy()), -1, 0)
                for x in sample["dem"].clone().detach()
            ]
            depth_as_rgb = torch.tensor(depth_as_rgb).float()
            imgs.append(depth_as_rgb)

        if "seg" in gen_input:
            seg_no_one_hot = torch.argmax(sample["seg"], 1).unsqueeze(1)
            seg_as_rgb = [
                np.moveaxis(dataset.seg2rgb(x.squeeze()), -1, 0) for x in seg_no_one_hot
            ]
            seg_as_rgb = torch.tensor(seg_as_rgb).float()
            imgs.append(seg_as_rgb)

        if "sar" in gen_input:
            sar_as_rgb = [
                np.moveaxis(dataset.sar2rgb(np.moveaxis(x, 0, -1)), -1, 0)
                for x in sample["sar"].clone().detach().numpy()
            ]
            sar_as_rgb = torch.tensor(sar_as_rgb).float()
            imgs.append(sar_as_rgb)

        imgs.append(fake)
        imgs.append(real)

        torchvision.utils.save_image(
            torch.cat(imgs), OUT_DIR / "{:04}.jpg".format(idx),
        )
