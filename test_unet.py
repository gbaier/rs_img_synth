import argparse
import pathlib

import numpy as np
import tqdm
import yaml
import torch
import matplotlib

# headless
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import options.segment
import datasets.nrw
import datasets.dfc
import models.unet

##########################
#                        #
# Comannd line arguments #
#                        #
##########################

parser = argparse.ArgumentParser(description="apply a model to the test data set")
parser.add_argument("model", help="pt file of the segmentation model")
args = parser.parse_args()

print("loading model {}".format(args.model))

OUT_DIR = pathlib.Path(args.model).absolute().parents[0]
# loading config
with open(OUT_DIR / "config.yml", "r") as stream:
    CONFIG = yaml.load(stream)
print("config: {}".format(CONFIG))


train_transforms, test_transforms = options.common.get_transforms(CONFIG)

dataset_test = options.common.get_dataset(
    CONFIG, split="test", transforms=test_transforms
)

###########
#         #
# Testing #
#         #
###########

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("This scripts expects CUDA to be available")

dset_class = getattr(datasets, CONFIG["dataset"]["name"])
n_labels = dset_class.N_LABELS
input_nc = dset_class.N_CHANNELS[CONFIG["dataset"]["input"]]

model = models.unet.UNet(input_nc, n_labels).to(device)
model.load_state_dict(torch.load(args.model).state_dict())
model.to(device).eval()


def vis_sample(input_img, true_seg, pred_seg):

    fig = plt.figure(figsize=(6.05, 2))
    fig.subplots_adjust(
        top=1.0, bottom=0.0, left=0.0, right=1.0, hspace=0.01, wspace=0.01
    )

    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_rgb.imshow(input_img)
    ax_rgb.axis("off")

    ax_seg_pred = fig.add_subplot(1, 3, 2)
    ax_seg_pred.imshow(dataset_test.seg2rgb(pred_seg))
    ax_seg_pred.axis("off")

    ax_seg_true = fig.add_subplot(1, 3, 3)
    ax_seg_true.imshow(dataset_test.seg2rgb(true_seg))
    ax_seg_true.axis("off")

    return fig


def vis_single(img):
    fig = plt.figure(figsize=(1, 1))
    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")

    return fig


test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1)

RGB_DIR = OUT_DIR / "rgb"
LABEL_DIR = OUT_DIR / "label"
SEGM_DIR = OUT_DIR / "segm"

for img_dir in [RGB_DIR, LABEL_DIR, SEGM_DIR]:
    img_dir.mkdir(exist_ok=True)

for idx, sample in tqdm.tqdm(enumerate(dataset_test), total=len(dataset_test)):
    asdf = sample[CONFIG["dataset"]["input"]].to(device).unsqueeze(0)
    output = model(asdf)
    pred = torch.argmax(output, dim=1)

    input_img = np.moveaxis(sample[CONFIG["dataset"]["input"]].numpy(), 0, -1)
    true_seg = torch.argmax(sample["seg"], 0).numpy()
    pred_seg = pred.cpu().squeeze().numpy()

    fig = vis_sample(input_img, true_seg, pred_seg)
    fig.savefig(OUT_DIR / "{:04d}.jpg".format(idx), dpi=200)

    fig_rgb = vis_single(input_img)
    fig_rgb.savefig(RGB_DIR / "{:04d}.jpg".format(idx), dpi=200)

    fig_seg = vis_single(dataset_test.seg2rgb(pred_seg))
    fig_seg.savefig(SEGM_DIR / "{:04d}.png".format(idx), dpi=200)

    fig_lab = vis_single(dataset_test.seg2rgb(true_seg))
    fig_lab.savefig(LABEL_DIR / "{:04d}.png".format(idx), dpi=200)

    plt.close(fig)
    plt.close(fig_rgb)
    plt.close(fig_seg)
    plt.close(fig_lab)
