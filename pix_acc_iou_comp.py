import argparse
import pathlib
import logging

import yaml
import torch
import tqdm
import numpy as np
import matplotlib
# headless
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ignite

# shitty workaround to include main directory in the python path
import sys
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

import options.segment
import options.gan
import datasets.nrw
import datasets.dfc
import models.unet
import utils
import loss


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
parser.add_argument("out_dir", type=pathlib.Path, help="output directory")
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("This scripts expects CUDA to be available")

OUT_DIR = args.out_dir
OUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    level=logging.INFO,
    filename=OUT_DIR / "log_segmentatation_analysis.txt",
)
logger = logging.getLogger()
logger.info("Saving logs, configs and models to %s", OUT_DIR)

GEN_DIR = pathlib.Path(args.generator).absolute().parents[0]
# loading config
with open(GEN_DIR / "config.yml", "r") as stream:
    CONFIG = yaml.load(stream)
logging.info("Generator config: {}".format(CONFIG))

################
#              #
# Dataset prep #
#              #
################

_, test_transforms = options.common.get_transforms(CONFIG)
dataset_test = options.common.get_dataset(CONFIG, split="test", transforms=test_transforms)
#dataset_test = torch.utils.data.Subset(dataset_test, list(range(16)))

BATCH_SIZE = 8

test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE)

###############
#             #
# Model setup #
#             #
###############


dset_class = getattr(datasets, CONFIG["dataset"]["name"])
n_labels = dset_class.N_LABELS
# get channels from output from generator
input_nc = dset_class.N_CHANNELS[CONFIG["dataset"]["output"]]
seg_model = models.unet.UNet(input_nc, n_labels).to(device)

seg_model.load_state_dict(torch.load(args.segmentor).state_dict())
seg_model.to(device).eval()

gen_model = options.gan.get_generator(CONFIG)
# remove distributed wrapping, i.e. module. from keynames
state_dict = utils.unwrap_state_dict(torch.load(args.generator))
gen_model.load_state_dict(state_dict)
gen_model = gen_model.to(device).eval()

###########
#         #
# Testing #
#         #
###########

def vis_sample(real_img, fake_img,  true_seg, pred_seg_real, pred_seg_fake):

    fig = plt.figure(figsize=(6., 4.2))
    fig.subplots_adjust(top=.95, bottom=0.0, left=0.0, right=1.0, hspace=0.01, wspace=0.01)

    ax_rgb = fig.add_subplot(2, 3, 2)
    ax_rgb.imshow(real_img)
    ax_rgb.axis('off')
    ax_rgb.set_title('real')

    ax_rgb = fig.add_subplot(2, 3, 3)
    ax_rgb.imshow(fake_img)
    ax_rgb.axis('off')
    ax_rgb.set_title('fake')

    ax_seg_true = fig.add_subplot(2, 3, 4)
    ax_seg_true.imshow(dataset_test.seg2rgb(true_seg))
    ax_seg_true.axis('off')
    ax_seg_true.set_title('ground truth')

    ax_seg_pred = fig.add_subplot(2, 3, 5)
    ax_seg_pred.imshow(dataset_test.seg2rgb(pred_seg_real))
    ax_seg_pred.axis('off')

    ax_seg_pred = fig.add_subplot(2, 3, 6)
    ax_seg_pred.imshow(dataset_test.seg2rgb(pred_seg_fake))
    ax_seg_pred.axis('off')

    return fig

from ignite.metrics import Accuracy, IoU, mIoU, ConfusionMatrix
from ignite.engine import Engine, create_supervised_evaluator 
from ignite.metrics.metrics_lambda import MetricsLambda
from typing import Optional

def cmAccuracy(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates accuracy using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
    Returns:
        MetricsLambda
    """
    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)

    correct_pixels = cm.diag()
    total_class_pixels = cm.sum(dim=1)

    pix_accs = correct_pixels / (total_class_pixels + 1e-15)

    if ignore_index is not None:

        def ignore_index_fn(pix_accs_vector):
            if ignore_index >= len(pix_accs_vector):
                raise ValueError(
                    "ignore_index {} is larger than the length of pix_accs vector {}".format(ignore_index, len(pix_accs_vector))
                )
            indices = list(range(len(pix_accs_vector)))
            indices.remove(ignore_index)
            return pix_accs_vector[indices]

        return MetricsLambda(ignore_index_fn, pix_accs)
    else:
        return pix_accs


def output_transform(y_pred_and_y):
    y_pred, y = y_pred_and_y
    # remove one one encoding
    y = torch.argmax(y, dim=1)
    return y_pred, y

def make_engine(process_function):
    evaluator = Engine(process_function)

    cm = ConfusionMatrix(num_classes=getattr(datasets, CONFIG["dataset"]["name"]).N_LABELS, output_transform=output_transform)
    IoU(cm, ignore_index=0).attach(evaluator, 'IoU')
    mIoU(cm, ignore_index=0).attach(evaluator, 'mIoU')
    Accuracy(output_transform=output_transform).attach(evaluator, 'Accuracy')
    cmAccuracy(cm, ignore_index=0).attach(evaluator, 'ClasswiseAccuracy')

    return evaluator

def log_metrics(metrics):
    logging.info("mIoU: {:0>6.4f}".format(metrics['mIoU']))
    logging.info("class-wise IoU:")
    for ds_class, iou_val in zip(getattr(datasets, CONFIG["dataset"]["name"]).classes, metrics['IoU']):
        logging.info("{:>40s}: {:0>6.4f}".format(ds_class, iou_val))
    logging.info("pixel accuracy: {:0>6.4f}".format(metrics['Accuracy']))
    logging.info("class-wise pixel accuracy:")
    for ds_class, iou_val in zip(getattr(datasets, CONFIG["dataset"]["name"]).classes, metrics['ClasswiseAccuracy']):
        logging.info("{:>40s}: {:0>6.4f}".format(ds_class, iou_val))


#######################
#                     #
# Validation original #
#                     #
#######################

def validation_step_original(engine, batch):
    seg_model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        y_pred = seg_model(batch[CONFIG["dataset"]["output"]])
        y = batch["seg"]
        return y_pred, y

evaluator = make_engine(validation_step_original)
state = evaluator.run(test_dataloader)
logging.info("real + ground truth labels")
logging.info("==========================")
log_metrics(evaluator.state.metrics)


###########################################
#                                         #
# Validation with respect to ground truth #
#                                         #
###########################################

def validation_step_wrt_gt(engine, batch):
    seg_model.eval()
    with torch.no_grad():
        gen_input = {dt: batch[dt].to(device) for dt in CONFIG["dataset"]["input"]}
        gen_output = gen_model(gen_input)
        y_pred = seg_model(gen_output)
        y = batch["seg"].to(device)
        return y_pred, y

evaluator = make_engine(validation_step_wrt_gt)
state = evaluator.run(test_dataloader)
logging.info("fake + ground truth labels")
logging.info("==========================")
log_metrics(evaluator.state.metrics)


#######################################
#                                     #
# Validation with respect to Original #
#                                     #
#######################################

def validation_step_wrt_gt(engine, batch):
    seg_model.eval()
    with torch.no_grad():
        gen_input = {dt: batch[dt].to(device) for dt in CONFIG["dataset"]["input"]}
        gen_output = gen_model(gen_input)
        y_pred = seg_model(gen_output)
        y = seg_model(batch[CONFIG["dataset"]["output"]].to(device))
        return y_pred, y

evaluator = make_engine(validation_step_wrt_gt)
state = evaluator.run(test_dataloader)
logging.info("fake + labels from real")
logging.info("=======================")
log_metrics(evaluator.state.metrics)

############
#          #
# Plotting #
#          #
############


with torch.no_grad():
    for idx, sample in tqdm.tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        sample = {k: v.to(device) for k,v in sample.items()}

        # generator fake images
        gen_input = {dt: sample[dt] for dt in CONFIG["dataset"]["input"]}
        gen_output = gen_model(gen_input)

        seg_real = seg_model(sample[CONFIG["dataset"]["output"]])
        est_real = torch.argmax(seg_real, dim=1)
        est_real_one_hot = torch.nn.functional.one_hot(est_real, seg_real.shape[1]).permute(0, 3, 1, 2)

        seg_fake = seg_model(gen_output)
        est_fake = torch.argmax(seg_fake, dim=1)
        est_fake_one_hot = torch.nn.functional.one_hot(est_fake, seg_fake.shape[1]).permute(0, 3, 1, 2)

        seg_gt = sample["seg"]
        seg_gt_not_one_hot = torch.argmax(seg_gt, 1)

        for idy, (rgb_real, rgb_fake, true_seg, pred_seg_real, pred_seg_fake) in enumerate(zip(sample[CONFIG["dataset"]["output"]], gen_output, sample["seg"], est_real, est_fake)):
            rgb_real = np.moveaxis(rgb_real.cpu().numpy(), 0, 2)
            rgb_fake = np.clip(np.moveaxis(rgb_fake.cpu().numpy(), 0, 2), 0, 1)
            true_seg = torch.argmax(true_seg, 0).cpu().numpy()
            pred_seg_real = pred_seg_real.cpu().numpy()
            pred_seg_fake = pred_seg_fake.cpu().numpy()
            fig = vis_sample(rgb_real, rgb_fake, true_seg, pred_seg_real, pred_seg_fake)
            fig.savefig(OUT_DIR / "{:04d}_{:04d}.jpg".format(idx, idy), dpi=200)
            plt.close(fig)
