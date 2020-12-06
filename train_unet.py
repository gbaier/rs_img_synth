# Script for distributed training with synchronized batch norm
# Guide for converting a regular training script to a distributed one can
# be found at https://github.com/dougsouza/pytorch-sync-batchnorm-example

import argparse
import datetime
import numbers
import pathlib
import warnings
import logging

import torch
import tqdm
import yaml
import numpy as np

import loss
import options.segment
import datasets.nrw
import datasets.dfc
import models.unet

logger = logging.getLogger(__name__)


def train_and_eval(
    model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs, device
):

    pbar = tqdm.tqdm(total=num_epochs)
    for n_epoch in range(num_epochs):

        loss_train, iou_val_train = train(model, optimizer, criterion, train_dataloader, 'rgb', device)

        loss_test, iou_val_test = evaluate(
            model, criterion, test_dataloader, 'rgb', device
        )

        info_str = "epoch {:3d} train: loss={:6.3f} iou={:6.3f}, test loss={:6.3f} iou={:6.3f}".format(
            n_epoch, loss_train, loss_test, iou_val_train, iou_val_test
        )

        logger.info(info_str)

        pbar.update(1)
        pbar.set_description(info_str)
        pbar.write(info_str)
    return model


def evaluate(model, criterion, dataloader, src="rgb", device='cuda'):
    """ evaluates a model on the given dataset

    Parameters
    ----------

    model: (torch.nn.Module)
        the neural network
    optimizer: (torch.optim)
        optimizer for parameters of model
    criterion: unction
        takes batch_output and batch_labels and computes the loss for the batch
    dataloader: torch.utils.data.DataLoader
        fetches training data

    """

    model.eval()

    running_loss = 0.0
    iou_val = 0.0
    for sample in dataloader:
        sample = {k: v.to(device) for k, v in sample.items()}

        with torch.no_grad():
            output = model(sample[src])

        pred = (output > 0).long()

        loss_t = criterion(output, sample["seg"])
        running_loss += loss_t.item()
        iou_val += loss.iou(pred, sample["seg"]).item()

    running_loss /= len(dataloader)
    iou_val /= len(dataloader)

    return running_loss, iou_val


def train(model, optimizer, criterion, dataloader, src="rgb", device='cuda'):
    """ trains a model for one epoch

    Parameters
    ----------

    model: (torch.nn.Module)
        the neural network
    optimizer: (torch.optim)
        optimizer for parameters of model
    loss_fn: unction
        takes batch_output and batch_labels and computes the loss for the batch
    dataloader: torch.utils.data.DataLoader
        fetches training data

    """

    model.train()

    running_loss = 0.0
    iou_val = 0.0
    for sample in dataloader:
        optimizer.zero_grad()

        sample = {k: v.to(device) for k, v in sample.items()}

        output = model(sample[src])
        pred = (output > 0).long()

        loss_t = criterion(output, sample["seg"])

        loss_t.backward()
        running_loss += loss_t.item()
        iou_val += loss.iou(pred, sample["seg"]).item()
        optimizer.step()

    running_loss /= len(dataloader)
    iou_val /= len(dataloader)

    return running_loss, iou_val


if __name__ == "__main__":

    parser = options.segment.get_parser()
    args = parser.parse_args()

    # Reproducibilty config https://pytorch.org/docs/stable/notes/randomness.html
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if len(args.crop) == 1:
        args.crop = args.crop[0]

    if len(args.resize) == 1:
        args.resize = args.resize[0]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("This scripts expects CUDA to be available")
    OUT_DIR = args.out_dir / options.segment.args2str(args)
    # All process make the directory.
    # This avoids errors when setting up logging later due to race conditions.
    OUT_DIR.mkdir(exist_ok=True)

    ###########
    #         #
    # Logging #
    #         #
    ###########

    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        level=logging.INFO,
        filename=OUT_DIR / "log_training.txt",
    )
    logger = logging.getLogger()
    logger.info("Saving logs, configs and models to %s", OUT_DIR)

    CONFIG = options.segment.args2dict(args)
    with open(OUT_DIR / "config.yml", "w") as cfg_file:
        yaml.dump(CONFIG, cfg_file)

    #########################
    #                       #
    # Dataset configuration #
    #                       #
    #########################

    train_transforms, test_transforms = options.common.get_transforms(CONFIG)

    dataset_train = options.common.get_dataset(CONFIG, split='train', transforms=train_transforms)
    dataset_test = options.common.get_dataset(CONFIG, split='test', transforms=test_transforms)

    print("training dataset statistics")
    print(dataset_train)

    print("testing dataset statistics")
    print(dataset_test)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ###############
    #             #
    # Model setup #
    #             #
    ###############

    criterion = torch.nn.BCEWithLogitsLoss()

    dset_class = getattr(datasets, CONFIG["dataset"]["name"])
    n_labels = dset_class.N_LABELS
    input_nc = dset_class.N_CHANNELS[CONFIG["dataset"]["input"]]
    model = models.unet.UNet(input_nc, n_labels).to(device)

    ############
    #          #
    # TRAINING #
    #          #
    ############

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    train_and_eval(
        model,
        optimizer,
        criterion,
        dataloader_train,
        dataloader_test,
        num_epochs=args.epochs,
        device=device,
    )

    ##############
    #            #
    # save model #
    #            #
    ##############

    torch.save(model, OUT_DIR / "{}.pt".format(options.segment.args2str(args)))
