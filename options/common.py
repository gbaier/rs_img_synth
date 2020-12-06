""" common command line arguments and helper functions for GANs and U-Net """

import argparse
import pathlib

import torch
import torchvision

import datasets.transforms


def get_parser():
    """ returns common ArgumentParser for GANs and U-Net """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=(256,),
        nargs="+",
        help="Size of crop. Can be a tuple of height and width",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=(256,),
        nargs="+",
        help="Resizing after cropping. Can be a tuple of height and width",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of epochs to train"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--suffix", help="suffix appended to the otuput directory", default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="nrw",
        choices=["nrw", "dfc"],
        help="Which dataset to use: GeoNRWa or Data Fusion Contest 2020?",
    )
    parser.add_argument(
        "--dataroot", type=str, default="./data/nrw", help="Path to dataset"
    )

    parser.add_argument(
        "--num_workers", default=2, type=int, help="Number of workers for data loader.",
    )

    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default="./results",
        help="Where to store models, log, etc.",
    )

    return parser


def get_transforms(config):
    """ returns dataset transforms

    Parameters
    ----------
    config : dict
        configuration returned by args2dict

    Returns
    -------
    train and test transforms for the dataset

    """

    n_labels = getattr(datasets, config["dataset"]["name"]).N_LABELS

    if config["dataset"]["name"] == "nrw":
        train_transforms = [
            datasets.transforms.RandomCrop(config["training"]["crop"]),
            datasets.transforms.RandomHorizontalFlip(),
            datasets.transforms.Resize(config["training"]["resize"]),
            datasets.transforms.ToTensor(),
            datasets.transforms.TensorApply(
                seg=lambda x: torch.nn.functional.one_hot(x.long(), n_labels)
                .squeeze()
                .permute(2, 0, 1)
                .float()
            ),
        ]

        # remove resize layer if size of crop and resize are identical
        if config["training"]["crop"] == config["training"]["resize"]:
            train_transforms = [
                tt
                for tt in train_transforms
                if not isinstance(tt, datasets.transforms.Resize)
            ]

        train_transforms = torchvision.transforms.Compose(train_transforms)

        # Get test transform from train trainsform.
        # Test transform should be deterministic.
        # 1) replace random crop with center crop
        # 2) remove horizontal flip
        test_transforms = torchvision.transforms.Compose(
            [
                datasets.transforms.CenterCrop(config["training"]["crop"]),
                *train_transforms.transforms[2:],
            ]
        )
    elif config["dataset"]["name"] == "dfc":
        train_transforms = torchvision.transforms.Compose(
            [
                datasets.transforms.ToTensor(),
                datasets.transforms.TensorApply(
                    seg=lambda x: torch.nn.functional.one_hot(x.long(), n_labels)
                    .squeeze()
                    .permute(2, 0, 1)
                    .float()
                ),
            ]
        )
        test_transforms = train_transforms
    else:
        raise RuntimeError("Invalid dataset. This should never happen")
    return train_transforms, test_transforms


def get_dataset(config, split, transforms):
    """ returns dataset

    Parameters
    ----------
    config : dict
        configuration returned by args2dict
    split : string
        use train or test split
    transforms
        train or test transforms returned by get_transforms

    Returns
    -------
    dataset class

    """

    name = config["dataset"]["name"]
    root = config["dataset"]["root"]

    if name == "dfc":
        return datasets.dfc.DFC2020(root, split, transforms)
    if name == "nrw":
        # extra check whether also to load SAR acquisitions
        try:
            include_sar = config["dataset"]["output"] == "sar"
        except KeyError:
            include_sar = False
        return datasets.nrw.NRW(root, split, include_sar, transforms)
    # raising should never happen
    raise ValueError("Dataset must be nrw or dfc, but is {}".format(name))
