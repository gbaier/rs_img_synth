""" command line arguments and helper functions for GANs """

import argparse
import datetime

import models.generator
import models.discriminator
import datasets
from . import common


def get_parser():
    """ returns ArgumentParser for GANs """

    # Get common ArgumentParser.
    parser = common.get_parser()

    # Add GAN specific arguments to the parser.
    parser.add_argument(
        "--input",
        nargs="+",
        default="seg",
        help="Input of the generator. Depends on the dataset.",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Concatenate inputs before feeding the generator.",
    )
    parser.add_argument(
        "--output",
        default="rgb",
        choices=["rgb", "sar"],
        help="Output of the generator.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="for distributed training"
    )
    parser.add_argument(
        "--n_sampling",
        default=4,
        type=int,
        help="number of upsampling/downsampling in the generator",
    )
    parser.add_argument(
        "--model_cap",
        default=64,
        type=int,
        choices=[16, 32, 48, 64],
        help="Model capacity, i.e. number of features.",
    )
    parser.add_argument(
        "--n_scales",
        default=2,
        type=int,
        help="Number of scales for multiscale discriminator",
    )
    parser.add_argument(
        "--lbda", default=None, type=float, help="weighting of feature loss"
    )

    return parser


def args2str(args):
    """ converts arguments to string

    Parameters
    ----------
    args: arguments returned by parser

    Returns
    -------
    string of arguments

    """

    # translate what to what
    trans_str = "_".join(args.input) + "2{}".format(args.output)

    # training arguments
    train_str = "{args.dataset}_{w2w}_bs{args.batch_size}_ep{args.epochs}_cap{args.model_cap}".format(
        args=args, w2w=trans_str
    )

    if args.seed:
        train_str += "_rs{args.seed}".format(args=args)
    if args.concat:
        train_str += "_concat"

    datestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    idstr = "_".join([train_str, datestr])
    if args.suffix:
        idstr = idstr + "_{}".format(args.suffix)
    return idstr


def args2dict(args):
    """ converts arguments to dict

    Parameters
    ----------
    args: arguments returned by parser

    Returns
    -------
    dict of arguments

    """

    # model_parameters
    model_args = ["model_cap", "n_sampling", "n_scales"]
    if args.concat:
        model_args.append("concat")
    train_args = ["crop", "resize", "batch_size", "epochs", "lbda"]
    if args.seed:
        train_args.append("seed")

    model = {param: getattr(args, param) for param in model_args}
    train = {param: getattr(args, param) for param in train_args}
    data = {
        "name": args.dataset,
        "root": args.dataroot,
        "input": args.input,
        "output": args.output,
    }

    return {"model": model, "training": train, "dataset": data}


def get_generator(config):
    """ returns generator

    Parameters
    ----------
    config : dict
        configuration returned by args2dict

    Returns
    -------
    torch.nn.Model of generator

    """

    dset_class = getattr(datasets, config["dataset"]["name"])
    n_labels = dset_class.N_LABELS
    output_nc = dset_class.N_CHANNELS[config["dataset"]["output"]]

    # only segmentation map as input -> SPADE generator
    if config["dataset"]["input"] == ["seg"]:
        return models.generator.SPADEGenerator(
            n_labels,
            config["model"]["model_cap"] * 2 ** config["model"]["n_sampling"],
            output_nc,
            n_up_stages=config["model"]["n_sampling"],
        )
    # no segmentation map as input -> Pix2Pix generator
    if "seg" not in config["dataset"]["input"]:
        input_nc = sum(dset_class.N_CHANNELS[it] for it in config["dataset"]["input"])
        return models.generator.ResnetEncoderDecoder(
            input_nc,
            config["model"]["model_cap"],
            output_nc,
            n_downsample=config["model"]["n_sampling"],
        )

    # Deal with generator architectures that deal with segmentation maps
    # and continous raster data as input
    # 1) concatenate and use regular Pix2Pix
    # 2) use proposed archicture from the paper

    # number of channels for all input types except the segmentation_map,
    input_nc = sum(
        dset_class.N_CHANNELS[it] for it in config["dataset"]["input"] if it != "seg"
    )

    # Which generator architecture to use with multiple inputs.
    # Conventional generator (pix2pix) with concatenated input
    try:
        if config["model"]["concat"]:
            input_nc += n_labels
            return models.generator.ResnetEncoderDecoder(
                input_nc,
                config["model"]["model_cap"],
                output_nc,
                n_downsample=config["model"]["n_sampling"],
            )
    except KeyError:
        # Proposed conventional generator with SPADE norm layers everywhere
        return models.generator.SPADEResnetEncoderDecoder(
            input_nc,
            n_labels,
            config["model"]["model_cap"],
            output_nc,
            n_downsample=config["model"]["n_sampling"],
        )


def get_discriminator(config):
    """ returns discriminator

    Parameters
    ----------
    config : dict
        configuration returned by args2dict

    Returns
    -------
    torch.nn.Model of discriminator
    """
    dset_class = getattr(datasets, config["dataset"]["name"])
    # generator conditioned on this input
    gen_input_nc = sum(
        dset_class.N_CHANNELS[it] for it in config["dataset"]["input"] if it != "seg"
    )
    if "seg" in config["dataset"]["input"]:
        gen_input_nc += dset_class.N_LABELS

    disc_input_nc = gen_input_nc + dset_class.N_CHANNELS[config["dataset"]["output"]]

    # Downsampling is done in the multiscale discriminator,
    # i.e., all discriminators are identically configures
    d_nets = [
        models.discriminator.PatchGAN(input_nc=disc_input_nc, init_nc=64)
        for _ in range(config["model"]["n_scales"])
    ]

    return models.discriminator.Multiscale(d_nets)
