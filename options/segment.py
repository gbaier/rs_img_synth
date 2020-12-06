""" command line arguments and helper functions for segmentation network """

import argparse
import datetime

from . import common


def get_parser():
    """ returns ArgumentParser for segmentation networks """

    parser = common.get_parser()

    parser.add_argument(
        "--input",
        default="rgb",
        choices=["rgb", "sar"],
        help="Input of the segmentation network.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="weight decay"
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

    # training arguments
    train_str = "{args.dataset}_unet_{args.input}_bs{args.batch_size}_ep{args.epochs}_lr{args.learning_rate}".format(
        args=args
    )
    if args.seed:
        train_str += "_rs{args.seed}".format(args=args)

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
    train_args = ["crop", "resize", "batch_size", "epochs", "learning_rate"]
    if args.seed:
        train_args.append("seed")

    train = {param: getattr(args, param) for param in train_args}
    data = {
        "name": args.dataset,
        "root": args.dataroot,
        "input": args.input,
    }

    return {"training": train, "dataset": data}
