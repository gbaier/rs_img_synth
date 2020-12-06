import logging

import torch
import torch.nn
import torch.distributed

import numpy as np
import yaml

import datasets.nrw
import datasets.dfc
import options.common
import options.gan
from trainer import Trainer


##################################
#                                #
# Parsing command line arguments #
#                                #
##################################

parser = options.gan.get_parser()
args = parser.parse_args()

OUT_DIR = args.out_dir / options.gan.args2str(args)
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
if args.local_rank == 0:
    logger.info("Saving logs, configs and models to %s", OUT_DIR)


###################################
#                                 #
# Checking command line arguments #
#                                 #
###################################

# Reproducibilty config https://pytorch.org/docs/stable/notes/randomness.html
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning(
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

CONFIG = options.gan.args2dict(args)

with open(OUT_DIR / "config.yml", "w") as cfg_file:
    yaml.dump(CONFIG, cfg_file)


if not torch.cuda.is_available():
    raise RuntimeError("This scripts expects CUDA to be available")

device = torch.device("cuda:{}".format(args.local_rank))

# set device of this process. Otherwise apex.amp throws errors.
# see https://github.com/NVIDIA/apex/issues/319
torch.cuda.set_device(device)
torch.distributed.init_process_group(
    "nccl",
    init_method="env://",
    world_size=torch.cuda.device_count(),
    rank=args.local_rank,
)


#########################
#                       #
# Dataset configuration #
#                       #
#########################

train_transforms, test_transforms = options.common.get_transforms(CONFIG)

dataset = options.common.get_dataset(CONFIG, split="train", transforms=train_transforms)

if args.local_rank == 0:
    logger.info(dataset)


################################
#                              #
# Neural network configuration #
#                              #
################################

g_net = options.gan.get_generator(CONFIG).to(device)
d_net = options.gan.get_discriminator(CONFIG).to(device)

#####################
#                   #
# Distributed setup #
#                   #
#####################

# separate processing groups for generator and discriminator
# https://discuss.pytorch.org/t/calling-distributeddataparallel-on-multiple-modules/38055
g_pg = torch.distributed.new_group(range(torch.distributed.get_world_size()))
g_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(g_net, process_group=g_pg)
g_net = torch.nn.parallel.DistributedDataParallel(
    g_net.cuda(args.local_rank),
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    process_group=g_pg,
)

d_pg = torch.distributed.new_group(range(torch.distributed.get_world_size()))
# no batch norms in discriminator that need to be synced
d_net = torch.nn.parallel.DistributedDataParallel(
    d_net.cuda(args.local_rank),
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    process_group=d_pg,
)

############
#          #
# Training #
#          #
############

trainer = Trainer(
    g_net,
    d_net,
    args.input,
    args.output,
    feat_loss=CONFIG["training"]["lbda"],
)

train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, shuffle=True, num_replicas=torch.cuda.device_count(), rank=args.local_rank,
)
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size // torch.cuda.device_count(),
    sampler=train_sampler,
    num_workers=args.num_workers,
)

trainer.train(train_dataloader, args.epochs)


##########
#        #
# Saving #
#        #
##########

if args.local_rank == 0:
    torch.save(trainer.g_net.state_dict(), OUT_DIR / "model_gnet.pt")
    torch.save(trainer.d_net.state_dict(), OUT_DIR / "model_dnet.pt")
