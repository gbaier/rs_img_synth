""" The GeoNRW dataset """

import pathlib
import itertools
from PIL import Image

import matplotlib
import matplotlib.cm
import numpy as np

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

classes = [
    "Forest",
    "Water",
    "Agricultural",
    "Urban",
    "Grassland",
    "Railway",
    "Highway",
    "Airport, shipyard",
    "Roads",
    "Buildings",
]

lcov_cmap = matplotlib.colors.ListedColormap(
    [
        "#2ca02c",  # matplotlib green for forest
        "#1f77b4",  # matplotlib blue for water
        "#8c564b",  # matplotlib brown for agricultural
        "#7f7f7f",  # matplotlib gray residential_commercial_industrial
        "#bcbd22",  # matplotlib olive for grassland_swamp_shrubbery
        "#ff7f0e",  # matplotlib orange for railway_trainstation
        "#9467bd",  # matplotlib purple for highway_squares
        "#17becf",  # matplotlib cyan for airport_shipyard
        "#d62728",  # matplotlib red for roads
        "#e377c2",  # matplotlib pink for buildings
    ]
)
lcov_norm = matplotlib.colors.Normalize(vmin=1, vmax=10)

# number of classes + invalid
N_LABELS = 11

N_CHANNELS = {"rgb": 3, "sar": 1, "dem": 1, "seg": N_LABELS}


class NRW(VisionDataset):
    """ Optical, SAR, LiDAR and landcover data from North Rhine-Westphalia.

    There are fewer SAR images then for the other types of data.
    If you don't need SAR, set include_sar to ``False`` for a bigger dataset.

    Parameters
    ----------
    root : string
        Root directory of dataset
    split : string, optional
        Image split to use, ``train`` or ``test``
    include_sar : boolean, optional
        Include SAR imagery when returning samples
    transforms : callable, optional
        A function/transform that takes input sample and returns a transformed version.

    """

    splits = ["train", "test"]

    train_list = [
        "aachen",
        "bergisch",
        "bielefeld",
        "bochum",
        "bonn",
        "borken",
        "bottrop",
        "coesfeld",
        "dortmund",
        "dueren",
        "duisburg",
        "ennepetal",
        "erftstadt",
        "essen",
        "euskirchen",
        "gelsenkirchen",
        "guetersloh",
        "hagen",
        "hamm",
        "heinsberg",
        "herford",
        "hoexter",
        "kleve",
        "koeln",
        "krefeld",
        "leverkusen",
        "lippetal",
        "lippstadt",
        "lotte",
        "moenchengladbach",
        "moers",
        "muelheim",
        "muenster",
        "oberhausen",
        "paderborn",
        "recklinghausen",
        "remscheid",
        "siegen",
        "solingen",
        "wuppertal",
    ]

    test_list = ["duesseldorf", "herne", "neuss"]

    # Convert segmentation map to different PIL mode.
    # Otherwise PyTorch later normalizes
    readers = {
        "sar": lambda path: Image.open(path).copy(),
        "rgb": lambda path: Image.open(path).convert("RGB"),
        "dem": lambda path: Image.open(path).copy(),
        "seg": lambda path: Image.open(path).convert("I;16"),
    }

    filenames = {
        "sar": lambda utm_coords: "{}_{}_sar.tif".format(*utm_coords),
        "rgb": lambda utm_coords: "{}_{}_rgb.jp2".format(*utm_coords),
        "dem": lambda utm_coords: "{}_{}_dem.tif".format(*utm_coords),
        "seg": lambda utm_coords: "{}_{}_seg.tif".format(*utm_coords),
    }

    def __init__(self, root, split="train", include_sar=False, transforms=None):
        super().__init__(pathlib.Path(root), transforms=transforms)
        verify_str_arg(split, "split", self.splits)
        if split == "test":
            self.city_names = self.test_list
        elif split == "train":
            self.city_names = self.train_list
        self.datatypes = ["rgb", "dem", "seg"]
        if include_sar:
            self.file_list = self._get_file_list("*sar.tif")
            self.datatypes.append("sar")
        else:
            self.file_list = self._get_file_list("*rgb.jp2")

    def _get_file_list(self, pattern):
        # iterate over citynames
        return list(
            sorted(
                itertools.chain.from_iterable(
                    (self.root / cn).glob(pattern) for cn in self.city_names
                )
            )
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]
        utm_coords = path.stem.split("_")[:2]

        sample = {}
        for datatype in self.datatypes:
            path = path.parents[0] / self.filenames[datatype](utm_coords)
            sample[datatype] = self.readers[datatype](path)

        try:
            sample["sar"] = Image.fromarray(self.sar_norm(sample["sar"]))
        except KeyError:
            pass

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def sar_norm(arr):
        """ normalizes SAR to the interval [0, 1] """
        arr = 20.0 * np.log10(arr)
        return np.clip(arr / 100.0, 0, 1)

    @staticmethod
    def seg2rgb(segm):
        """ converts segmentation map to a plotable RGB image """
        return lcov_cmap(lcov_norm(segm))[:, :, :3]

    @staticmethod
    def depth2rgb(depth):
        """ converts DEM to a plotable RGB image """
        depth -= depth.min()
        depth /= depth.max()
        return matplotlib.cm.viridis(depth)[:, :, :3]

    @staticmethod
    def sar2rgb(sar):
        """ converts SAR to a plotable RGB image """
        sar = np.squeeze(np.clip(255 * sar, 0, 255).astype(np.uint8))

        return matplotlib.cm.gray(sar)[:, :, :3]
