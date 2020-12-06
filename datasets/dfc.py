""" The dataset of the IEEE GRSS data fusion contest """

import pathlib
import logging

import rasterio
import numpy as np
import matplotlib
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

logging.getLogger("rasterio").setLevel(logging.WARNING)

classes = [
    "Forest",
    "Shrubland",
    "Savanna",
    "Grassland",
    "Wetlands",
    "Croplands",
    "Urban/Built-up",
    "Snow/Ice",
    "Barren",
    "Water",
]


# check http://www.grss-ieee.org/community/technical-committees/data-fusion/2020-ieee-grss-data-fusion-contest/
lcov_cmap = matplotlib.colors.ListedColormap(
    [
        "#009900",  # Forest
        "#c6b044",  # Shrubland
        "#fbff13",  # Savanna
        "#b6ff05",  # Grassland
        "#27ff87",  # Wetlands
        "#c24f44",  # Croplands
        "#a5a5a5",  # Urban/Built-up
        "#69fff8",  # Snow/Ice
        "#f9ffa4",  # Barren
        "#1c0dff",  # Water
    ]
)
lcov_norm = matplotlib.colors.Normalize(vmin=1, vmax=10)

N_LABELS = 10 + 1  # +1 due to 0 having no label
N_CHANNELS = {"rgb": 3, "sar": 2, "seg": N_LABELS}


class DFC2020(VisionDataset):
    """ IEEE GRSS data fusion contest dataset

    http://www.grss-ieee.org/community/technical-committees/data-fusion/2020-ieee-grss-data-fusion-contest/

    Parameters
    ----------
    root : string
        Root directory of dataset
    split : string, optional
        Image split to use, ``train`` or ``test``
    transforms : callable, optional
        A function/transform that takes input sample and returns a transformed version.
    """

    splits = ["train", "test"]
    datatypes = ["s1", "s2", "dfc"]

    def __init__(self, root, split="train", transforms=None):
        super().__init__(pathlib.Path(root), transforms=transforms)
        verify_str_arg(split, "split", self.splits)
        self.split = split
        self.tif_paths = {dt: self._get_tif_paths(dt) for dt in self.datatypes}

    def _get_tif_paths(self, datatype):
        if self.split == "test":
            pat = "ROIs0000*/{}_*/*9.tif".format(datatype)
        else:
            pat = "ROIs0000*/{}_*/*[0-8].tif".format(datatype)
        return list(sorted(self.root.glob(pat)))

    def __len__(self):
        return len(self.tif_paths["dfc"])

    def __getitem__(self, index):
        def read_tif_as_np_array(path):
            with rasterio.open(path) as src:
                return src.read()

        sample = {
            dt: read_tif_as_np_array(self.tif_paths[dt][index]) for dt in self.datatypes
        }

        # Rename keys and exctract rgb bands from Sentinel-2.
        # Also move channels to last dimension, which is expected by pytorch's to_tensor
        sample["sar"] = (
            self.sar_norm(sample.pop("s1")).transpose((1, 2, 0)).astype(np.float32)
        )
        sample["rgb"] = self.s2_as_rgb(sample.pop("s2"))
        sample["seg"] = sample.pop("dfc").transpose((1, 2, 0))

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def sar_norm(arr):
        """ normalizes SAR to the interval [0, 1] """
        arr = np.clip(arr, -20, 5)
        arr = (arr + 20.0) / 25.0
        return arr

    @staticmethod
    def extract_rgb_from_s2(s2_bands):
        """ extracts RGB bands from Sentinel-2 """
        rgb = np.empty((*s2_bands.shape[1:], 3), dtype=s2_bands.dtype)
        rgb[:, :, 0] = s2_bands[3]
        rgb[:, :, 1] = s2_bands[2]
        rgb[:, :, 2] = s2_bands[1]
        rgb = np.clip(rgb, 0, 3500)
        rgb = rgb / 3500
        return rgb.astype(np.float32)

    @staticmethod
    def sar2rgb(arr):
        """ converts SAR to a plotable RGB image """
        co_pol = arr[:, :, 0]
        cx_pol = arr[:, :, 1]

        rgb = np.empty((*arr.shape[:2], 3), dtype=arr.dtype)
        rgb[:, :, 0] = cx_pol + 0.25
        rgb[:, :, 1] = co_pol
        rgb[:, :, 2] = cx_pol + 0.25
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb.astype(np.float32)

    @staticmethod
    def seg2rgb(arr):
        """ converts segmentation map to a plotable RGB image """
        return lcov_cmap(lcov_norm(np.squeeze(arr)))[:, :, :3]
