# Building a Parallel Universe - Image Synthesis from Land Cover Maps and Auxiliary Raster Data

![examples](../assets/nrw_sar_rgb_comp.jpg?raw=true)

This repository contains the code for our paper [Building a Parallel Universe - Image Synthesis from Land Cover Maps and Auxiliary Raster Data](https://arxiv.org/abs/2011.11314)

## Installation

Just clone this repository and also make sure you have a recent version of [PIL](https://github.com/python-pillow/Pillow) with JPEG2000 support.

```bash
git clone https://github.com/gbaier/rs_img_synth.git
```

## Datasets

We employ two datasets of high and medium resolution.
Both are freely available at the IEEE DataPort.

### GeoNRW

The [GeoNRW](https://ieee-dataport.org/open-access/geonrw) dataset consists of 1m aerial photographs, digital elevation models and land cover maps.
It can additionally be augmented with TerraSAR-X spotlight acquisitions, provided you have the corresponding data. 
In that case, please visit https://github.com/gbaier/geonrw to check how to process that data.

### DFC2020

This dataset was used for the IEEE data fusion contest in 2020 and consists of Sentinel-1 and Sentinel-2 patches of 256x256pixels, together with semantic maps.
All of roughly 10m resolution.
You can get the dataset from here https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest, where you need to download **DFC_Public_Dataset.zip**.

## Training new models

1. Training the models as in the paper requires a multiple GPUs due to memory constraints.
Alternatively, the batch size or model capacity can be adjusted using command line parameters.

2. We advise to use multiple workers when training the NRW dataset.
Reading the JPEG2000 files of the GeoNRW's aerial photographs seems to be CPU intensive, and only a single worker bottlenecks the GPU.

3. Both datasets have different types of input and output that the generator can consume or produce.
These can be set using the corresponding command line parameters.
The following table lists all of them.

    Dataset | Input    | Output
    ------- | -------- | --------
    NRW     | dem, seg | rgb, sar
    DFC2020 | seg      | rgb, sar

4. The *crop* and *resize* parameters are ignored for the DFC2020 dataset.

5. The following two examples show how to use the GeoNRW dataset to generate RGB images from digital elevation models and land cover maps
    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --crop 256 \
        --resize 256 \
        --epochs 200 \
        --batch_size 32 \
        --model_cap 64 \
        --lbda 5.0 \
        --num_workers 4 \
        --dataset 'nrw' \
        --dataroot './data/geonrw' \
        --input 'dem' 'seg' \
        --output 'rgb'
    ```
    and dual-pol SAR images from land cover maps alone using the dataset of the 2020 IEEE GRSS data fusion contest
    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 train.py \
    --epochs 200 \
    --batch_size 32 \
    --model_cap 64 \
    --lbda 5.0 \
    --num_workers 0 \
    --dataset 'dfc' \
    --dataroot './data/DFC_Public_Dataset' \
    --input 'seg' \
    --output 'sar
    ```
    The training script creates a directory named something like **nrw_seg_dem2rgb_bs32_ep200_cap64_2020_09_05_07_58**, where the training configuration, logs and the generator and discriminator models will be stored.

## Testing

Run the testing script with a generator model as a command line argument.
```bash
python test.py results/nrw_seg_dem2rgb_bs32_ep200_cap64_2020_09_05_07_58/model_gnet.pt
```
This goes through the testing set and plots the results in the corresponding directory.

## Computing FID scores

1. Train a U-Net segmentation network
    ```bash
    python train_unet.py \
        --crop 256  \
        --resize 256 \
        --epochs 100 \
        --batch_size 32 \
        --num_workers 4 \
        --dataset 'nrw' \
        --dataroot './data/geonrw' \
        --input 'rgb'
    ```
1. Compute FID scores by passing the generator to be tested and the just trained U-Net as arguments
    ```bash
    python fid_comp.py \
            path_to_generator/model_gnet.pt \
            path_to_unet/nrw_unet.pt
    ```
1. Compute intersection-over-union and pixel accuracy
    ```bash
    python pix_acc_iou_comp.py \
            path_to_generator/model_gnet.pt \
            path_to_unet/nrw_unet.pt
            output_dir
    ```
1. You can *optionally* also compute the segmentation results
    ```bash
    python test_unet.py \
           path_to_unet/nrw_unet.pt
    ```
    which stores the segmentation results in the model's directory.


## Codes structure

### GAN training and image synthesis
* `train.py` and `test.py` for training and testing image synthesis.
* `options/` defines command line arguments for training the GANs and U-Net.
* `datasets/` contains data loaders and transforms.
* `models/` contains the various network architectures.
* `loss.py` defines the GAN loss functions.
* `trainer.py` our general GAN trainer.
* `utils.py` utility functions.

### Numerical analysis
* `train_unet.py` and `test_unet.py` for training and testing U-Net..
* `fid_comp.py` calculates Fréchet inception distances using a pretrained U-Net
* `pix_acc_iou_comp.py` calculates pixel accuarcy and IoU using a pretrained U-Net


## Citation

In case you use this code in your research please consider citing
```
@misc{baier2020building,
      title={Building a Parallel Universe - Image Synthesis from Land Cover Maps and Auxiliary Raster Data}, 
      author={Gerald Baier and Antonin Deschemps and Michael Schmitt and Naoto Yokoya},
      year={2020},
      eprint={2011.11314},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
