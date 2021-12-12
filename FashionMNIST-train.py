import os
import logging
import math
from filelock import FileLock
import random
import sys

# __import_lightning_begin__
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision import transforms
# __import_lightning_end__


from FashionMNISTLightningDataModule import FashionMNISTLightningDataModule
from augmentation.augmentation import TRANSFORM_NAMES
from FashionMNISTLightningModule import FashionMNISTLightningModule
from helpers import utils

log = logging.getLogger('App')
logging.basicConfig(level=logging.INFO)

utils.set_seed(1234)

def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log.info("Starting...")

    # set up the augmentations
    # tuple of augmentation name and its magnitude
    augmentations = []
    for tfn_name in TRANSFORM_NAMES:
        level = random.random()
        augmentations.append((tfn_name, level))

    conf = {
        "progress_bar_refresh_rate": 25,
        "layer_1_size": 512,
        "layer_2_size": 512,
        "lr": 0.001,
        "batch_size": 32,
        "data_dir": "./data",
        # Fashion mnist mean and std
        "data_mean": 0.28604063391685486,
        "data_std": 0.35302430391311646,
        "augmentations": augmentations,
    }

    log.info(f"Conf {conf}")

    data = FashionMNISTLightningDataModule(conf=conf)

    model = FashionMNISTLightningModule(conf=conf)

    trainer = pl.Trainer(
        default_root_dir="./data",
        gpus=-1 if torch.cuda.device_count() > 0 else 0,
        max_epochs=10,
        progress_bar_refresh_rate=conf["progress_bar_refresh_rate"],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
    sys.exit()
