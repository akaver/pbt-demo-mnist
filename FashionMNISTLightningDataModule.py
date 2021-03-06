import logging
import sys
import os
import multiprocessing
from typing import Optional, Callable, Tuple, Any
from filelock import FileLock
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch

from FashionMNISTDataset import FashionMNISTDataset
from augmentation.augmentation import NAME_TO_TRANSFORM

log = logging.getLogger(__name__)


class FashionMNISTLightningDataModule(pl.LightningDataModule):

    def __init__(self, conf=None):
        super().__init__()
        self.conf = {} if (conf is None) else conf

        self.batch_size = 32 if "batch_size" not in conf else conf["batch_size"]
        self.num_workers = self.get_cpu_count() if "num_workers" not in conf else conf["num_workers"]
        self.data_dir = os.getcwd() if "data_dir" not in conf else conf["data_dir"]
        log.info(f"batch_size: {self.batch_size}. num_workers: {self.num_workers}. data_dir: {self.data_dir}.")
        self.data_mean = None if "data_mean" not in conf else conf["data_mean"]
        self.data_std = None if "data_std" not in conf else conf["data_std"]
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def setup(self, stage: Optional[str] = None):
        mnist_full_config = {
            "data_mean": self.data_mean,
            "data_std": self.data_mean,
            "augmentations": self.conf["augmentations"]
            # TODO: AUGMENTATIONS_HERE
        }

        mnist_full = FashionMNISTDataset(self.data_dir, train=True, download=False, conf=mnist_full_config)

        # split data 0.9/0.1 between training and validation
        train_len = int(len(mnist_full) - len(mnist_full) * 0.1)
        val_len = len(mnist_full) - train_len
        self.mnist_train, self.mnist_val = random_split(mnist_full, [train_len, val_len])

        self.mnist_test = FashionMNISTDataset(self.data_dir, train=False, download=False, conf=None)

    def prepare_data(self):
        # download both Train and Test data, FileLock takes care of Tune concurrent runs
        with FileLock(os.path.expanduser("~/.data.lock")):
            mnist_full = FashionMNISTDataset(self.data_dir, train=True, download=True)
            mnist_test = FashionMNISTDataset(self.data_dir, train=False, download=True)

        if self.data_mean is None or self.data_std is None:
            # Calculate mean and standard deviation needed for data normalization
            log.warning("Calculating dataset mean and std!")
            mean_full = 0
            for img, _ in mnist_full:
                mean_full += torch.mean(img)

            mean_full = mean_full / len(mnist_full)

            std_full = 0
            for img, _ in mnist_full:
                img = torch.square(img - mean_full)
                std_full += torch.mean(img)
            std_full = torch.sqrt(std_full / len(mnist_full))

            self.data_mean = mean_full
            self.data_std = std_full
            log.info(f"dataset mean: {self.data_mean} std: {self.data_std}");

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass

    @staticmethod
    def get_cpu_count():
        # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
        return multiprocessing.cpu_count()

    # alter or apply augmentations to your batch before it is transferred to the device.
    # works well in macbook pro m1 max, cpu/hdd are really fast
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        # augment only in training
        if not self.trainer.training:
            return batch

        # augmentations missing or empty
        if "augmentations" not in self.conf or len(self.conf["augmentations"]) == 0:
            return batch

        # apply augmentations, increasing the amount of samples batch_size * (1 + aug_amount)
        # possibility to run out of memory (main and/or gpu)
        x = batch[0]  # (32,1,28,28)
        y = batch[1]  # (32,)
        out_x = []
        out_y = []
        for img_no in range(x.shape[0]):
            img_original = x[img_no, :, :, :]
            # keep the original
            out_x.append(img_original)
            out_y.append(y[img_no])
            # augmented copies
            for aug_name, level in self.conf["augmentations"]:
                img_transformed = NAME_TO_TRANSFORM[aug_name](img_original, level)
                out_x.append(img_transformed)
                out_y.append(y[img_no])

        batch = [torch.stack(out_x), torch.stack(out_y)]

        return batch

    #  alter or apply augmentations to your batch after it is transferred to the device.
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return batch


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log.info("Starting...")

    data = FashionMNISTDataset(conf={"data_dir": "./data"})


if __name__ == '__main__':
    main()
