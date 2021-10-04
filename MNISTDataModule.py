import logging
import sys
import os
import multiprocessing

from typing import Optional, Callable, Tuple, Any
from filelock import FileLock
from PIL import Image
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
# https://github.com/zalandoresearch/fashion-mnist
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader
import torch

from augmentation import NAME_TO_TRANSFORM

log = logging.getLogger(__name__)


# https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
class MNISTDataSet(FashionMNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            conf={}
    ) -> None:
        super(MNISTDataSet, self).__init__(root, train, transform, target_transform, download)
        self.conf = conf

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # do not apply outside transformation to image
        """
        if self.transform is not None:
            img = self.transform(img)
        """

        # converts byte values to 0.0-1.0 float tensor
        img = TF.to_tensor(img)

        if "data_mean" in self.conf and "data_std" in self.conf:
            img = TF.normalize(img, self.conf["data_mean"], self.conf["data_std"])

        # pixel value to use when transformation needs to fill empty space (on rotation for example)
        # use corner pixel, should be black
        fill_pixel = img[0, 0, 0].item()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, conf=None):
        super().__init__()
        self.conf = {} if (conf is None) else conf

        self.batch_size = 32 if "batch_size" not in conf else conf["batch_size"]
        self.num_workers = self.get_cpu_count() if "num_workers" not in conf else conf["num_workers"]
        self.data_dir = os.getcwd() if "data_dir" not in conf else conf["data_dir"]
        log.info(f"batch_size: {self.batch_size}. num_workers: {self.num_workers}. data_dir: {self.data_dir}.")
        self.data_mean = None if "data_mean" not in conf else conf["data_mean"]
        self.data_std = None if "data_std" not in conf else conf["data_std"]

    def setup(self, stage: Optional[str] = None):
        mnist_full_config = {
            "data_mean": self.data_mean,
            "data_std": self.data_mean,
            "augmentations": self.conf["augmentations"]
            # TODO: AUGMENTATIONS_HERE
        }

        mnist_full = MNISTDataSet(self.data_dir, train=True, download=False, conf=mnist_full_config)

        # split data 0.9/0.1 between training and validation
        train_len = int(len(mnist_full) - len(mnist_full) * 0.1)
        val_len = len(mnist_full) - train_len
        self.mnist_train, self.mnist_val = random_split(mnist_full, [train_len, val_len])

        self.mnist_test = MNISTDataSet(self.data_dir, train=False, download=False, conf=None)

    def prepare_data(self):
        # download both Train and Test data, FileLock takes care of Tune concurrent runs
        with FileLock(os.path.expanduser("~/.data.lock")):
            mnist_full = MNISTDataSet(self.data_dir, train=True, download=True)
            mnist_test = MNISTDataSet(self.data_dir, train=False, download=True)

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

    @staticmethod
    def get_cpu_count():
        # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
        return multiprocessing.cpu_count()

    # alter or apply augmentations to your batch before it is transferred to the device.
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if "augmentations" not in self.conf:
            return batch # nothing to do

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

    data = MNISTDataModule(conf={"data_dir": "./data"})


if __name__ == '__main__':
    main()
