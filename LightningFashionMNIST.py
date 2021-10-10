import logging
import os
import random
import sys
from filelock import FileLock

# __import_lightning_begin__
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision import transforms
# __import_lightning_end__

import torchvision.transforms.functional as TF

from MNISTDataModule import MNISTDataModule
from augmentation import TRANSFORM_NAMES

log = logging.getLogger(__name__)


# __lightning_begin__
class LightningFashionMNIST(pl.LightningModule):
    def __init__(self, conf=None):
        super().__init__()
        self.conf = conf

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.conf["layer_1_size"])
        self.layer_2 = torch.nn.Linear(self.conf["layer_1_size"], self.conf["layer_2_size"])
        # there are 10 classes
        self.layer_3 = torch.nn.Linear(self.conf["layer_2_size"], 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    @staticmethod
    def cross_entropy_loss(logits, labels):
        return F.nll_loss(logits, labels)

    @staticmethod
    def accuracy(logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def model_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        
        return loss, accuracy
        
    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self.model_step(train_batch, batch_idx)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self.model_step(val_batch, batch_idx)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val_acc', avg_acc, on_epoch=True, prog_bar=True, logger=True)

        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
        
    def test_step(self, batch, batch_idx):
        loss, accuracy = self.model_step(train_batch, batch_idx)
        return {"test_loss": loss, "test_acc": acc accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf["lr"])
        return optimizer


# __lightning_end__


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

    data = MNISTDataModule(conf=conf)

    model = LightningFashionMNIST(conf=conf)

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
