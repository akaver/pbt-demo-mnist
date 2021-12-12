import logging
import os
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

log = logging.getLogger(__name__)


# __lightning_begin__
class FashionMNISTLightningModule(pl.LightningModule):
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
    """
    def __init__(self, conf=None):
        super().__init__()
        self.conf = conf

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc1 = torch.nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = torch.nn.Dropout2d(0.25)
        self.fc2 = torch.nn.Linear(in_features=600, out_features=120)
        self.fc3 = torch.nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.log_softmax(out, dim=1)
        return out
    """



    @staticmethod
    def cross_entropy_loss(logits, labels):
        # negative log likelihood loss
        # https://pytorch.org/docs/1.9.0/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
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
        self.log("ptl/train_acc", accuracy)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self.model_step(val_batch, batch_idx)
        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val_acc', avg_acc, on_epoch=True, prog_bar=True, logger=True)

        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_acc", avg_acc)
        
    def test_step(self, batch, batch_idx):
        loss, accuracy = self.model_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"test_loss": loss, "test_acc": accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        self.log('avg_test_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_test_acc', avg_acc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf["lr"])
        return optimizer


# __lightning_end__

