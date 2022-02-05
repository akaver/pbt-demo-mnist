import os
import logging
import math
from filelock import FileLock
import random
import multiprocessing
from time import sleep, perf_counter as pc

# __import_lightning_begin__
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from torchvision import transforms
# __import_lightning_end__

# __import_tune_begin__
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
# __import_tune_end__

from FashionMNISTLightningDataModule import FashionMNISTLightningDataModule
from augmentation.augmentation import TRANSFORM_NAMES
from FashionMNISTLightningModule import FashionMNISTLightningModule

log = logging.getLogger('App')
logging.basicConfig(level=logging.INFO)


def train_mnist_tune_checkpoint(conf,
                                checkpoint_dir=None,
                                # no really used, we stop after every epoch and let tune decide what to do
                                num_epochs=999,
                                num_gpus=0):
    # data_dir = os.path.expanduser("~/data")
    data_dir = conf["data_dir"]
    progress_bar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=25);

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=conf["progress_bar_refresh_rate"],
        num_sanity_val_steps=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_acc"
                },
                filename="checkpoint",
                on="validation_end"
            ),
            progress_bar
        ]
    )

    if checkpoint_dir:
        model = FashionMNISTLightningModule.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"), conf=conf)
        log.info('Lightning loaded from checkpoint')
    else:
        model = FashionMNISTLightningModule(conf=conf)
        log.info('Lightning initialized')

    data = FashionMNISTLightningDataModule(conf=conf)

    trainer.fit(model, data)


def tune_mnist_pbt(num_samples=15, training_iteration=15, cpus_per_trial=1, gpus_per_trial=0):
    def explore(config):
        log.info("======================================= EXPLORE =========================================")
        print("======================================= EXPLORE =========================================")
        # calculate new magnitudes for augmentations
        augmentations = []
        for tfn_name in TRANSFORM_NAMES:
            augmentations.append((tfn_name, random.random()))

        config["augmentations"] = augmentations
        log.info(config)
        print(config)
        return config

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        # Models will be considered for perturbation at this interval of time_attr="time_total_s"
        perturbation_interval=1,
        custom_explore_fn=explore,
        log_config=True,
        require_attrs=True,
        quantile_fraction=0.25 # % of top performes used
    )

    progress_reporter = CLIReporter(
        # overwrite=True,
        parameter_columns=["augmentations", "layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    """
    # initial config
            config={
            "progress_bar_refresh_rate": 0,
            "layer_1_size": tune.choice([32, 64, 128, 256, 512, 1024]),
            "layer_2_size": tune.choice([32, 64, 128, 256, 512, 1024]),
            "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
            "batch_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        },
    """

    # set up the augmentations
    # tuple of augmentation name and its magnitude
    def initial_augmentations(spec):
        augmentations = []
        for tfn_name in TRANSFORM_NAMES:
            augmentations.append((tfn_name, random.random()))

        return augmentations


    conf = {
        # https://docs.ray.io/en/master/tune/api_docs/search_space.html?highlight=tune.choice#
        "progress_bar_refresh_rate": 0,
        "layer_1_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        "layer_2_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        "lr": tune.choice([1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        "data_dir": "~/mldata",
        "data_mean": 0.28604063391685486,
        "data_std": 0.35302430391311646,
        "augmentations": tune.sample_from(lambda spec: initial_augmentations(spec))
    }

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune_checkpoint,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=conf,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=progress_reporter,
        verbose=1,
        name="FashionMNIST-pbt",
        stop={  # Stop a single trial if one of the conditions are met
            "mean_accuracy": 0.95,
            "training_iteration": training_iteration},
        local_dir="./data",
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


cpu_count = multiprocessing.cpu_count()
gpu_count = torch.cuda.device_count()

print(f"CPUs {cpu_count} GPUs {gpu_count}")

start_time = pc()

analysis = tune_mnist_pbt(num_samples=16, training_iteration=20, cpus_per_trial=cpu_count / 10,
                          gpus_per_trial=gpu_count / 8)
analysis.best_config
analysis.results

elapsed_time = pc() - start_time

print(f"Time spent on training: {elapsed_time}")
