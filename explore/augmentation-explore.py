import logging
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from FashionMNISTLightningDataModule import MNISTDataSet
from augmentation.augmentation import TRANSFORM_NAMES, NAME_TO_TRANSFORM

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log.info("Starting aug testing...")

    log.info(TRANSFORM_NAMES)

    mnist_full_config = {
        "data_mean": 0.28604063391685486,
        "data_std":  0.35302430391311646,
    }

    mnist_full = MNISTDataSet("./data", train=True, download=False, conf=mnist_full_config)

    img, label = mnist_full[0]

    log.info("Original image")
    image_show = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(image_show, cmap='gray')
    plt.show()

    # how strong should the transformation be in range of 0.0-1.0
    level = random.random()
    for f_name in TRANSFORM_NAMES:
        log.info(f"TF: {f_name} {level}")
        # apply the transform function
        img_transformed = NAME_TO_TRANSFORM[f_name](img, level)
        image_show = np.transpose(img_transformed.numpy(), (1, 2, 0))
        plt.imshow(image_show, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
    sys.exit()
