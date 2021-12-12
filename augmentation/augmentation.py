import logging
import PIL
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF


log = logging.getLogger(__name__)


# define augmentation functions

def auto_contrast(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    fill = img[0, 0, 0].item() if fill is None else fill
    if level > 0.1:
        img = TF.autocontrast(img)
    return img


def blur(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    kernel_size = int(level * 4)
    if kernel_size % 2 == 0:
        if random.random() > 0.5:
            kernel_size = kernel_size + 1
        else:
            kernel_size = kernel_size - 1
    if kernel_size > 0:
        img = TF.gaussian_blur(img, kernel_size = kernel_size)
    return img


def crop(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    height, width = img.shape[1], img.shape[2]
    crop_h = int(height * level)
    crop_w = int(width * level)
    # crop from center
    # img = TF.resized_crop(img,[crop_h, crop_w])

    return img


def cutout(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def equalize(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def flip_leftright(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def flip_updown(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def identity(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def posterize(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def rotate_left(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    fill = img[0, 0, 0].item() if fill is None else fill
    # max 30 degrees of rotation
    degrees = level * 30
    img = TF.rotate(img, degrees, fill=fill)
    return img


def rotate_right(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    fill = img[0, 0, 0].item() if fill is None else fill
    # max 30 degrees of rotation
    degrees = level * -30
    img = TF.rotate(img, degrees, fill=fill)
    return img


def shear_x(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def shear_y(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def smooth(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def solarize(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def translate_x(img: torch.Tensor, level: float, fill=None) -> torch.Tensor:
    pass


def translate_y(img):
    pass


ALL_TRANSFORMS = [
    auto_contrast,
    blur,
    crop,
    cutout,
    equalize,
    flip_leftright,
    flip_updown,
    identity,
    posterize,
    rotate_left,
    rotate_right,
    shear_x,
    shear_y,
    smooth,
    solarize,
    translate_x,
    translate_y,
]

# actual working augmentations. just add more here!
ALL_TRANSFORMS = [
    # auto_contrast,
    blur,
    rotate_left,
    rotate_right,
]

NAME_TO_TRANSFORM = {t.__name__: t for t in ALL_TRANSFORMS}

TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()

