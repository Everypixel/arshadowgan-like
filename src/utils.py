# -*- coding: utf-8 -*-

import os
import random
import numpy as np

import torch
import albumentations as albu


def seed_everything(seed=42):
    """Set seed for all random functions

    Args:
        seed (int): seed for random functions
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_training_augmentation(image_size):
    """Augmentations for all train images (images and masks!)

    Args:
        image_size (int): width and height of image for Resize function
    """
    train_transform = [
        albu.Resize(image_size,image_size),
        albu.HorizontalFlip(p=0.5),
        albu.Rotate(p=0.3, limit=(-10, 10), interpolation=3, border_mode=2),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size):
    """Augmentations for all validation / test images (images and masks!)

    Args:
        image_size (int): width and height of image for Resize function
    """
    test_transform = [
        albu.Resize(image_size,image_size),
    ]
    return albu.Compose(test_transform)


def get_image_augmentation():
    """ Augmentations just for input and output images (not for masks) """
    image_transform = [
        albu.OneOf([
          albu.Blur(p=0.2, blur_limit=(3, 5)),
          albu.GaussNoise(p=0.2, var_limit=(10.0, 50.0)),
          albu.ISONoise(p=0.2, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
          albu.ImageCompression(p=0.2, quality_lower=90, quality_upper=100, compression_type=0),
          albu.MultiplicativeNoise(p=0.2, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
        ], p=1),
        albu.OneOf([
          albu.HueSaturationValue(p=0.2, hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10)),
          albu.RandomBrightness(p=0.3, limit=(-0.1, 0.1)),
          albu.RandomGamma(p=0.3, gamma_limit=(80, 100), eps=1e-07),
          albu.ToGray(p=0.1),
          albu.ToSepia(p=0.1),
        ], p=1)
    ]
    return albu.Compose(image_transform, additional_targets={
        'image1': 'image',
        'image2': 'image'
    })


def get_preprocessing():
    """ Preprocessing function """
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def to_tensor(x, **kwargs):
    """ Modify image to format: [channels, width, height]

    Args:
        x (np.array): an input image
    """
    return x.transpose(2, 0, 1).astype('float32')
