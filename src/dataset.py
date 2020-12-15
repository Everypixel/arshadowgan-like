# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class ARDataset(Dataset):
    def __init__(self, dataset_path, augmentation=None, \
                 augmentation_images=None, preprocessing=None, is_train=True, ):
        """Dataset parameters initialization

        Args:
            dataset_path: path to train or test folder
            augmentation: augmentations for images and masks
            augmentation_images: augmentations just for images
            preprocessing: image preprocessing
            is_train: train flag [True - train mode / False - inference mode]
        """
        noshadow_path = os.path.join(dataset_path, 'noshadow')
        mask_path = os.path.join(dataset_path, 'mask')

        # collect paths to images
        self.noshadow_paths = []; self.mask_paths = [];
        self.rshadow_paths = []; self.robject_paths = [];
        self.shadow_paths = [];

        if is_train:
            rshadow_path = os.path.join(dataset_path, 'rshadow')
            robject_path = os.path.join(dataset_path, 'robject')
            shadow_path = os.path.join(dataset_path, 'shadow')

        files_names_list = sorted(os.listdir(noshadow_path))

        for file_name in files_names_list:
            self.noshadow_paths.append(os.path.join(noshadow_path, file_name))
            self.mask_paths.append(os.path.join(mask_path, file_name))

            if is_train:
                self.rshadow_paths.append(os.path.join(rshadow_path, file_name))
                self.robject_paths.append(os.path.join(robject_path, file_name))
                self.shadow_paths.append(os.path.join(shadow_path, file_name))

        self.augmentation = augmentation
        self.augmentation_images = augmentation_images
        self.preprocessing = preprocessing
        self.is_train = is_train

    def __getitem__(self, i):
        """ Getting the i-th set from dataset.

        Args:
            i: index

        Returns:
            image: image with normalization for attention block
            mask: mask with normalization for attention block
            image1: image with normalization for shadow-generation block
            mask1: mask with normalization for shadow-generaion block
        """
        # source image
        image = cv2.imread(self.noshadow_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask for inserted object
        mask = cv2.imread(self.mask_paths[i], 0)

        if self.is_train:
            # neighbours mask
            robject_mask = cv2.imread(self.robject_paths[i], 0)

            # neighbours shadow mask
            rshadow_mask = cv2.imread(self.rshadow_paths[i], 0)

            # result image
            res_image = cv2.imread(self.shadow_paths[i])
            res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

            # apply augmentations for images
            if self.augmentation_images:
                sample = self.augmentation_images(image=image, image1=res_image)
                image = sample['image']
                res_image = sample['image1']

            # collect masks for augmentations
            mask = np.stack([robject_mask, rshadow_mask, mask], axis=-1).astype('float')

            # collect images for augmentations
            image = np.concatenate([image, res_image], axis=2).astype('float')

        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]

        # apply augmentations for images and masks
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # masks normalization
        mask[mask >= 128] = 255; mask[mask < 128] = 0
        # normalization for shadow-generation module
        image1, mask1 = image.astype(np.float) / 127.5 - 1.0, \
        								mask.astype(np.float) / 127.5 - 1.0
        # normalization for attention module
        image, mask = image.astype(np.float) / 255.0, \
        							mask.astype(np.float) / 255.0

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            sample = self.preprocessing(image=image1, mask=mask1)
            image1, mask1 = sample['image'], sample['mask']

        return image, mask, image1, mask1

    def __len__(self):
        """ Returns the dataset size """
        return len(self.noshadow_paths)
