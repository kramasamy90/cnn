import matplotlib.pyplot as plt
import albumentations
import numpy as np
import cv2
import tifffile as tif

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ISBIEMDataset(Dataset):
    def __init__(self, file_path, train=True, transform=None):
        self.file_path = file_path
        self.transform = transform
        if train:
            self._get_augmented_data()    
        else:
            self._get_data()


    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, index):

        image = self.images[index] / 255.0
        image = image.astype(np.float32)
        label = self.labels[index] / 255

        image = self.transform(image)
        label = torch.from_numpy(label)
        label = label.long()

        return image, label
    
    @staticmethod
    def _rotate_flip(image):
        transformed_images = []
        timage = image
        fimage = cv2.flip(image, 1)
        transformed_images.append(fimage)
        for i in range(3):
            timage = cv2.rotate(timage, cv2.ROTATE_90_CLOCKWISE)
            fimage = cv2.rotate(timage, cv2.ROTATE_90_CLOCKWISE)
            transformed_images.append(timage)
            transformed_images.append(fimage)
        return transformed_images

    
    def _get_data(self):
        self.images = tif.imread(self.file_path + "/test-volume.tif")
        self.labels = tif.imread(self.file_path + "/test-labels.tif")


    def _get_augmented_data(self):
        images = tif.imread(self.file_path + "/train-volume.tif")
        labels = tif.imread(self.file_path + "/train-labels.tif")
    
        # Transform image by rotation and flipping
        transformed_images = []
        transformed_labels = []
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            transformed_images.extend(self._rotate_flip(image))
            transformed_labels.extend(self._rotate_flip(label))

        transformed_images = np.array(transformed_images)
        transformed_label = np.array(transformed_labels)

        self.images = np.concatenate((images, transformed_images), 0)
        self.labels = np.concatenate((labels, transformed_labels), 0)

        deformed_images = []
        deformed_labels = []
        # For each image 5 random elastic deformed image.
        transform = albumentations.Compose([
            albumentations.ElasticTransform(alpha=10, sigma = 10, p = 0.5)
        ])

        for i in range(len(self.images)):
            for j in range(5):
                transformed = transform(image = self.images[i],
                                        mask=self.labels[i])
                deformed_images.append(transformed['image'])
                deformed_labels.append(transformed['mask'])
        deformed_images = np.array(deformed_images)
        deformed_label = np.array(deformed_labels)

        self.images = np.concatenate((self.images, deformed_images), 0)
        self.labels = np.concatenate((self.labels, deformed_labels), 0)

