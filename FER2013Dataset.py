import cv2
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from AugmentedDataset import AugmentedDataset


class FER2013Dataset(Dataset):
    def __init__(self, aug_transform=None):
        self.aug_transform = aug_transform

    def load_train_dataset(self):
        augmented_train_dataset = AugmentedDataset('augmented_images', self.aug_transform)

        # Create a DataLoader to load the data in batches
        batch_size = 128
        train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader

    def load_test_dataset(self):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load the augmented images dataset
        test_dataset = ImageFolder('./test_images', transform=transform)

        # Create data loader
        batch_size = 128
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return test_loader

    def load_dataset(self, path):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        with open(path, 'r') as file:
            lines = file.readlines()

            for i, line in enumerate(lines):
                if i == 0:
                    continue

                emotion, usage, pixels = line.split(",")
                pixels = np.array(pixels.split(), dtype='uint8')
                image = pixels.reshape((48, 48))

                if usage == 'Training':
                    X_train.append(image)
                    y_train.append(int(emotion))
                else:
                    X_test.append(image)
                    y_test.append(int(emotion))

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        pixels = sample['pixels']
        emotion = sample['emotion']

        # Convert pixel values to a 48x48 grayscale image tensor
        image = torch.Tensor([int(pixel) for pixel in pixels.split()]).reshape(48, 48)
        image = image.unsqueeze(0)  # Add a channel dimension

        # Perform any additional transformations on the image or emotion label here if needed
        if self.transform:
            image = self.transform(image)

        return image, emotion
