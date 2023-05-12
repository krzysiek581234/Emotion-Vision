import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

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