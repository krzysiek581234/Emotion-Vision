import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class AugmentedDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        print(image_paths)
        return image_paths

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        label = int(image_path.split("\\")[1].split("_")[0])  # Extract label from the filename

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
