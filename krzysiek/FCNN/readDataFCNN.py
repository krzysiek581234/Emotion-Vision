import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchvision.transforms as transforms
class readDataFCNN(Dataset):
    def __init__(self, rel):
        xy = pd.read_csv(rel)
        self.n_samples = len(xy)
        self.label = torch.from_numpy(np.array(xy.emotion.astype('float32')))
        self.features = torch.from_numpy(np.array(
            [np.array(xy.pixels[i].split(' ')).astype('float32') for i in range(self.n_samples)]))

        # self.features = torch.from_numpy(np.array(xy.pixels[i].split(' ') for i in range(self.n_samples)))
        # # normalizacja danych
        # self.transform = transforms.Compose([
        #     transforms.Normalize((0.5,))  # Normalize the image tensor
        #     # obicia lutrzane
        #     # lekkine przebarwienia
        #     # szum gausowski
        #
        # ])
    def __getitem__(self, index):
        # aplikowanie normalizacji do features i zwrócenie par label, normalized_feature
        return self.label[index], self.features[index]
        #self.transform(self.features[index])

    def __len__(self):
        return self.n_samples