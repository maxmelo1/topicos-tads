from cProfile import label

import torch

from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.split = split

        self.data = []
        self.labels = []

        for file in os.listdir(os.path.join(root, split)):
            file_path = os.path.join(root, split, file)
            self.data.append(file_path)
            self.labels.append(0 if file.split('.')[0]=='cat' else 1)  # Assuming label is the prefix of the filename   


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]

        # Load the data from the file
        img = np.array(Image.open(file_path))

        if self.transform:
            img = self.transform(img)

        return img, label