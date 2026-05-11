import torch

from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root, flist, transform=None):
        self.root = root
        self.transform = transform
        self.flist = flist



    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.flist[idx])
        label = 0 if self.flist[idx].split('.')[0]  == 'cat' else 1

        # print(file_path, label)

        # Load the data from the file
        img = np.array(Image.open(file_path))

        if self.transform:
            img = self.transform(img)

        return img, label