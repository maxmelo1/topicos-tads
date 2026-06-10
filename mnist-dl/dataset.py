import torch
from PIL import Image
import os
import numpy as np


class CatsDogsDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_path, lista_imgs, transforms=None):
        super().__init__()

        self.root_path = root_path
        self.lista_imgs = lista_imgs
        self.transforms = transforms
        
        self.class2idx = {'cat':0, 'dog':1}

    def __getitem__(self, index):
        img_name = self.lista_imgs[index]
        label = self.class2idx[ img_name[0:3] ]

        img = Image.open( os.path.join(self.root_path, img_name) )

        # print(img_name, label)
        
        if self.transforms is not None:
            img = self.transforms(img)

        # print(np.shape(img), label, type(img), img.dtype)

        # input()
        # img2 = img.permute(1,2,0).numpy()
        # img2 = (img2*255).astype(np.uint8)
        # img2 = Image.fromarray(img2)
        # img2.show()

        return img, label
    
    def __len__(self):
        return len(self.lista_imgs)