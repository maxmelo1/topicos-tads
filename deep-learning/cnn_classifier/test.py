import torch 

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CustomDataset
from model import CustomModel

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt

import random
import numpy as np


import os


CLASSES = ['cat', 'dog']


def unnormalize(tensor, mean=(0.5, 0.5, 0.5,), std=(1.0, 1.0, 1.0,)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5,), std=(1.0, 1.0, 1.0,)),
        transforms.Resize((64, 64))
    ])

    files_val = []
    with open('val_files.txt', 'r') as f:
        for line in f:
            files_val.append(line.strip())


    dataset_val = CustomDataset(root='/media/max/data/dogs-vs-cats/train', flist=files_val, transform=transform)

    model = CustomModel().cuda()
    ### AQUI O MODELO É CARREGADO PÓS TREINAMENTO, PARA AVALIAÇÃO
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()



    img, label = dataset_val[random.randint(0, len(dataset_val)-1)]

    logits = model(img.unsqueeze(0).cuda())
    pred = torch.argmax(logits, dim=1).item()

    img_vis = unnormalize(img.clone(), mean=(0.5, 0.5, 0.5,), std=(1.0, 1.0, 1.0,)).clamp(0, 1)*255
    img_vis = img_vis.to(torch.uint8).permute(1, 2, 0)
    # print(np.shape(img_vis))
    plt.imshow(img_vis)
    plt.title(f"Label: {CLASSES[label]}, predicted: {CLASSES[pred]}")
    plt.show()


