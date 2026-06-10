
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import numpy as np

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from tqdm import tqdm


from model import RaphaelModel

from dataset import CatsDogsDataset

import os

from torchvision import transforms
from torchvision.transforms import v2

import random


DATASET_ROOT = '/media/max/data/datasets/dogs-vs-cats/train'


lista_imgs = os.listdir(DATASET_ROOT)
random.shuffle(lista_imgs)

transforms = v2.Compose([
    v2.Resize( (128, 128), ),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.ToTensor(),
    v2.Normalize(mean=(0.,0.,0.), std=(1.,1.,1.)),
])

MAX_EPOCHS = 3

maior_acuracia = 0

tamanho = len(lista_imgs)


dataset_train = CatsDogsDataset(DATASET_ROOT, lista_imgs[0: int(0.8*tamanho)], transforms)
dataset_val = CatsDogsDataset(DATASET_ROOT, lista_imgs[int(0.8*tamanho):], transforms)

dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
dataloader_val   = DataLoader(dataset_val, batch_size=1, shuffle=False)


model = RaphaelModel()#.cuda()
model.train()

funcao_perda = CrossEntropyLoss()#.cuda()
otimizador = SGD(model.parameters(), 0.0005)

lista_acc_train = []
lista_loss_train = []
lista_acc_val = []
lista_loss_val = []

for epoch in range(MAX_EPOCHS):
    # print(len(dataset_train))
    total_loss = []
    total_acc = []
    it = 0
    train_tqdm = tqdm(dataloader_train, desc='Train')
    for img, label in train_tqdm:
        otimizador.zero_grad()

        #img, label = img.cuda(), label.cuda()
        # label = label.unsqueeze(0)

        batch_size = img.size(0)
        
        saida = model(img)

        _, pred = torch.max(saida, dim=1)

        # print(pred)
        # print(label)

        loss = funcao_perda(saida, label)

        loss.backward()
        otimizador.step()

        acc = torch.sum(pred == label) / batch_size

        # print(loss.item(), acc.item())
        train_tqdm.set_postfix(loss=f'{loss.item():.2f}', acc=f'{acc.item():.2f}')

        total_loss.append(loss.item())
        total_acc.append(acc.item())

        it += 1
    
    epoch_loss = np.array(total_loss).mean()
    epoch_acc  = np.array(total_acc).mean()

    print(f'epoch: {epoch}, epoch loss: {epoch_loss}, epoch acc: {epoch_acc}')

    lista_acc_train.append(epoch_acc)
    lista_loss_train.append(epoch_loss)

    total_loss_val = []
    total_acc_val  = []
    with torch.no_grad():
        val_tqdm = tqdm(dataloader_val, desc='Val')
        it = 0
        for img, label in val_tqdm:
            #img, label = img.cuda(), label.cuda()
            # label = label.unsqueeze(0)

            batch_size = img.size(0)

            saida = model(img)

            _, pred = torch.max(saida, dim=1)

            loss = funcao_perda(saida, label)

            acc = torch.sum(pred == label) / batch_size

            # print(loss.item(), acc.item())
            val_tqdm.set_postfix(loss=f'{loss.item():.2f}', acc=f'{acc.item():.2f}')

            total_acc_val.append(acc.item())
            total_loss_val.append(loss.item())
    
    epoch_loss = np.array(total_loss_val).mean()
    epoch_acc  = np.array(total_acc_val).mean()

    lista_acc_val.append(epoch_acc)
    lista_loss_val.append(epoch_loss)

    if epoch_acc > maior_acuracia:
        maior_acuracia = epoch_acc.item()
        torch.save(model.state_dict(), "melhor_modelo.pth" )
        print('Melhor modelo encontrado!!')

    print(f'Época {epoch}/{MAX_EPOCHS}: Val Loss: {epoch_loss}, Val Acc: {epoch_acc}')