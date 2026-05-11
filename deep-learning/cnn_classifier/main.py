import torch 

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CustomDataset
from model import CustomModel

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

import os
import random
import numpy as np

N_EPOCHS = 10
PRINT_INTERVAL = 50

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5,), std=(1.0, 1.0, 1.0,)),
        transforms.Resize((64, 64))
    ])


    file_list = os.listdir('/media/max/data/dogs-vs-cats/train')
    random.shuffle(file_list)
    
    files_train = file_list[:int(len(file_list)*0.8)]
    files_val = file_list[int(len(file_list)*0.8):]

    if not os.path.exists('val_files.txt'):
        with open('val_files.txt', 'w') as f:
            for file in files_val:
                f.write(file + '\n')


    dataset_train = CustomDataset(root='/media/max/data/dogs-vs-cats/train', flist=files_train, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_val = CustomDataset(root='/media/max/data/dogs-vs-cats/train', flist=files_val, transform=transform)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)


    model = CustomModel().cuda()

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)



    best_loss = float('inf')

    for epoch in range(N_EPOCHS):
        total_loss = 0.0

        model.train()

        for i, (images, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)

            
            loss.backward()
            
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

            if (i + 1) % PRINT_INTERVAL == 0:
                print(f"Iter: {i+1}/{len(dataloader_train)}, Loss: {loss_val:.4f}")

        total_loss /= len(dataloader_train)

        total_loss_val = 0.0
        total_acc_val = 0.0

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader_val):
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss_val = loss.item()
                total_loss_val += loss_val

                acc_val = (outputs.argmax(dim=1) == labels).float().mean().item()
                total_acc_val += acc_val

                if (i + 1) % PRINT_INTERVAL == 0:
                    print(f"Validation Loss: {loss_val:.4f}, Validation Accuracy: {acc_val:.4f}")


        total_loss_val /= len(dataloader_val)
        total_acc_val /= len(dataloader_val)

        if total_loss_val < best_loss:
            best_loss = total_loss_val
            torch.save(model.state_dict(), 'best_model.pth')
            print('New best model saved!')

        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Total Loss: {total_loss:.4f}, Validation Loss: {total_loss_val:.4f}, Validation Accuracy: {total_acc_val:.4f}")