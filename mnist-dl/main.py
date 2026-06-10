# from torch.utils.data import Dataset

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import numpy as np

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from tqdm import tqdm


from model import JoaoModel


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

print(f'Treinamento iniciando com {len(training_data)} imagens')


img, label = next(iter(training_loader))

# print(img.shape)
# input()

# print("antes: ",img.size())
model = JoaoModel()
logits = model(img)

_, pred = torch.max(logits, dim=1)

# print(logits[0])
# print(pred)
# print(label)

print(len(training_loader))

print('Inciando treinamento')

MAX_EPOCHS = 3

# model = model.cuda()
model.train()

funcao_perda = CrossEntropyLoss()#.cuda()
otimizador = SGD(model.parameters(), 0.005)

maior_acuracia = 0

lista_acc_treino = []
lista_loss_treino = []

lista_acc_val = []
lista_loss_val = []

for epoch in range(MAX_EPOCHS):
    total_loss = []
    total_acc = []
    it = 0
    tqdm_treino = tqdm(training_loader, desc='Loop de treino')
    for  (x, label) in tqdm_treino:
        otimizador.zero_grad()
        # x = x.cuda()
        # label = label.cuda()

        batch_size = x.size(0)

        logits = model(x)

        _, pred = torch.max(logits, dim=1)

        # print(pred)
        # print(label)

        loss = funcao_perda(logits, label)

        loss.backward()
        otimizador.step()

        acc = torch.sum(pred == label) / batch_size

        total_loss.append(loss.item())
        total_acc.append(acc.item())

        tqdm_treino.set_postfix(perda=f'{loss.item():.2}', acc=f'{acc.item():.2}' )

        it += 1

        # if it % 100:
        #     print(f'Época {epoch}/{MAX_EPOCHS}, Iteração {it}/{len(training_loader)}:')
        #     print(f'Perda no treino: {loss.item()}')
        #     print(f'Acurácia de treino: {acc.item()}')

        # input()
        # print()
    
    epoch_loss = np.array(total_loss).mean()
    epoch_acc  = np.array(total_acc).mean()

    lista_acc_treino.append(epoch_acc)
    lista_loss_treino.append(epoch_loss)

    print(f'Época {epoch}/{MAX_EPOCHS}: Train Loss: {epoch_loss}, Train Acc: {epoch_acc}')

    # parte de validação
    model.eval()

    total_loss_val = []
    total_acc_val  = []

    with torch.no_grad():
        for it, (x, label) in enumerate(validation_loader):
            # x = x.cuda()
            # label = label.cuda()

            batch_size = x.size(0)

            logits = model(x)

            _, pred = torch.max(logits, dim=1)

            loss = funcao_perda(logits, label)

            acc = torch.sum(pred == label) / batch_size

            total_loss_val.append(loss.item())
            total_acc_val.append(acc.item())
    
    epoch_loss = np.array(total_loss_val).mean()
    epoch_acc  = np.array(total_acc_val).mean()

    lista_acc_val.append(epoch_acc)
    lista_loss_val.append(epoch_loss)

    if epoch_acc > maior_acuracia:
        maior_acuracia = epoch_acc.item()
        torch.save(model.state_dict(), "melhor_modelo.pth" )
        print('Melhor modelo encontrado!!')

    print(f'Época {epoch}/{MAX_EPOCHS}: Val Loss: {epoch_loss}, Val Acc: {epoch_acc}')


eixo_x = list(range(0, MAX_EPOCHS))

plt.plot(eixo_x, lista_loss_treino)
plt.title('Loss de treino')
plt.savefig('loss_treino.png', dpi=300)
plt.show()

plt.plot(eixo_x, lista_acc_treino)
plt.title('Acurácia de treino')
plt.savefig('acc_treino.png', dpi=300)
plt.show()

plt.plot(eixo_x, lista_loss_val)
plt.title('Loss de validação')
plt.savefig('loss_val.png', dpi=300)
plt.show()

plt.plot(eixo_x, lista_acc_val)
plt.title('Acurácia de treino')
plt.savefig('acc_val.png', dpi=300)
plt.show()