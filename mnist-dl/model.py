import torch
import torch.nn as nn

class JoaoModel(nn.Module):
    def __init__(self ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (3,3), padding='same'),
            nn.Conv2d(8, 64, (3,3), padding='same'),
        )

        self.classificador = nn.Sequential(
            nn.Linear(64*28*28, 10)
        )

    
    def forward(self, x):
        x = self.backbone(x)
        # print(x.size())

        x = x.view(x.size(0), -1)
        # print(x.size())

        x = self.classificador(x)
        # print(x.size())
        return x
    
class RaphaelModel(nn.Module):
    def __init__(self ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, (3,3), padding='same'),
            nn.Conv2d(8, 64, (3,3), padding='same'),
        )

        self.classificador = nn.Sequential(
            nn.Linear(64*128*128, 2)
        )

    
    def forward(self, x):
        x = self.backbone(x)
        # print(x.size())

        x = x.view(x.size(0), -1)
        # print(x.size())

        x = self.classificador(x)
        # print(x.size())
        return x