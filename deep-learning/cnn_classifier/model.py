import torch
import torch.nn as nn

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.backbone = nn.Sequential(
            DoubleBlock(3, 16),
            DoubleBlock(16, 32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleBlock(32, 64),
            DoubleBlock(64, 128),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleBlock(128, 128),
            DoubleBlock(128, 128),
            nn.Dropout(0.2),            
        )
        
        self.classifier = nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        # print(x.shape)
        # input()
        x = self.classifier(x)
        return x
    
class DoubleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleBlock, self).__init__()
        self.block = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x