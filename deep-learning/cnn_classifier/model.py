import torch
import torch.nn as nn

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.backbone = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.classifier = nn.Sequential(
            torch.nn.Linear(16 * 32 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x