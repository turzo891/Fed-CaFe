import torch
import torch.nn as nn
import torch.nn.functional as F

class RaceHead(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, classes)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, x):
        h = self.cnn(x).view(x.size(0), -1)
        return self.fc(h)

class ProbeA(nn.Module):
    def __init__(self, in_ch=64, classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, classes)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self, h_mid):
        h = self.net(h_mid).view(h_mid.size(0), -1)
        return self.fc(h)
