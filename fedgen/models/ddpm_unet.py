import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet64(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.e2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.e3 = nn.Conv2d(64, 64, 3, 2, 1)
        self.b  = nn.Conv2d(64, 64, 3, 1, 1)
        self.d1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.d2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
        for m in [self.e1,self.e2,self.e3,self.b,self.d1,self.d2,self.out]:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    def forward(self, x):
        x1 = F.relu(self.e1(x))
        x2 = F.relu(self.e2(x1))
        x3 = F.relu(self.e3(x2))
        b  = F.relu(self.b(x3))
        y  = F.relu(self.d1(b))
        y  = F.relu(self.d2(y))
        y  = self.out(y)
        return y
