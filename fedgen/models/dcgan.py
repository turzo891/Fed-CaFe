import torch
import torch.nn as nn
import torch.nn.functional as F

class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Generator64(nn.Module):
    def __init__(self, z_dim=128, groups=4, emb_dim=16):
        super().__init__()
        self.groups = groups
        self.emb = nn.Embedding(groups, emb_dim)
        in_dim = z_dim + emb_dim
        self.fc = nn.Linear(in_dim, 512*4*4)
        self.g1 = GenBlock(512, 256)
        self.g2 = GenBlock(256, 128)
        self.g3 = GenBlock(128, 64)
        self.out = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        nn.init.xavier_uniform_(self.out.weight)
    def forward(self, z, y):
        yemb = self.emb(y)
        h = torch.cat([z, yemb], dim=1)
        x = self.fc(h).view(-1, 512, 4, 4)
        x = self.g1(x)
        x = self.g2(x)
        h_mid = self.g3(x)
        img = torch.tanh(self.out(h_mid)) * 0.5 + 0.5
        return img, h_mid

class Discriminator64(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ic, oc, bn=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn: layers.insert(1, nn.BatchNorm2d(oc))
            return nn.Sequential(*layers)
        self.d = nn.Sequential(
            block(3, 64, bn=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
        )
        self.out = nn.Conv2d(512, 1, 4, 1, 0)
        nn.init.xavier_uniform_(self.out.weight)
    def forward(self, x):
        h = self.d(x)
        logits = self.out(h).view(-1, 1)
        return torch.sigmoid(logits)
