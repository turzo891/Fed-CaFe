# fed_gan_poc.py
import copy, random, os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, utils as vutils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)

# --- Models: tiny DCGAN for 28x28 ---
NZ = 64   # latent
NGF = 64
NDF = 64
NC = 1    # channels

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF*4, 3, 1, 0, bias=False),   # 1->3
            nn.BatchNorm2d(NGF*4), nn.ReLU(True),
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),# 3->6
            nn.BatchNorm2d(NGF*2), nn.ReLU(True),
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),  # 6->12
            nn.BatchNorm2d(NGF), nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),     # 12->24
            nn.ConvTranspose2d(NC, NC, 5, 1, 0, bias=False),      # 24->28  (fixed)
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),    # 28->14
            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF*2), nn.LeakyReLU(0.2, True), # 14->7
            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF*4), nn.LeakyReLU(0.2, True), # 7->3
            nn.Conv2d(NDF*4, 1, 3, 1, 0, bias=False)  # 3->1  (fixed)
        )
    def forward(self, x): return self.net(x).view(-1)

# --- Small classifier (for bias proxy eval), expects 28x28 ---
class SmallCls(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1)  # 28->14
        self.c3 = nn.Conv2d(32, 64, 3, 2, 1)  # 14->7
        self.fc = nn.Linear(64*7*7, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Data: biased client partitions ---
def make_biased_indices(labels, prefer_even=True, major=0.8, total=30000):
    even_idx = [i for i, y in enumerate(labels) if y % 2 == 0]
    odd_idx  = [i for i, y in enumerate(labels) if y % 2 == 1]
    major_pool, minor_pool = (even_idx, odd_idx) if prefer_even else (odd_idx, even_idx)
    m = int(total * major); n = total - m
    return random.sample(major_pool, m) + random.sample(minor_pool, n)

def get_dataloaders(batch_size=128):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    labels = train.targets.tolist()
    idx_a = make_biased_indices(labels, prefer_even=True,  major=0.8, total=20000)
    idx_b = make_biased_indices(labels, prefer_even=False, major=0.8, total=20000)
    dl_a = DataLoader(Subset(train, idx_a), batch_size=batch_size, shuffle=True, drop_last=True)
    dl_b = DataLoader(Subset(train, idx_b), batch_size=batch_size, shuffle=True, drop_last=True)
    cls_dl = DataLoader(train, batch_size=256, shuffle=True, drop_last=True)  # balanced for classifier
    return dl_a, dl_b, cls_dl

# --- Train classifier for proxy eval ---
def train_classifier(cls_dl, epochs=1):
    net = SmallCls().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        net.train()
        for x, y in cls_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward(); opt.step()
    net.eval(); return net

# --- Local GAN training for one client ---
def local_gan_train(G, D, loader, epochs=1):
    G = copy.deepcopy(G); D = copy.deepcopy(D)
    G.to(DEVICE); D.to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for real, _ in loader:
            real = real.to(DEVICE)
            bsz = real.size(0)

            # Train D
            z = torch.randn(bsz, NZ, 1, 1, device=DEVICE)
            fake = G(z).detach()
            d_real = D(real); d_fake = D(fake)
            lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            optD.zero_grad(); lossD.backward(); optD.step()

            # Train G
            z = torch.randn(bsz, NZ, 1, 1, device=DEVICE)
            fake = G(z)
            d_fake = D(fake)
            lossG = bce(d_fake, torch.ones_like(d_fake))
            optG.zero_grad(); lossG.backward(); optG.step()
    return G.cpu().state_dict(), D.cpu().state_dict()

# --- FedAvg for two state_dicts ---
def fedavg(sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor]):
    return {k: (sd1[k] + sd2[k]) / 2.0 for k in sd1.keys()}

# --- Eval: class distribution & confidence on generated ---
@torch.no_grad()
def eval_bias(G, cls_net, n=2000):
    G = copy.deepcopy(G).to(DEVICE).eval()
    cls_net = copy.deepcopy(cls_net).to(DEVICE).eval()
    z = torch.randn(n, NZ, 1, 1, device=DEVICE)
    fake = G(z)
    logits = cls_net(fake)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1).cpu().tolist()
    even = sum(1 for p in preds if p % 2 == 0)
    odd  = n - even
    conf = probs.max(dim=1).values.mean().item()   # <-- fixed
    return {"n": n, "even": even, "odd": odd, "even_share": even/n, "odd_share": odd/n, "mean_conf": conf}


def save_samples(G, path, nrow=8, n=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    z = torch.randn(n, NZ, 1, 1, device=DEVICE)
    with torch.no_grad():
        imgs = G.to(DEVICE)(z).cpu()
    vutils.save_image(imgs, path, normalize=True, value_range=(-1,1), nrow=nrow)

def main():
    dl_a, dl_b, cls_dl = get_dataloaders()
    cls_net = train_classifier(cls_dl, epochs=1)

    G_global, D_global = Generator(), Discriminator()
    rounds, local_epochs = 5, 1

    for r in range(1, rounds+1):
        Ga_sd, Da_sd = local_gan_train(G_global, D_global, dl_a, epochs=local_epochs)
        Gb_sd, Db_sd = local_gan_train(G_global, D_global, dl_b, epochs=local_epochs)
        G_global.load_state_dict(fedavg(Ga_sd, Gb_sd))
        D_global.load_state_dict(fedavg(Da_sd, Db_sd))

        os.makedirs("./out", exist_ok=True)
        save_samples(G_global, f"./out/gen_round_{r}.png")
        stats = eval_bias(G_global, cls_net, n=1000)
        print(f"[Round {r}] even={stats['even']} odd={stats['odd']} "
              f"even_share={stats['even_share']:.3f} conf={stats['mean_conf']:.3f}")

    torch.save({"G": G_global.state_dict(), "D": D_global.state_dict()}, "./out/fed_gan_ckpt.pt")
    print("Done. Samples saved in ./out/")

if __name__ == "__main__":
    main()
