# fed_gan_poc_5clients.py
import argparse, copy, csv, os, random, sys
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms, utils as vutils

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", type=str, default="./out")
    p.add_argument("--device", type=str, default="auto", help='"auto", "cpu", or "cuda:0"')
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ---------- Setup ----------
torch.manual_seed(0); random.seed(0)

def get_device(opt):
    if opt.device == "cpu": return "cpu"
    if opt.device.startswith("cuda"): return opt.device
    return "cuda" if torch.cuda.is_available() else "cpu"

NZ, NGF, NDF, NC = 64, 64, 64, 1  # 28x28 GAN

# ---------- Models ----------
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
            nn.ConvTranspose2d(NC, NC, 5, 1, 0, bias=False),      # 24->28
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                         # 28->14
            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF*2), nn.LeakyReLU(0.2, True), # 14->7
            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF*4), nn.LeakyReLU(0.2, True), # 7->3
            nn.Conv2d(NDF*4, 1, 3, 1, 0, bias=False)                                                   # 3->1
        )
    def forward(self, x): return self.net(x).view(-1)

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

# ---------- Data ----------
def make_biased_indices(labels, prefer_even=True, major=0.9, total=30000):
    even_idx = [i for i, y in enumerate(labels) if y % 2 == 0]
    odd_idx  = [i for i, y in enumerate(labels) if y % 2 == 1]
    major_pool, minor_pool = (even_idx, odd_idx) if prefer_even else (odd_idx, even_idx)
    m = int(total * major); n = total - m
    return random.sample(major_pool, m) + random.sample(minor_pool, n)

def get_dataloaders_5(batch_size=128):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    labels = train.targets.tolist()
    specs = [
        {"prefer_even": True,  "major": 0.90, "total": 20000},  # A even-heavy
        {"prefer_even": False, "major": 0.90, "total": 20000},  # B odd-heavy
        {"prefer_even": False, "major": 0.80, "total": 15000},  # C odd-heavy
        {"prefer_even": False, "major": 0.70, "total": 10000},  # D odd-heavy
        {"prefer_even": True,  "major": 0.60, "total":  8000},  # E even-leaning
    ]
    dls, sizes = [], []
    for s in specs:
        idx = make_biased_indices(labels, s["prefer_even"], s["major"], s["total"])
        dls.append(DataLoader(Subset(train, idx), batch_size=batch_size, shuffle=True, drop_last=True))
        sizes.append(len(idx))
    cls_dl = DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
    return dls, sizes, cls_dl

# ---------- Train helpers ----------
def train_classifier(cls_dl, device, epochs=1):
    net = SmallCls().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        net.train()
        for x, y in cls_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward(); opt.step()
    net.eval(); return net

def local_gan_train(G, D, loader, device, epochs=1, nz=NZ):
    G = copy.deepcopy(G).to(device); D = copy.deepcopy(D).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for real, _ in loader:
            real = real.to(device)
            bsz = real.size(0)
            # D
            z = torch.randn(bsz, nz, 1, 1, device=device)
            fake = G(z).detach()
            d_real = D(real); d_fake = D(fake)
            lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            optD.zero_grad(); lossD.backward(); optD.step()
            # G
            z = torch.randn(bsz, nz, 1, 1, device=device)
            fake = G(z)
            d_fake = D(fake)
            lossG = bce(d_fake, torch.ones_like(d_fake))
            optG.zero_grad(); lossG.backward(); optG.step()
    return G.cpu().state_dict(), D.cpu().state_dict()

def fedavg_weighted(sd_list: List[Dict[str, torch.Tensor]], weights: List[float]):
    wsum = float(sum(weights)); norm_w = [w/wsum for w in weights]
    keys = sd_list[0].keys(); out = {}
    for k in keys:
        acc = 0.0
        for sd, w in zip(sd_list, norm_w):
            acc = acc + sd[k] * w
        out[k] = acc
    return out

# ---------- Evaluation ----------
@torch.no_grad()
def eval_bias(G, cls_net, device, n=2000, nz=NZ):
    G = copy.deepcopy(G).to(device).eval()
    cls_net = copy.deepcopy(cls_net).to(device).eval()
    z = torch.randn(n, nz, 1, 1, device=device)
    fake = G(z)
    logits = cls_net(fake)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1).cpu().tolist()
    even = sum(1 for p in preds if p % 2 == 0)
    odd  = n - even
    conf = probs.max(dim=1).values.mean().item()
    return {"n": n, "even": even, "odd": odd, "even_share": even/n, "odd_share": odd/n, "mean_conf": conf}

def save_samples(G, device, path, nrow=8, n=64, nz=NZ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    z = torch.randn(n, nz, 1, 1, device=device)
    with torch.no_grad():
        imgs = G.to(device)(z).cpu()
    vutils.save_image(imgs, path, normalize=True, value_range=(-1,1), nrow=nrow)

# ---------- Main ----------
def main():
    opt = parse_args()
    device = get_device(opt)

    # Normalize output path and create dir early
    out_dir = os.path.abspath(opt.out or "./out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] CWD: {os.getcwd()}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Rounds: {opt.rounds}, Local epochs: {opt.local_epochs}, Batch: {opt.batch_size}")
    sys.stdout.flush()

    if opt.rounds <= 0:
        raise ValueError("rounds must be >= 1")

    dls, sizes, cls_dl = get_dataloaders_5(batch_size=opt.batch_size)
    cls_net = train_classifier(cls_dl, device=device, epochs=1)

    G_global, D_global = Generator(), Discriminator()
    logs_even, logs_conf = [], []

    print("[INFO] Starting training loop"); sys.stdout.flush()
    for r in range(1, opt.rounds + 1):
        G_states, D_states = [], []
        for dl in dls:
            Ga_sd, Da_sd = local_gan_train(G_global, D_global, dl, device=device, epochs=opt.local_epochs)
            G_states.append(Ga_sd); D_states.append(Da_sd)

        G_global.load_state_dict(fedavg_weighted(G_states, sizes))
        D_global.load_state_dict(fedavg_weighted(D_states, sizes))

        img_path = os.path.join(out_dir, f"gen_round_{r}.png")
        save_samples(G_global, device, img_path)
        stats = eval_bias(G_global, cls_net, device, n=1000)
        logs_even.append(stats['even_share']); logs_conf.append(stats['mean_conf'])
        print(f"[Round {r}] even={stats['even']} odd={stats['odd']} "
              f"even_share={stats['even_share']:.3f} conf={stats['mean_conf']:.3f} -> saved {img_path}")
        sys.stdout.flush()

    ckpt_path = os.path.join(out_dir, "fed_gan_ckpt_5c.pt")
    csv_path  = os.path.join(out_dir, "metrics_5c.csv")
    torch.save({"G": G_global.state_dict(), "D": D_global.state_dict()}, ckpt_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["round","even_share","mean_conf"])
        for i,(e,c) in enumerate(zip(logs_even, logs_conf), 1): w.writerow([i, e, c])

    print(f"[DONE] Wrote: {ckpt_path}")
    print(f"[DONE] Wrote: {csv_path}")
    print(f"[DONE] Image grids in: {out_dir}")

if __name__ == "__main__":
    main()
