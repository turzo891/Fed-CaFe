import argparse, yaml, os, json, random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from algos.fedavg import aggregate as fedavg_aggregate
from algos.fedprox import prox_loss as fedprox_reg
from algos.fedcafe import bias_aware_weights, wasserstein_hist

from models.dcgan import Generator64, Discriminator64
from models.ddpm_unet import UNet64
from models.heads import RaceHead, ProbeA

# -------------------- eval on generated samples (GAN only) --------------------
def _palette(device):
    return torch.tensor(
        [
            [0.9, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.9],
            [0.9, 0.9, 0.2],
        ],
        device=device,
    )

def eval_generated_hist(G_state, device: str, samples: int = 512, bs: int = 64, groups: int = 4):
    G = Generator64(z_dim=128, groups=groups).to(device)
    G.load_state_dict(G_state, strict=False)
    G.eval()
    ref = _palette(device)  # (4,3)
    counts = [0] * groups
    import math

    iters = math.ceil(samples / bs)
    done = 0
    with torch.no_grad():
        for _ in range(iters):
            b = min(bs, samples - done)
            if b <= 0:
                break
            z = torch.randn(b, 128, device=device)
            y = torch.randint(0, groups, (b,), device=device)  # uniform labels
            img, _ = G(z, y)  # (b,3,64,64) in [0,1]
            center = img[:, :, 20:44, 20:44].mean(dim=(2, 3))  # (b,3)
            d = ((center.unsqueeze(1) - ref.unsqueeze(0)) ** 2).sum(dim=2)  # (b,4)
            pred = d.argmin(dim=1).tolist()
            for t in pred:
                counts[t] += 1
            done += b
    tot = sum(counts) or 1
    return [c / tot for c in counts]

# -------------------- synthetic dataset --------------------
class SyntheticFaces(Dataset):
    def __init__(self, num_samples: int, races: int = 4, seed: int = 0, race_hist: List[float] = None):
        g = torch.Generator().manual_seed(seed)
        self.races = races
        if race_hist is None:
            race_hist = [1.0 / races] * races
        probs = torch.tensor(race_hist, dtype=torch.float)
        idx = torch.multinomial(probs, num_samples=num_samples, replacement=True, generator=g)
        self.labels = idx.tolist()
        self.num_samples = num_samples
        self.g = g

    def _base_patch(self, y: int) -> torch.Tensor:
        colors = torch.tensor(
            [
                [0.9, 0.2, 0.2],
                [0.2, 0.9, 0.2],
                [0.2, 0.2, 0.9],
                [0.9, 0.9, 0.2],
            ]
        )
        c = colors[y % colors.size(0)]
        img = torch.rand((3, 64, 64), generator=self.g) * 0.15
        img[:, 20:44, 20:44] = c.view(3, 1, 1)          # center “attribute” patch
        img[:, 10:18, 10:54] += 0.1 * c.view(3, 1, 1)  # stripe
        img = img.clamp(0.0, 1.0)
        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        y = self.labels[idx]
        x = self._base_patch(y)
        return x, y

def client_hist(labels: List[int], G: int = 4) -> List[float]:
    cnt = [0] * G
    for y in labels:
        cnt[y] += 1
    s = sum(cnt) if sum(cnt) > 0 else 1
    return [c / s for c in cnt]

def make_client_race_hists(num_clients: int, alpha: float, G: int = 4, seed: int = 0) -> List[List[float]]:
    torch.manual_seed(seed)
    alpha_vec = torch.full((G,), alpha)
    hists = []
    for _ in range(num_clients):
        samp = torch.distributions.Dirichlet(alpha_vec).sample()  # no generator kwarg in PyTorch API
        hists.append((samp / samp.sum()).tolist())
    return hists

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------- local training --------------------
bce = nn.BCELoss()
ce = nn.CrossEntropyLoss()

def train_local_gan(
    cfg: Dict[str, Any],
    device: str,
    global_G_state: Dict[str, torch.Tensor],
    local_data: DataLoader,
    race_hist: List[float],
    algo: str,
    mu: float,
    lam_adv: float,
    lam_cf: float,
    lam_hid: float,
    lam_div: float,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    zdim = 128
    groups = 4

    G = Generator64(z_dim=zdim, groups=groups).to(device)
    D = Discriminator64().to(device)
    C = RaceHead(classes=groups).to(device)
    A = ProbeA(in_ch=64, classes=groups).to(device)

    G.load_state_dict(global_G_state, strict=False)

    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optC = optim.Adam(C.parameters(), lr=1e-3)
    optA = optim.Adam(A.parameters(), lr=1e-3)

    ones = lambda n: torch.ones(n, 1, device=device)
    zeros = lambda n: torch.zeros(n, 1, device=device)

    G.train(); D.train(); C.train(); A.train()
    batches = 0

    mask = torch.ones((1, 1, 64, 64), device=device)
    mask[:, :, 20:44, 20:44] = 0.0  # exclude attribute patch from counterfactual consistency

    for _ in range(cfg["local_epochs"]):
        for real, y in local_data:
            real = real.to(device); y = y.to(device)
            bs = real.size(0)

            # D step
            optD.zero_grad()
            z = torch.randn(bs, zdim, device=device)
            y_fake = torch.randint(0, groups, (bs,), device=device)
            fake, _ = G(z, y_fake)
            d_real = D(real)
            d_fake = D(fake.detach())
            lossD = bce(d_real, ones(bs)) + bce(d_fake, zeros(bs))
            lossD.backward(); optD.step()

            # A step (probe)
            optA.zero_grad()
            z = torch.randn(bs, zdim, device=device)
            fake_for_A, hA = G(z, y)
            logitsA = A(hA)
            lossA = ce(logitsA, y)
            lossA.backward(); optA.step()

            # C step (attribute head)
            optC.zero_grad()
            with torch.no_grad():
                z = torch.randn(bs, zdim, device=device)
                fake_for_C, _ = G(z, y)
            logitsC = C(fake_for_C)
            lossC = ce(logitsC, y)
            lossC.backward(); optC.step()

            # G step
            optG.zero_grad()
            z = torch.randn(bs, zdim, device=device)
            fake, h = G(z, y)
            dg = D(fake)
            L_gen = bce(dg, ones(bs))

            prox = 0.0
            if algo == "fedprox":
                prox = fedprox_reg(G.state_dict(), global_G_state, mu)

            extra = 0.0
            if algo == "fedcafe":
                # adversarial invariance on features h
                logitsA_g = A(h)
                inv = -ce(logitsA_g, y)
                # counterfactual consistency off the attribute patch
                y_cf = (y + torch.randint(1, groups, (bs,), device=device)) % groups
                fake_cf, _ = G(z, y_cf)
                dist_nt = torch.mean(torch.abs((fake - fake_cf) * mask))
                # keep attribute head predictive
                logitsC_g = C(fake)
                ce_C = ce(logitsC_g, y)
                # local distribution penalty placeholder
                W_local = 0.0

                extra = (
                    lam_adv * inv
                    + lam_cf * dist_nt
                    + lam_hid * ce_C
                    + lam_div * W_local
                )

            (L_gen + prox + extra).backward()
            optG.step()

            batches += 1
            if batches >= 100:  # extended for your current setup
                break

    # local label histogram
    lbls = [int(y) for _, y in local_data.dataset]
    pi_hat = client_hist(lbls, groups)
    return G.state_dict(), pi_hat

def train_local_diffusion(
    cfg: Dict[str, Any],
    device: str,
    global_E_state: Dict[str, torch.Tensor],
    local_data: DataLoader,
    race_hist: List[float],
    algo: str,
    mu: float,
) -> Dict[str, torch.Tensor]:
    E = UNet64().to(device)
    E.load_state_dict(global_E_state, strict=False)
    opt = optim.Adam(E.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    for _ in range(cfg["local_epochs"]):
        steps = 0
        for x, _ in local_data:
            x = x.to(device)
            noise = torch.randn_like(x)
            x_noisy = (x + 0.2 * noise).clamp(0, 1)
            pred = E(x_noisy)
            loss = mse(pred, noise)
            if algo == "fedprox":
                loss = loss + fedprox_reg(E.state_dict(), global_E_state, mu)
            loss.backward(); opt.step(); opt.zero_grad()
            steps += 1
            if steps >= 100:
                break
    return E.state_dict()

# -------------------- federated orchestration --------------------
def load_cfg(path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--algo", default=None, choices=[None, "fedavg", "fedprox", "fedcafe"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.config, {"algo": args.algo} if args.algo else {})
    algo = cfg["algo"]
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = f"{Path(args.config).stem}__{algo}__seed{args.seed}"
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    clients = cfg["clients"]
    clients_per_round = cfg["clients_per_round"]
    groups = 4
    if cfg["split"] == "iid":
        hists = [[1.0 / groups] * groups for _ in range(clients)]
    else:
        alpha = float(cfg["dirichlet_alpha"])
        hists = make_client_race_hists(clients, alpha=alpha, G=groups, seed=args.seed)

    local_loaders = []
    for i in range(clients):
        ds = SyntheticFaces(num_samples=1000, races=groups, seed=1000 + args.seed * 13 + i, race_hist=hists[i])
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
        local_loaders.append(dl)

    rounds = int(cfg["rounds"])
    if cfg["task"] == "gan":
        G_global = Generator64(z_dim=128, groups=groups).to(device)
        G_global_sd = G_global.state_dict()
    else:
        E_global = UNet64().to(device)
        E_global_sd = E_global.state_dict()

    hist_glob = [0] * groups
    for i_round in range(rounds):
        idxs = random.sample(range(clients), clients_per_round)
        updates = []
        pi_hats = []

        if cfg["task"] == "gan":
            for i in idxs:
                sd_i, pi_i = train_local_gan(
                    cfg, device, G_global_sd, local_loaders[i], hists[i], algo,
                    mu=float(cfg.get("mu", 0.0)),
                    lam_adv=float(cfg.get("lam_adv", 0.5)),
                    lam_cf=float(cfg.get("lam_cf", 1.0)),
                    lam_hid=float(cfg.get("lam_hid", 0.2)),
                    lam_div=float(cfg.get("lam_div", 0.1)),
                )
                updates.append(sd_i)
                pi_hats.append(pi_i)
            if algo == "fedcafe":
                w, _ = bias_aware_weights(pi_hats, [1.0 / groups] * groups)
                keys = updates[0].keys()
                new_sd = {k: sum(wj * updates[j][k] for j, wj in enumerate(w)) for k in keys}
                G_global_sd = new_sd
            else:
                G_global_sd = fedavg_aggregate(updates)
        else:
            for i in idxs:
                sd_i = train_local_diffusion(
                    cfg, device, E_global_sd, local_loaders[i], hists[i], algo,
                    mu=float(cfg.get("mu", 0.0)),
                )
                updates.append(sd_i)
            E_global_sd = fedavg_aggregate(updates)

        for pi in pi_hats or []:
            for g in range(groups):
                hist_glob[g] += pi[g]

        if (i_round + 1) % 5 == 0:
            print(f"[round {i_round+1}/{rounds}]", flush=True)

    if hist_glob and sum(hist_glob) > 0:
        s = sum(hist_glob)
        pi_glob = [x / s for x in hist_glob]
        w_dist = wasserstein_hist(pi_glob, [1.0 / groups] * groups)
    else:
        pi_glob = None
        w_dist = None

    # generated-output fairness (GAN only)
    hist_gen = None
    w_gen = None
    if cfg["task"] == "gan":
        hist_gen = eval_generated_hist(G_global_sd, device, samples=512, bs=64, groups=groups)
        w_gen = wasserstein_hist(hist_gen, [1.0 / groups] * groups)

    out = {
        "run_id": run_id,
        "model": cfg["model"],
        "task": cfg["task"],
        "split": cfg["split"],
        "alpha": cfg.get("dirichlet_alpha"),
        "algo": algo,
        "seed": args.seed,
        "metrics": {
            "Data_Wasserstein": w_dist,
            "Global_data_hist": pi_glob,
            "Data_Wasserstein_gen": w_gen,
            "Global_gen_hist": hist_gen,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"[OK] Run complete -> {out_dir/'metrics.json'}")

if __name__ == "__main__":
    main()
