# run.py
# One-shot scaffolder for the federated generative fairness project.
# Usage: python run.py

import os, stat, textwrap, json, pathlib

ROOT = pathlib.Path("fedgen")

FILES = {
# ---------- top-level ----------
"runner.py": textwrap.dedent(r"""
import argparse, yaml, os, json, random
from pathlib import Path
from typing import Dict, Any
# Minimal runner stub that logs config and creates placeholder results.
# Replace TODOs with real training loop calling algos/* and models/*.

def load_cfg(path:str, overrides:Dict[str,Any])->Dict[str,Any]:
    with open(path) as f: cfg = yaml.safe_load(f)
    cfg.update(overrides or {})
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--algo", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_cfg(args.config, {"algo": args.algo} if args.algo else {})
    random.seed(args.seed)
    run_id = f"{Path(args.config).stem}__{cfg['algo']}__seed{args.seed}"
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO: build clients from splits, load model by cfg["model"], run FL rounds.
    # For now, write a dummy metrics JSON so the pipeline is end-to-end.
    dummy = {
        "run_id": run_id,
        "model": cfg["model"],
        "task": cfg["task"],
        "split": cfg["split"],
        "alpha": cfg.get("dirichlet_alpha"),
        "algo": cfg["algo"],
        "seed": args.seed,
        "metrics": {
            "FID": None, "KID": None, "Precision": None, "Recall": None,
            "FID_gap": None, "Mean_SA": None, "Cond_Disparity": None,
            "Bytes_per_round": None, "Rounds_to_target_FID": None, "Client_time_s": None
        }
    }
    (out_dir / "metrics.json").write_text(json.dumps(dummy, indent=2))
    print(f"[OK] Scaffold run logged at {out_dir}")

if __name__ == "__main__":
    main()
"""),
"run_matrix.sh": textwrap.dedent(r"""#!/usr/bin/env bash
set -e
MODELS=("dcgan64" "ddpm64")
SPLITS=("iid" "noniid_light" "noniid_heavy")
METHODS=("fedavg" "fedprox" "fedcafe")
SEEDS=(0 1 2)
mkdir -p results
for m in "${MODELS[@]}"; do
  for s in "${SPLITS[@]}"; do
    for a in "${METHODS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python fedgen/runner.py --config fedgen/configs/${m}_${s}.yaml --algo ${a} --seed ${seed}
      done
    done
  done
done
python fedgen/fairness/metrics.py --summarize runs --out results/summary.csv
echo "[OK] results/summary.csv"
"""),
# ---------- configs ----------
"configs/dcgan64_iid.yaml": textwrap.dedent(r"""
model: dcgan64
task: gan
rounds: 50
clients: 10
clients_per_round: 5
batch_size: 64
local_epochs: 1
split: iid
dirichlet_alpha:
algo: fedavg
mu: 0.0
lam_adv: 0.5
lam_cf: 1.0
lam_hid: 0.2
lam_div: 0.1
eta: 0.05
tau: 0.0
p_star: uniform
seed: 0
"""),
"configs/dcgan64_noniid_light.yaml": textwrap.dedent(r"""
model: dcgan64
task: gan
rounds: 50
clients: 10
clients_per_round: 5
batch_size: 64
local_epochs: 1
split: noniid
dirichlet_alpha: 0.5
algo: fedavg
mu: 0.0
lam_adv: 0.5
lam_cf: 1.0
lam_hid: 0.2
lam_div: 0.1
eta: 0.05
tau: 0.0
p_star: uniform
seed: 0
"""),
"configs/dcgan64_noniid_heavy.yaml": textwrap.dedent(r"""
model: dcgan64
task: gan
rounds: 50
clients: 10
clients_per_round: 5
batch_size: 64
local_epochs: 1
split: noniid
dirichlet_alpha: 0.2
algo: fedavg
mu: 0.0
lam_adv: 0.5
lam_cf: 1.0
lam_hid: 0.2
lam_div: 0.1
eta: 0.05
tau: 0.0
p_star: uniform
seed: 0
"""),
"configs/ddpm64_iid.yaml": textwrap.dedent(r"""
model: ddpm64
task: diffusion
rounds: 50
clients: 10
clients_per_round: 5
batch_size: 32
local_epochs: 1
split: iid
dirichlet_alpha:
algo: fedavg
mu: 0.0
lam_adv: 0.5
lam_cf: 1.0
lam_hid: 0.0
lam_div: 0.1
eta: 0.05
tau: 0.0
p_star: uniform
seed: 0
"""),
"configs/ddpm64_noniid_light.yaml": textwrap.dedent(r"""
model: ddpm64
task: diffusion
rounds: 50
clients: 10
clients_per_round: 5
batch_size: 32
local_epochs: 1
split: noniid
dirichlet_alpha: 0.5
algo: fedavg
mu: 0.0
lam_adv: 0.5
lam_cf: 1.0
lam_hid: 0.0
lam_div: 0.1
eta: 0.05
tau: 0.0
p_star: uniform
seed: 0
"""),
"configs/ddpm64_noniid_heavy.yaml": textwrap.dedent(r"""
model: ddpm64
task: diffusion
rounds: 50
clients: 10
clients_per_round: 5
batch_size: 32
local_epochs: 1
split: noniid
dirichlet_alpha: 0.2
algo: fedavg
mu: 0.0
lam_adv: 0.5
lam_cf: 1.0
lam_hid: 0.0
lam_div: 0.1
eta: 0.05
tau: 0.0
p_star: uniform
seed: 0
"""),
# ---------- algos ----------
"algos/fedavg.py": textwrap.dedent(r"""
# FedAvg aggregator stub
def aggregate(params_list, weights=None):
    if not params_list: return None
    if weights is None:
        weights = [1.0/len(params_list)]*len(params_list)
    out = {}
    keys = params_list[0].keys()
    for k in keys:
        out[k] = sum(w * p[k] for w,p in zip(weights, params_list))
    return out
"""),
"algos/fedprox.py": textwrap.dedent(r"""
# FedProx proximal term stub (to be added inside local train loop)
def prox_loss(params, global_params, mu):
    # params/global_params: dict of tensors in real code. Here numeric stub.
    # Implement per-parameter L2 in your training step.
    return mu
"""),
"algos/fedcafe.py": textwrap.dedent(r"""
# Fed-CaFe minimal utilities (placeholders; integrate into your training)
def wasserstein_hist(p_hat, p_star):
    # p_hat, p_star: lists/histograms that sum to 1
    from itertools import accumulate
    def cdf(a): 
        return list(accumulate(a))
    c1, c2 = cdf(p_hat), cdf(p_star)
    return sum(abs(x-y) for x,y in zip(c1,c2))

def bias_aware_weights(P_hats, p_star, eps=1e-6):
    W = [wasserstein_hist(p, p_star) for p in P_hats]
    inv = [1.0/(eps+w) for w in W]
    s = sum(inv)
    return [x/s for x in inv], W
"""),
# ---------- models ----------
"models/dcgan.py": textwrap.dedent(r"""
# Minimal placeholders; replace with real torch modules.
class Generator64:
    def __init__(self, z_dim=128): self.z_dim=z_dim
    def state_dict(self): return {"w": 0.0}
    def load_state_dict(self, d): pass

class Discriminator64:
    def __init__(self): pass
"""),
"models/ddpm_unet.py": textwrap.dedent(r"""
# Placeholder UNet/diffusion
class UNet64:
    def __init__(self): pass
    def state_dict(self): return {"w": 0.0}
    def load_state_dict(self, d): pass
"""),
"models/heads.py": textwrap.dedent(r"""
# Placeholders for race head C and probe A
class RaceHead:
    def __init__(self, classes=4): self.classes=classes

class ProbeA:
    def __init__(self): pass
"""),
# ---------- splits ----------
"splits/make_splits.py": textwrap.dedent(r"""
# Dirichlet-based race splits (light/heavy skew)
def dirichlet_race_splits(meta_rows, num_clients, alpha, seed=0):
    """
    #meta_rows: list of dicts with keys {"path","race"}
    #returns: dict client_id -> list of paths
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    races = sorted({r["race"] for r in meta_rows})
    by_race = {r: [m["path"] for m in meta_rows if m["race"]==r] for r in races}
    client_bins = {i: [] for i in range(num_clients)}
    priors = np.ones(len(races))*alpha
    for r in races:
        alloc = rng.dirichlet(priors)
        idx = np.arange(len(by_race[r])); rng.shuffle(idx)
        chunks = np.array_split(idx, num_clients)
        for i,(a,ix) in enumerate(zip(alloc, chunks)):
            client_bins[i].extend([by_race[r][k] for k in ix])
    return client_bins
"""),
# ---------- fairness ----------
"fairness/race_clf.py": textwrap.dedent(r"""
# Stub: hook in a pretrained race classifier for scoring generated images.
class RaceClassifier:
    def __init__(self, labels=("raceA","raceB","raceC","raceD")):
        self.labels = labels
    def predict(self, imgs): 
        # return dummy labels same length as imgs
        return [self.labels[0]]*len(imgs)
"""),
"fairness/metrics.py": textwrap.dedent(r"""
import argparse, json, os, csv

def gather(root):
    rows=[]
    for dirpath,_,files in os.walk(root):
        if "metrics.json" in files:
            p=os.path.join(dirpath,"metrics.json")
            with open(p) as f: rows.append(json.load(f))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summarize", required=True, help="runs/ dir")
    ap.add_argument("--out", default="results/summary.csv")
    args=ap.parse_args()
    rows=gather(args.summarize)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols=["run_id","model","task","split","alpha","algo","seed"]
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in cols})
    print(f"[OK] wrote {args.out}")

if __name__=="__main__":
    main()
"""),
}

def write_files():
    for rel, content in FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n")
    # make run_matrix.sh executable
    p = ROOT / "run_matrix.sh"
    if p.exists():
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    write_files()
    print("[OK] Scaffolding created under ./fedgen")
    print("Next:")
    print("  1) Implement real models and training in models/* and runner.py")
    print("  2) Add FedProx term in local loop, and Fed-CaFe loss + weighted aggregation")
    print("  3) bash fedgen/run_matrix.sh")

if __name__ == "__main__":
    main()
