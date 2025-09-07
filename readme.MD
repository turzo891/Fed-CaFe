```markdown
# Fed-CaFe: Federated Counterfactual Fairness for Generative Models

Minimal research scaffold to compare **FedAvg**, **FedProx**, and **Fed-CaFe** on synthetic DCGAN-64 and DDPM-64 tasks with IID vs non-IID racial skew.  
Outputs include per-run `metrics.json` and CSV summaries.

---

## Repo layout

```

fedgen/
runner.py
run\_matrix.sh
configs/
dcgan64\_{iid,noniid\_light,noniid\_heavy}.yaml
ddpm64\_{iid,noniid\_light,noniid\_heavy}.yaml
algos/{fedavg.py,fedprox.py,fedcafe.py}
models/{dcgan.py,ddpm\_unet.py,heads.py}
fairness/{metrics.py}
splits/{make\_splits.py}
runs/           # auto-created; per-run folders with metrics.json
results/        # auto-created; CSV summaries and plots (optional)

````

---

## Requirements

- Python **3.10+**
- CPU is fine; GPU optional
- pip packages:
  ```bash
  pip3 install torch torchvision numpy pyyaml
````

---

## 1) Scaffold

From repo root:

```bash
python3 run.py
```

This creates the `fedgen/` folder with configs, runner, and scripts.

---

## 2) Prepare the matrix script

```bash
sed -i 's/\bpython\b/python3/g' fedgen/run_matrix.sh
chmod +x fedgen/run_matrix.sh
```

---

## 3) Install deps

```bash
pip3 install torch torchvision numpy pyyaml
```

---

## 4) Run the experiment matrix

Runs 2 models × 3 splits × 3 methods × seeds {0,1,2}.

```bash
./fedgen/run_matrix.sh
```

Each job writes `runs/<model>_<split>__<algo>__seed<k>/metrics.json` and updates `results/summary.csv`.

---

## 5) Inspect outputs quickly

```bash
ls runs | head
column -t -s, results/summary.csv | head
```

Single-run example:

```bash
python3 fedgen/runner.py \
  --config fedgen/configs/dcgan64_noniid_heavy.yaml \
  --algo fedcafe \
  --seed 0
```

---

## Configuration knobs

Edit YAML in `fedgen/configs/*`:

* `rounds` — global FL rounds
* `clients`, `clients_per_round`
* `batch_size`, `local_epochs`
* `split: iid | noniid`

  * for noniid: set `dirichlet_alpha` (e.g., 0.2 heavy, 0.5 light)
* `algo: fedavg | fedprox | fedcafe`
* FedProx: `mu`
* Fed-CaFe: `lam_adv`, `lam_cf`, `lam_hid`, `lam_div`, `eta`, `tau`, `p_star`

---

## Metrics written

Each `metrics.json` contains:

```json
{
  "run_id": "dcgan64_noniid_heavy__fedcafe__seed0",
  "model": "dcgan64",
  "task": "gan",
  "split": "noniid",
  "alpha": 0.2,
  "algo": "fedcafe",
  "seed": 0,
  "metrics": {
    "Data_Wasserstein": <float|null>,      // distribution skew across clients (train histogram agg)
    "Global_data_hist": [p0,p1,p2,p3],     // aggregated client label histogram
    "Data_Wasserstein_gen": <float|null>,  // GAN-only: gen-outputs vs uniform
    "Global_gen_hist": [q0,q1,q2,q3]       // GAN-only: inferred race histogram from generated images
  }
}
```

CSV summaries are written to `results/summary.csv`.
Aggregate further with your own scripts.

---

## Reproducing a smaller run

For faster debugging, reduce rounds and dataset size:

```bash
# edit configs to small numbers (e.g., rounds: 5, clients: 4, clients_per_round: 2)
python3 fedgen/runner.py --config fedgen/configs/dcgan64_iid.yaml --algo fedavg --seed 0
```

---

## Tips

* Force CPU:

  ```bash
  CUDA_VISIBLE_DEVICES="" ./fedgen/run_matrix.sh
  ```
* More clients or rounds will increase runtime.
* `fedgen/runner.py` currently includes simple synthetic data and placeholder models to keep runs quick. Replace with real models and datasets when ready.

---

## Troubleshooting

* `python: command not found` → use `python3` and ensure it’s on `PATH`.
* `ModuleNotFoundError` → reinstall deps with `pip3 install ...`.
* Permission errors → run `chmod +x fedgen/run_matrix.sh`.

---

## Citation

If you compare against FedProx, please cite the original paper by **Li et al., 2018** and any other baselines you use.

```
@article{li2018federated,
  title={Federated Optimization in Heterogeneous Networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  journal={arXiv:1812.06127},
  year={2018}
}
```

---

## License
GNU

```
```
 
 ## TO Enable GPU

 pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0

 # To Check GPU
python -c "import torch; print('cuda?', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
