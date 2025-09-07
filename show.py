from pathlib import Path
import pickle, matplotlib.pyplot as plt

candidates = [p for p in Path(".").rglob("results*.pkl") if "fedprox" in str(p).lower()]
if not candidates:
    raise SystemExit("No results*.pkl found under this directory. Run a simulation first.")

p = max(candidates, key=lambda x: x.stat().st_mtime)
h = pickle.load(open(p, "rb"))
rounds, acc = zip(*h["history"].metrics_centralized["accuracy"])
plt.plot(rounds, acc, marker="o"); plt.grid(True); plt.show()
