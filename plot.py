from pathlib import Path
import pickle, glob, matplotlib.pyplot as plt

roots = [
    Path("."), 
    Path("./Flower/flower/baselines/fedprox"),
    Path("./flower/baselines/fedprox"),
]
candidates = []
for r in roots:
    candidates += list(r.glob("results/**/results*.pkl"))

if not candidates:
    raise SystemExit("No results*.pkl found. Adjust the path or run a simulation first.")

p = sorted(candidates, key=lambda x: x.stat().st_mtime)[-1]
print("Loaded:", p)

h = pickle.load(open(p, "rb"))
rounds, acc = zip(*h["history"].metrics_centralized["accuracy"])

plt.plot(rounds, acc, marker="o")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("FedProx Accuracy over Rounds")
plt.grid(True); plt.show()
