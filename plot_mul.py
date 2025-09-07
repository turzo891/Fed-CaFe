from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# Search likely roots
roots = [Path.cwd(), Path.cwd() / "Flower/flower/baselines/fedprox"]

pkls = []
for r in roots:
    pkls += list(r.glob("results/**/results*.pkl"))
pkls = sorted(pkls, key=lambda p: p.stat().st_mtime)

if not pkls:
    raise SystemExit("No results*.pkl files found. Run a simulation or fix the search paths.")

plotted = 0
for f in pkls:
    try:
        h = pickle.load(open(f, "rb"))
        hist = h["history"]
        metrics = getattr(hist, "metrics_centralized", {})
        acc = metrics.get("accuracy")
        if not acc:
            print("Skipping (no accuracy):", f)
            continue

        rounds, vals = zip(*acc)
        parts = f.parts
        ds = parts[parts.index("results")+1] if "results" in parts else f.parent.name
        plt.plot(rounds, vals, label=f"{ds} (final={vals[-1]:.2f})")
        plotted += 1
    except Exception as e:
        print("Skipping", f, ":", e)

if plotted == 0:
    raise SystemExit(f"Found {len(pkls)} files, but none had history.metrics_centralized['accuracy'].")

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("FedProx Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
