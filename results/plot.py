# Create comparison charts from uploaded CSVs and save PNGs to /mnt/data
import os, ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from caas_jupyter_tools import display_dataframe_to_user

base = Path("/mnt/data")

# Load CSVs if present
def safe_read(path):
    p = base / path
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            return None
    return None

agg = safe_read("agg.csv")
data_means = safe_read("data_W_means.csv")
gen_means = safe_read("gen_W_means.csv")
means = safe_read("means.csv")
summary = safe_read("summary.csv")

# 1) Data Wasserstein by (model, split, algo) — bar per model/split
charts = []

if data_means is not None and {"model","split","algo","mean","n"}.issubset(data_means.columns):
    for (model, split), df in data_means.groupby(["model","split"]):
        fig = plt.figure()
        plt.bar(df["algo"], df["mean"])
        plt.title(f"Mean Data Wasserstein ↓\nmodel={model}, split={split}")
        plt.xlabel("Method")
        plt.ylabel("Mean W(data→uniform)")
        out = base / f"chart_dataW_{model}_{split}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)
        charts.append(out)

# 2) Gen Wasserstein by algo per model/split
if gen_means is not None and {"model","split","algo","mean","n"}.issubset(gen_means.columns):
    for (model, split), df in gen_means.groupby(["model","split"]):
        fig = plt.figure()
        plt.bar(df["algo"], df["mean"])
        plt.title(f"Mean Generated Wasserstein ↓\nmodel={model}, split={split}")
        plt.xlabel("Method")
        plt.ylabel("Mean W(gen→uniform)")
        out = base / f"chart_genW_{model}_{split}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)
        charts.append(out)

# 3) Train vs Generated histogram example for one run if available
if agg is not None and {"run_id","gen_hist","train_hist","model","split","algo"}.issubset(agg.columns):
    # Pick the first row that has both hists
    row = None
    for _, r in agg.iterrows():
        if isinstance(r["gen_hist"], str) and isinstance(r["train_hist"], str):
            try:
                gh = ast.literal_eval(r["gen_hist"])
                th = ast.literal_eval(r["train_hist"])
                if isinstance(gh, list) and isinstance(th, list) and len(gh)==len(th)==4:
                    row = r
                    break
            except Exception:
                continue
    if row is not None:
        gh = ast.literal_eval(row["gen_hist"])
        th = ast.literal_eval(row["train_hist"])
        x = list(range(len(gh)))
        width = 0.35
        fig = plt.figure()
        plt.bar([i - width/2 for i in x], th, width=width, label="Train mix")
        plt.bar([i + width/2 for i in x], gh, width=width, label="Generated mix")
        plt.xticks(x, [f"g{i}" for i in x])
        plt.legend()
        plt.title(f"Train vs Generated mix\n{row['model']} | {row['split']} | {row['algo']} | {row['run_id']}")
        plt.xlabel("Subgroup")
        plt.ylabel("Proportion")
        out = base / f"chart_train_vs_gen_{row['run_id']}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)
        charts.append(out)

# 4) Stereotype Amplification (SA) per algo (only for rows with both histograms)
if agg is not None and {"algo","gen_hist","train_hist"}.issubset(agg.columns):
    sa_rows = []
    for _, r in agg.iterrows():
        if isinstance(r["gen_hist"], str) and isinstance(r["train_hist"], str):
            try:
                gh = ast.literal_eval(r["gen_hist"])
                th = ast.literal_eval(r["train_hist"])
                if isinstance(gh, list) and isinstance(th, list) and len(gh)==len(th)==4:
                    # SA_g = gen/train - 1
                    sa = [(gh[i]/max(th[i],1e-9))-1 for i in range(4)]
                    for i, val in enumerate(sa):
                        sa_rows.append({"algo": r["algo"], "group": f"g{i}", "SA": val})
            except Exception:
                pass
    if sa_rows:
        sa_df = pd.DataFrame(sa_rows)
        # Boxplot of SA by algo
        fig = plt.figure()
        # Assemble data for boxplot grouped by algo
        algo_groups = sorted(sa_df["algo"].unique())
        data = [sa_df.loc[sa_df["algo"]==a, "SA"].values for a in algo_groups]
        plt.boxplot(data, labels=algo_groups, showfliers=False)
        plt.title("Stereotype Amplification by method\nSA = gen/train - 1 (all groups pooled)")
        plt.ylabel("SA (lower is better, 0 ideal)")
        out = base / "chart_SA_by_algo.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)
        charts.append(out)
        # Save table too
        display_dataframe_to_user("StereotypeAmplificationByAlgo", sa_df)

# Save and display the input summaries if available
for name, df in [
    ("AggregateRuns", agg),
    ("DataMeans", data_means),
    ("GenMeans", gen_means),
    ("MeansLegacy", means),
    ("SummaryRaw", summary),
]:
    if df is not None:
        display_dataframe_to_user(name, df)

# List generated charts
[str(p) for p in charts]
