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
