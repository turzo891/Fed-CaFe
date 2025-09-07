# Dirichlet-based race splits (light/heavy skew)
def dirichlet_race_splits(meta_rows, num_clients, alpha, seed=0):

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
