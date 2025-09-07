# splits/make_splits.py
def dirichlet_race_splits(meta_df, num_clients, alpha):
    # meta_df: columns [path, race]
    import numpy as np, collections
    races = sorted(meta_df['race'].unique())
    priors = np.ones(len(races))*alpha
    client_bins = {i: [] for i in range(num_clients)}
    by_race = {r: meta_df[meta_df.race==r]['path'].tolist() for r in races}
    rng = np.random.default_rng(0)
    for r in races:
        alloc = rng.dirichlet(priors)
        idx = np.arange(len(by_race[r]))
        rng.shuffle(idx)
        splits = np.array_split(idx, num_clients)
        for i,(a,ix) in enumerate(zip(alloc, splits)):
            client_bins[i].extend([by_race[r][k] for k in ix])
    return client_bins
# α=5.0 -> light, α=0.2 -> heavy. α=∞ approximates IID.
