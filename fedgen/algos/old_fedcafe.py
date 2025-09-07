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
