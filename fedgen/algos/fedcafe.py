from itertools import accumulate
import torch
import torch.nn.functional as F

def wasserstein_hist(p_hat, p_star):
    c1 = list(accumulate(p_hat))
    c2 = list(accumulate(p_star))
    return sum(abs(x - y) for x, y in zip(c1, c2))

def bias_aware_weights(P_hats, p_star, eps=1e-6):
    W = [wasserstein_hist(p, p_star) for p in P_hats]
    inv = [1.0 / (eps + w) for w in W]
    s = sum(inv)
    return [x / s for x in inv], W

def perceptual_dist_l1(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    return torch.mean(torch.abs((x - y) * mask))
