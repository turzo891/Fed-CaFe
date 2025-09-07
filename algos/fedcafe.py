# algos/fedcafe.py (loss utilities)
def wasserstein_hist(p_hat, p_star):
    # p_hat, p_star: numpy histograms over races (sum=1)
    import numpy as np
    cdf1, cdf2 = np.cumsum(p_hat), np.cumsum(p_star)
    return np.sum(np.abs(cdf1 - cdf2))

def fedcafe_loss_gen(
    L_gen,                 # standard GAN or diffusion loss (scalar)
    logpA,                 # log p_A(z|h) from probe A on G's mid features
    x_hat, x_cf,           # generated image and counterfactual (z -> z')
    mask_non_target,       # 0/1 mask where distance applies (keep identity)
    ce_C,                  # CE(C(x_hat), z) if conditional, else 0
    W_local,               # W(π_hat_i, π*)
    lam_adv=0.5, lam_cf=1.0, lam_hid=0.2, lam_div=0.1
):
    # LPIPS/VGG perceptual distance on non-target regions
    dist_nt = perceptual_dist(x_hat*mask_non_target, x_cf*mask_non_target)
    return (L_gen
            + lam_adv * (-logpA)
            + lam_cf  * dist_nt
            + lam_hid * ce_C
            + lam_div * W_local)
