import torch

def prox_loss(params, global_params, mu: float):
    if mu == 0.0:
        return 0.0
    reg = 0.0
    for k, p in params.items():
        if k in global_params and isinstance(p, torch.Tensor):
            gp = global_params[k].to(p.device)
            reg = reg + torch.sum((p - gp) ** 2)
    return 0.5 * mu * reg
