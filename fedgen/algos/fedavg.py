# FedAvg aggregator stub
def aggregate(params_list, weights=None):
    if not params_list: return None
    if weights is None:
        weights = [1.0/len(params_list)]*len(params_list)
    out = {}
    keys = params_list[0].keys()
    for k in keys:
        out[k] = sum(w * p[k] for w,p in zip(weights, params_list))
    return out
