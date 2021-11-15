import torch

def fm(feature_map: str = None):
    if (feature_map is None) or (feature_map == "elu"):
        return elu_feature_map
    elif feature_map == "exp":
        return exp_feature_map
    elif feature_map == "dpfp":
        return dpfp_feature_map
    else:
        raise ValueError(f"Unknown feature map {feature_map}")


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


def exp_feature_map(x):
    return torch.exp(x)


def dpfp_feature_map(x, nu=1):
    x_ = torch.cat([
      torch.nn.functional.relu(x), torch.nn.functional.relu(-x)
    ], dim=-1)

    x_rolled = torch.cat([
        x_.roll(shifts=j, dims=-1) for j in range(1, nu + 1)
    ], dim=-1)

    x_repeat = torch.cat([x_] * nu, dim=-1)

    return x_repeat * x_rolled