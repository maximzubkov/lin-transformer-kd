import torch


def fm(feature_map: str = None, num_heads: int = None, hidden_size: int = None):
    if (feature_map is None) or (feature_map == "elu"):
        return elu_feature_map
    elif feature_map == "exp":
        return exp_feature_map
    elif feature_map == "dpfp":
        return dpfp_feature_map
    elif feature_map == "approx":
        return ApproxFM(num_heads=num_heads, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown feature map {feature_map}")


class ApproxFM(torch.nn.Module):
    def __init__(self, num_heads: int, hidden_size: int, degree: int = 2):
        super(ApproxFM, self).__init__()
        self.hidden_size = hidden_size
        self.degree = degree
        w_ = torch.rand(1, 1, num_heads, hidden_size // num_heads, degree + 1)
        self.w = torch.nn.Parameter(w_, requires_grad=True)

    def forward(self, hidden_state):
        expanded = hidden_state.unsqueeze(-1)
        degrees = [torch.ones_like(expanded, device=expanded.device)]
        for i in range(self.degree):
            degrees.append(expanded ** (i + 1))
        powers_of_hidden = torch.cat(degrees, dim=-1)
        ouptut = (powers_of_hidden * self.w).sum(-1)
        return exp_feature_map(ouptut)


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


def exp_feature_map(x):
    return torch.exp(x - torch.max(x))


def dpfp_feature_map(x, nu=1):
    x_ = torch.cat([
      torch.nn.functional.relu(x), torch.nn.functional.relu(-x)
    ], dim=-1)

    x_rolled = torch.cat([
        x_.roll(shifts=j, dims=-1) for j in range(1, nu + 1)
    ], dim=-1)

    x_repeat = torch.cat([x_] * nu, dim=-1)

    return x_repeat * x_rolled
