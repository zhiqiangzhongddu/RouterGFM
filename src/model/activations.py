from torch import nn


_ACT_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "prelu": nn.PReLU,
    "selu": nn.SELU,
}


def get_activation(name: str) -> nn.Module:
    """Return an activation module from a string key."""
    key = (name or "relu").lower()
    act = _ACT_MAP.get(key)
    if act is None:
        raise ValueError(f"Unsupported activation: {name}")
    return act()
