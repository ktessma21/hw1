import torch
from cs336_basics.Linear import Linear


def test_linear():
    layer = Linear(in_features=3, out_features=5)
    weights = torch.randn(5, 3)
    layer.load_state_dict({"W": weights})

    x = torch.randn(2, 4, 3)
    expected = torch.nn.functional.linear(x, weights)

    torch.testing.assert_close(layer(x), expected)