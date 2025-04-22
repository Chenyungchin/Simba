from torch import nn

from typing import List

from simplicity_bias.util.models.activations import SinActivation, GaussianActivation


SUPPORTED_ACTIVATIONS = [
    nn.ReLU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.SiLU,
    nn.SELU,
    SinActivation,
    GaussianActivation,
]


class DummyNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.l1(x)


def initialize_weights(
    model: nn.Module,
    weight_mode: str = "uniform",
    bias_mode="zero",
    W_amplitude=1.0,
    b_amplitude=0.0,
):
    assert weight_mode in ["uniform", "normal", "xavier_uniform", "xavier_normal"]
    assert bias_mode in ["uniform", "normal", "zero"]
    if bias_mode != "zero":
        assert b_amplitude > 0.0, "Bias amplitude must be positive."

    if weight_mode == "uniform":
        init_fn = lambda weight: weight.data.uniform_(-W_amplitude, W_amplitude)
    elif weight_mode == "xavier_uniform":
        init_fn = lambda tensor: nn.init.xavier_uniform_(tensor, gain=W_amplitude)
    elif weight_mode == "normal":
        init_fn = lambda weight: weight.data.normal_(mean=0, std=W_amplitude)
    elif weight_mode == "xavier_normal":
        init_fn = lambda tensor: nn.init.xavier_normal_(tensor, gain=W_amplitude)
    else:
        raise ValueError(f"Invalid distribution {weight_mode}")

    if bias_mode == "uniform":
        bias_init = lambda bias: bias.data.uniform_(-b_amplitude, b_amplitude)
    elif bias_mode == "normal":
        bias_init = lambda bias: bias.data.normal_(mean=0, std=b_amplitude)
    elif bias_mode == "zero":
        bias_init = lambda bias: bias.data.zero_()
    else:
        raise ValueError(f"Invalid bias_mode {bias_mode}")
    

    # Initialize weights and biases for all linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                bias_init(module.bias)
                
