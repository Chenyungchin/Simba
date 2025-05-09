import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
from torch import nn

from simplicity_bias.util.models.neural_networks import DummyNeuralNetwork, initialize_weights
from simplicity_bias.util.models.activations import SinActivation, GaussianActivation
from simplicity_bias.util.estimations import LinearDecisionSpaceEstimator
from simplicity_bias.util.spectral_analysis import get_mean_frequency_2d

MIN_AMPLITUDE = 0.10
MAX_AMPLITUDE = 100
AMPLITUDE_SPACING = "log"  # "linear" or "log"
STEPS_AMPLITUDE = 15
# X1, X2, Y1, Y2 = -1, 1, -1, 1
POINTS_PER_AXIS = 300
BATCH_SIZE = 16384


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def analyze_output_space(model, input_space, param_init_config, policy_name="SAC"):
    """
    Inputs:
        model: The model to analyze.
        param_init_config: A dictionary containing the parameters for initialization.
            - method: The method for weight/bias initialization.
            - amplitude: The amplitude for weight/bias initialization.

    Outputs:
    """

    weight_mode = param_init_config["weight_mode"]
    bias_mode = param_init_config["bias_mode"]
    amplitude = param_init_config["amplitude"]

    X1, X2, Y1, Y2 = input_space

    estimator = LinearDecisionSpaceEstimator(start=(X1, Y1), end=(X2, Y2), steps=POINTS_PER_AXIS)

    # Estimate outputs spaces
    print(f"Weight Initialization Method: {weight_mode}")
    print(f"Bias Initialization Method: {bias_mode}")
    print(f"Param Initialization Amplitude: {amplitude:.2f}")

    # Initialize random weights
    initialize_weights(
        model,
        weight_mode=weight_mode,
        bias_mode=bias_mode,
        W_amplitude=amplitude,
        b_amplitude=amplitude,
    )

    # Perform warmup to calibrate running statistics without updating weights
    warmup(model, warmup_steps=1000)

    # Estimate output space
    output_space = estimator.estimate(model, batch_size=BATCH_SIZE, policy_name=policy_name)

    # Compute Fourier decomposition
    coeff_2d = torch.fft.rfft2(output_space)
    coeff_2d = abs(coeff_2d.detach()).cpu()
    # Ignore 0 "Hz" frequency, as shift is not important
    coeff_2d[0, 0] = 0
    # Ignore lower half
    coeff_2d = coeff_2d[: coeff_2d.shape[0] // 2 + 1]

    mean_freq = get_mean_frequency_2d(coeff_2d)
    simplicity_score = 1 / mean_freq if mean_freq != 0 else 1000

    return simplicity_score

def warmup(model, warmup_steps=1000):
    """Warmup the model to calibrate running statistics without updating weights."""
    # Freeze the weights
    for param in model.parameters():
        param.requires_grad = False

    # Ensure the model is in training mode
    model.train()

    # Perform the warmup
    for _ in range(warmup_steps):
        x = 2 * torch.rand(1, 2).to(DEVICE) - 1  # Random input in the range [-1, 1]
        _ = model(x)  # Forward pass to warm up the model


def test(model=None):
    assert AMPLITUDE_SPACING in ["linear", "log"]
    if AMPLITUDE_SPACING == "linear":
        amplitudes = np.linspace(MIN_AMPLITUDE, MAX_AMPLITUDE, STEPS_AMPLITUDE)
    elif AMPLITUDE_SPACING == "log":
        amplitudes = np.logspace(np.log10(MIN_AMPLITUDE), np.log10(MAX_AMPLITUDE), STEPS_AMPLITUDE)
    else:
        raise ValueError(f"AMPLITUDE_SPACING must be 'linear' or 'log'.")
    
    if model is None:
        model = DummyNeuralNetwork(input_dim=2, output_dim=1).to(DEVICE)
    
    for amplitude in amplitudes:
        param_init_config = {
            "method": "uniform",
            "amplitude": amplitude,
        }
        simplicity_score = analyze_output_space(model, param_init_config)
        print(f"simplicity score: {simplicity_score:.4f}")


if __name__ == "__main__":
    test()
