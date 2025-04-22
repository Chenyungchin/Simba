from scale_rl.agents import create_agent
from simplicity_bias.complexity_measurer import analyze_output_space
from simplicity_bias.util.models.neural_networks import DummyNeuralNetwork
import numpy as np
import torch


MIN_AMPLITUDE = 0.10
MAX_AMPLITUDE = 100
AMPLITUDE_SPACING = "log"  # "linear" or "log"
STEPS_AMPLITUDE = 15

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def SimplicityBiasAmplitudeSweep(model):
    assert AMPLITUDE_SPACING in ["linear", "log"]
    if AMPLITUDE_SPACING == "linear":
        amplitudes = np.linspace(MIN_AMPLITUDE, MAX_AMPLITUDE, STEPS_AMPLITUDE)
    elif AMPLITUDE_SPACING == "log":
        amplitudes = np.logspace(np.log10(MIN_AMPLITUDE), np.log10(MAX_AMPLITUDE), STEPS_AMPLITUDE)
    else:
        raise ValueError(f"AMPLITUDE_SPACING must be 'linear' or 'log'.")

    simplicity_scores = []
    
    for amplitude in amplitudes:
        param_init_config = {
            "method": "uniform",
            "amplitude": amplitude,
        }
        simplicity_score = analyze_output_space(model, param_init_config)
        print(f"simplicity score: {simplicity_score:.4f}")
        simplicity_scores.append(simplicity_score)
    
    return simplicity_scores

def main(model = None):

    if model is None:
        model = DummyNeuralNetwork(input_dim=2, output_dim=1).to(DEVICE)

    simplicity_scores = SimplicityBiasAmplitudeSweep(model)

    return simplicity_scores

if __name__ == "__main__":
    # SAC
    # agent = create_agent(
    #     policy_name="SAC",
    #     state_dim=2,
    #     action_dim=1,
    #     max_action=1.0,
    #     discount=0.99,
    #     tau=0.005,
    #     # 
    #     policy_noise=0.05 * 1.0,
    #     noise_clip=0.5 * 1.0,
    #     policy_freq=2,
    # )
    agent = create_agent(
        policy_name="TD3",
        state_dim=2,
        action_dim=1,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        #
        policy_noise=0.05 * 1.0,
        noise_clip=0.5 * 1.0,
        policy_freq=2,
    )
    actor = agent.actor
    critic = agent.critic
    main(actor)

    
