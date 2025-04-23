from scale_rl.agents import create_agent
from simplicity_bias.complexity_measurer import analyze_output_space
from simplicity_bias.util.models.neural_networks import DummyNeuralNetwork
import numpy as np
import torch
import json


MIN_AMPLITUDE = 0.10
MAX_AMPLITUDE = 100
AMPLITUDE_SPACING = "log"  # "linear" or "log"
STEPS_AMPLITUDE = 15

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def SimplicityBiasAmplitudeSweep(model, input_space=(-1, 1, -1, 1)):
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
        simplicity_score = analyze_output_space(model, input_space, param_init_config)
        print(f"simplicity score: {simplicity_score:.4f}")
        simplicity_scores.append(simplicity_score)
    
    return simplicity_scores

def get_agent(policy_name="TD3", use_RSNorm=False, use_LayerNorm=False, use_Residual=False):
    if policy_name == "TD3":
        model = create_agent(
            policy_name=policy_name,
            state_dim=2,
            action_dim=1,
            max_action=1.0,
            discount=0.99,
            tau=0.005,
            policy_noise=0.05 * 1.0,
            noise_clip=0.5 * 1.0,
            policy_freq=2,
            use_RSNorm=use_RSNorm,
            use_LayerNorm=use_LayerNorm,
            use_Residual=use_Residual,
        )

    elif policy_name == "SAC":
        raise NotImplementedError("SAC agent is not implemented yet.")
    
    actor = model.actor
    critic = model.critic

    return actor, critic

def main(model = None, input_space=(-1, 1, -1, 1)):

    if model is None:
        model = DummyNeuralNetwork(input_dim=2, output_dim=1).to(DEVICE)

    simplicity_scores = SimplicityBiasAmplitudeSweep(model, input_space)

    return simplicity_scores

if __name__ == "__main__":
    use_RSNorm = True
    use_LayerNorm = False
    use_Residual = False
    policy_name = "TD3"

    input_min, input_max = -100, 100
    input_space = (input_min, input_max, input_min, input_max)

    sweep_list = [(False, False, False), (True, False, False), (False, True, False), (False, False, True), (True, True, True)]
    num_trials = 10

    for use_RSNorm, use_LayerNorm, use_Residual in sweep_list:
        for i in range(num_trials):
            actor, critic = get_agent(policy_name=policy_name, use_RSNorm=use_RSNorm, use_LayerNorm=use_LayerNorm, use_Residual=use_Residual)

            simplicity_scores = main(actor, input_space)

            # Save simplicity_scores to a JSON file
            if use_RSNorm and use_LayerNorm and use_Residual:
                actor_mode = "SIMBA"
            elif use_RSNorm:
                actor_mode = "MLP+RSNorm"
            elif use_LayerNorm:
                actor_mode = "MLP+LayerNorm"
            elif use_Residual:
                actor_mode = "MLP+Residual"
            else:
                actor_mode = "MLP"

            # Read existing data from the JSON file
            try:
                with open(f"simplicity_scores_({input_min},{input_max}).json", "r") as f:
                    data = json.load(f)  # Load existing JSON data
            except FileNotFoundError:
                data = {}  # If file doesn't exist, start with an empty dictionary

            # Add new data to the existing JSON object
            if actor_mode not in data:
                data[actor_mode] = []  # Initialize an empty list if the key doesn't exist
            data[actor_mode].append(simplicity_scores)  # Append the new scores to the list

            # Write the updated JSON object back to the file
            with open(f"simplicity_scores_({input_min},{input_max}).json", "w") as f:
                json.dump(data, f, indent=4)

            print(f"Simplicity scores for {actor_mode} saved to simplicity_scores.json")

