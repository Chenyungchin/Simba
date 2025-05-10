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

def SimplicityBias(model, input_space=(-1, 1, -1, 1), policy_name="SAC", weight_mode="uniform", bias_mode="zero"):
    # Note: without amplitude sweep the figure looks wrong
    # assert AMPLITUDE_SPACING in ["linear", "log"]
    # if AMPLITUDE_SPACING == "linear":
    #     amplitudes = np.linspace(MIN_AMPLITUDE, MAX_AMPLITUDE, STEPS_AMPLITUDE)
    # elif AMPLITUDE_SPACING == "log":
    #     amplitudes = np.logspace(np.log10(MIN_AMPLITUDE), np.log10(MAX_AMPLITUDE), STEPS_AMPLITUDE)
    # else:
    #     raise ValueError(f"AMPLITUDE_SPACING must be 'linear' or 'log'.")
    
    param_init_config = {
        "weight_mode": weight_mode,
        "bias_mode": bias_mode, 
        "amplitude": torch.sqrt(torch.tensor(2.0)),
    }
    simplicity_score = analyze_output_space(model, input_space, param_init_config, policy_name)
    print(f"simplicity score: {simplicity_score:.4f}")
    
    return simplicity_score

def get_agent(policy_name="TD3", use_RSNorm=False, use_LayerNorm=False, use_Residual=False, use_MLP_ReLU=False):
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
            use_MLP_ReLU=use_MLP_ReLU,
        )

    elif policy_name == "SAC":
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
            use_MLP_ReLU=use_MLP_ReLU,
        )
    
    actor = model.actor
    critic = model.critic

    return actor, critic

def main(model = None, input_space=(-1, 1, -1, 1), policy_name="SAC", weight_mode="uniform", bias_mode="zero"):

    if model is None:
        model = DummyNeuralNetwork(input_dim=2, output_dim=1).to(DEVICE)

    simplicity_score = SimplicityBias(model, input_space, policy_name, weight_mode, bias_mode)

    return simplicity_score

if __name__ == "__main__":
    use_MLP_ReLU = False
    use_RSNorm = False
    use_LayerNorm = False
    use_Residual = False
    # policy_name = "SAC"
    policy_name = "TD3"

    input_min, input_max = -100, 100
    input_space = (input_min, input_max, input_min, input_max)

    sweep_list = [
        (True, False, False, False), # MLP
        (False, True, False, False), # MLP+RSNorm
        (False, False, True, False), # MLP+LayerNorm
        (False, False, False, True), # MLP+Residual
        (False, True, True, True) # SIMBA
    ]
    num_trials = 100

    # weight_mode = "kaiming_normal"
    weight_mode = "orthogonal_normal"
    # weight_mode = "xavier_normal"
    bias_mode = "zero"

    import time
    curr = time.localtime()
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", curr)
    json_filename = f"output/simplicity_scores_({input_min},{input_max})_{weight_mode}_{bias_mode}_{policy_name}_{timestamp}.json"
    # json_filename = "output/" + "simplicity_scores_(-100,100)_xavier_normal_zero_TD3_2025-05-09_16-31-38.json"

    for use_MLP_ReLU, use_RSNorm, use_LayerNorm, use_Residual in sweep_list:
        for i in range(num_trials):
            actor, critic = get_agent(policy_name=policy_name, use_RSNorm=use_RSNorm, use_LayerNorm=use_LayerNorm, use_Residual=use_Residual, use_MLP_ReLU=use_MLP_ReLU)

            simplicity_score = main(actor, input_space, policy_name, weight_mode=weight_mode, bias_mode=bias_mode)

            # Save simplicity_scores to a JSON file
            if use_MLP_ReLU:
                actor_mode = "MLP"
            elif use_RSNorm and use_LayerNorm and use_Residual:
                actor_mode = "Simba (all)"
            elif use_RSNorm:
                actor_mode = "Simba (RSNorm)"
            elif use_LayerNorm:
                actor_mode = "Simba (LayerNorm)"
            elif use_Residual:
                actor_mode = "Simba (Residual)"

            print(f"actor_mode: {actor_mode}")

            # Read existing data from the JSON file
            try:
                with open(json_filename, "r") as f:
                    data = json.load(f)  # Load existing JSON data
                    # ignore if the data already has num_trials data points
                    if actor_mode in data and len(data[actor_mode]) >= num_trials:
                        print(f"Skipping {actor_mode} as it already has {num_trials} data points.")
                        break
            except FileNotFoundError:
                data = {}  # If file doesn't exist, start with an empty dictionary

            

            # Add new data to the existing JSON object
            if actor_mode not in data:
                data[actor_mode] = []  # Initialize an empty list if the key doesn't exist
            data[actor_mode].append(simplicity_score)  # Append the new scores to the list

            # Write the updated JSON object back to the file
            with open(json_filename, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Simplicity scores for {actor_mode} saved to {json_filename}")

