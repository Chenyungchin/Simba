# =========== Import =============
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from scale_rl.buffers import create_buffer
from scale_rl.agents import create_agent
from scale_rl.envs import create_envs
from scale_rl.evaluation import eval_policy


# ================ main =================
def init_flags(env_name="Pendulum-v1"):

    flags = {
        "env_type": "gym",
        # "env_name": "humanoid-run",
        # "env_name": "pendulum-swingup",
        "env_name": env_name,
        "seed":0,
        "start_timesteps": 1e4,
        "max_timesteps": 2e5,
        "expl_noise": 0.01,
        "batch_size": 256,
        "discount":0.99,
        "tau": 0.005,
        "policy_noise": 0.05,
        "noise_clip":0.5,
        "policy_freq": 2,
        "save_model": "store_true",
        "rescale_action": True,
        "no_termination": False,
        "action_repeat": 2,
        "reward_scale": 1
    }

    return flags

def main(policy_name='TD3', env_name="pendulum-swingup", use_RSNorm=False, use_LayerNorm=False, use_Residual=False):
        
        #############################
        # envs
        #############################

        args = init_flags(env_name=env_name)
        env = create_envs(args["env_type"], args["env_name"], args["seed"], args["rescale_action"], args["no_termination"], args["action_repeat"], args["reward_scale"])
        # env = gym.make("Pendulum-v1") # use gym env for testing
        # env = gym.make("Humanoid-v5") # use gym env for testing
        env.reset(seed=args["seed"])
        env.action_space.seed(args["seed"])
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])

        #############################
        # agent
        #############################

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": args["discount"],
                "tau": args["tau"],}
        kwargs["policy_name"] = policy_name
        kwargs["policy_noise"] = args["policy_noise"] * max_action
        kwargs["noise_clip"] = args["noise_clip"] * max_action
        kwargs["policy_freq"] = args["policy_freq"]
        kwargs["use_RSNorm"] = use_RSNorm
        kwargs["use_LayerNorm"] = use_LayerNorm
        kwargs["use_Residual"] = use_Residual

        policy = create_agent(**kwargs)
        
        #############################
        # buffer
        #############################
        replay_buffer = create_buffer(state_dim, action_dim)



        evaluations = [eval_policy(policy, args["env_type"], args["env_name"], args["seed"], args["rescale_action"], args["no_termination"], args["action_repeat"], args["reward_scale"])]
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(args["max_timesteps"])):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args["start_timesteps"]:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args["expl_noise"], size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            done_bool = float(done) if episode_timesteps < 1000 else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args["start_timesteps"]:
                policy.train(replay_buffer, args["batch_size"])

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

                evaluations.append(episode_reward)

                # Reset environment
                state, _ = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        return evaluations


if __name__ == "__main__":
    # =========== run config ============
    policy_name = 'SAC'
    # dmc envs
    # env_name = "pendulum-swingup"
    # env_name = "humanoid-run"
    # gym envs
    # env_name = "Pendulum-v1"
    env_name = "Humanoid-v5"
    use_RSNorm = True
    use_LayerNorm = False
    use_Residual = False

    print(f"Policy: {policy_name}, Env: {env_name}, use_RSNorm: {use_RSNorm}, use_LayerNorm: {use_LayerNorm}, use_Residual: {use_Residual}")

    evaluation_sac = main(
        policy_name = policy_name,
        env_name = env_name,
        use_RSNorm = use_RSNorm,
        use_LayerNorm = use_LayerNorm,
        use_Residual = use_Residual
    )

    task_name = f"{policy_name}_{env_name}_RSNorm_{use_RSNorm}_LayerNorm_{use_LayerNorm}_Residual_{use_Residual}"

    plt.figure(figsize=(10, 5))
    plt.plot(evaluation_sac)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"{task_name} Evaluation")
    plt.grid()
    plt.show()
    plt.savefig(f'output/{task_name}.png')

    import pickle
    with open(f'output/{task_name}.pkl', 'wb') as file:
        pickle.dump(evaluation_sac, file)