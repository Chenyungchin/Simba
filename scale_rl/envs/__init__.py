import gymnasium as gym

# from scale_rl.envs.gym import make_gym_env
from scale_rl.envs.dmc import make_dmc_env
from gymnasium.wrappers import TimeLimit

def create_envs(
    env_type: str,
    env_name: str,
    seed: int,
    max_episode_steps: int = 1000,
):
    # TODO: separate train and eval envs
    env = make_env(
        env_type=env_type,
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )

    return env

def make_env(
    env_type: str,
    env_name: str,
    seed: int,
    max_episode_steps: int,
) -> gym.Env:

    if env_type == 'dmc':
        env = make_dmc_env(env_name, seed)
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")
    
    # max episode steps
    env = TimeLimit(env, max_episode_steps)
    
    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    return env