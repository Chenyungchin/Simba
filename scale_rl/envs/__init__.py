import gymnasium as gym

# from scale_rl.envs.gym import make_gym_env
from scale_rl.envs.dmc import make_dmc_env
from gymnasium.wrappers import TimeLimit, RescaleAction
from scale_rl.envs.wrappers import RepeatAction, ScaleReward, DoNotTerminate

def create_envs(
    env_type: str,
    env_name: str,
    seed: int,
    rescale_action: bool,
    no_termination: bool,
    action_repeat: int,
    reward_scale: float,
    max_episode_steps: int = 1000,
    
):
    # TODO: separate train and eval envs
    env = make_env(
        env_type=env_type,
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        rescale_action=rescale_action,
        no_termination=no_termination,
        action_repeat=action_repeat,
        reward_scale=reward_scale
    )

    return env

def make_env(
    env_type: str,
    seed: int,
    env_name: str,
    rescale_action: bool = False,
    no_termination: bool = False,
    action_repeat: int = 1,
    reward_scale: float = 1.0,
    max_episode_steps: int = 1000,
    **kwargs,
) -> gym.Env:

    if env_type == 'dmc':
        env = make_dmc_env(env_name, seed)

        if rescale_action:
            env = RescaleAction(env, -1.0, 1.0)

        if no_termination:
            env = DoNotTerminate(env)

        # limit max_steps before action_repeat.
        env = TimeLimit(env, max_episode_steps)

        if action_repeat > 1:
            env = RepeatAction(env, action_repeat)

        env = ScaleReward(env, reward_scale)

        env.observation_space.seed(seed)
        env.action_space.seed(seed)    

    elif env_type == 'gym':
        env = gym.make(env_name)

    return env