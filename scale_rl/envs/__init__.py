from scale_rl.envs.gym import make_gym_env

def create_envs(env_name):
    env = make_gym_env(env_name)

    return env