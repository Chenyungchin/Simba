
import gymnasium as gym

def make_gym_env(env_name):
    env = gym.make(env_name)

    return env