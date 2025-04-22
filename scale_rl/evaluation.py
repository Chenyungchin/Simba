import numpy as np
from scale_rl.envs import create_envs

# policy evaluation with Monte Carlo
def eval_policy(policy, env_type, env_name, seed, rescale_action, no_termination, action_repeat, reward_scale, eval_episodes=10):
        eval_env = create_envs(env_type, env_name, seed, rescale_action, no_termination, action_repeat, reward_scale)
        eval_env.reset(seed=seed)
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, _ = eval_env.reset()
            done = False
            step = 0
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, terminated, truncated, _ = eval_env.step(action)
                avg_reward += reward
                step += 1
                done = terminated or truncated
        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward