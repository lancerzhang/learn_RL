import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env():
    return gym.make("BipedalWalker-v3")


if __name__ == "__main__":
    start_time = time.time()
    num_envs = 10  # 你可以根据显卡性能调整这个值
    time_steps = 1000 * 1000
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # 设置模型保存的回调函数
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='models/PPO/BipedalWalker/')

    model = PPO("MlpPolicy", env, verbose=1)

    # 开始训练并定期保存模型
    model.learn(total_timesteps=time_steps, callback=checkpoint_callback)

    # 保存最终模型
    model.save("models/PPO/BipedalWalker/final")

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    end_time = time.time()
    print(f'Used {int(end_time - start_time)} seconds to train')
