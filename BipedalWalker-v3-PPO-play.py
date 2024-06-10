import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

model_path = "models/PPO/BipedalWalker/final"

_env = gym.make("BipedalWalker-v3", render_mode='human')

env = DummyVecEnv([lambda: _env])

# 加载模型
model = PPO.load(model_path)

# 观察效果
obs = env.reset()
total_reward = 0
for _ in range(1000):  # 你可以调整步数以观察更长时间
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    if np.any(done):
        obs = env.reset()
        print(f"Total reward: {total_reward}")
        total_reward = 0
        time.sleep(2)  # 暂停2秒以观察新一轮的开始

env.close()
