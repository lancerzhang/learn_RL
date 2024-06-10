import gymnasium as gym
from stable_baselines3 import PPO
import os

models_dir = 'models/PPO/LunarLander'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('LunarLander-v2', render_mode='human')
env.reset()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100)
TIMESTEPS = 10
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{models_dir}/{TIMESTEPS * iters}')

env.close()
