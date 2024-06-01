import gymnasium as gym
from stable_baselines3 import PPO
import os

models_dir = 'models/PPO'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('LunarLander-v2', render_mode='human')
env.reset()
model_path = f'{models_dir}/20.zip'
model = PPO.load(model_path, env=env)
vec_env = model.get_env()

episodes = 5
for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        action, states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        env.render()
        print(f'rewards {rewards}')

env.close()
