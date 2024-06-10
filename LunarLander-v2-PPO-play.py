import gymnasium as gym
from stable_baselines3 import PPO

model_path = 'models/PPO/LunarLander/20.zip'

env = gym.make('LunarLander-v2', render_mode='human')
env.reset()
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

env.close()
