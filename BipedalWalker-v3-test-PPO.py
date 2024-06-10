import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

_env = gym.make("BipedalWalker-v3", render_mode='human')

env = DummyVecEnv([lambda: _env])

model = PPO("MlpPolicy", env, verbose=1)

print('start to learn')
model.learn(total_timesteps=100)

print('start to play')
obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
env.close()
