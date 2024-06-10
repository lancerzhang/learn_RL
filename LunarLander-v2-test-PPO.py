import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('LunarLander-v2', render_mode='human')

model = PPO('MlpPolicy', env, verbose=1)

print('start to learn')
model.learn(total_timesteps=100)

episodes = 10
vec_env = model.get_env()
obs = vec_env.reset()

print('start to play')
for ep in range(episodes):
    done = False
    while not done:
        action, states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        env.render()

env.close()
