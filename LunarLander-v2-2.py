import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('LunarLander-v2', render_mode='human')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100)

episodes = 10
vec_env = model.get_env()
obs = vec_env.reset()

for ep in range(episodes):
    done = False
    while not done:
        action, states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        env.render()
        print(f'obs {obs}')
        print(f'rewards {rewards}')
        print(f'info {info}')

env.close()
