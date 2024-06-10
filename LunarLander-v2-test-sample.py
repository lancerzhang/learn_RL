import gymnasium as gym
env = gym.make('LunarLander-v2', render_mode='human')

env.reset()

print('action_space.sample', env.action_space.sample())
print('observation_space.shape', env.observation_space.shape)
print('observation_space.sample', env.observation_space.sample())

for step in range(400):
    env.render()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(reward, terminated)

env.close()
