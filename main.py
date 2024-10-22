import gymnasium as gym
import torch
from collections import deque
import numpy as np
from agent import Agent

# Setup the environment
env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape:', state_shape)
print('State size:', state_size)
print('Number of actions:', number_actions)

# Initialize the agent
agent = Agent(state_size, number_actions)

# Hyperparameters
number_episodes = 2000
max_timesteps = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_rate = 0.995
epsilon = epsilon_start
scores_on_100_episodes = deque(maxlen=100)

# Training the agent
for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(max_timesteps):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_end, epsilon_decay_rate * epsilon)
    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}', end="")
    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
    if np.mean(scores_on_100_episodes) >= 200.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break
