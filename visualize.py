# visualize.py
import gymnasium as gym
import torch
from agent import Agent
from utils import show_video_of_model, show_video
import os

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

# Load the trained model if it exists
if os.path.exists('checkpoint.pth'):
    agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth'))
    agent.local_qnetwork.eval()  # Set the network to evaluation mode
else:
    print("No checkpoint found. Please train the agent first.")

# Visualizing the results
show_video_of_model(agent, 'LunarLander-v3')
show_video()
