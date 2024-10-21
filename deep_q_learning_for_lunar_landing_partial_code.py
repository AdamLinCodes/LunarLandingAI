## Part 0 - Installing the required packages and importing the libraries

### Installing Gymnasium
"""

!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

"""### Importing the libraries"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

"""## Part 1 - Building the AI"""

class Network(nn.Module):
  def __init__(self, state_size, action_size, seed = 42):
      super(Network, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.fc1 = nn.Linear(state_size, 64)  #first fully connected layer of nn
      self.fc2 = nn.Linear(64, 64)          #second layer
      self.fc3 = nn.Linear(64, action_size) #output layer

  def forward(self, state):
      x = self.fc1(state) # pass to first
      x = F.relu(x)       # rectifier activation function

      x = self.fc2(x)     # pass to second
      x = F.relu(x)

      return self.fc3(x) # return output layer

"""### Creating the architecture of the Neural Network

## Part 2 - Training the AI

### Setting up the environment
"""

import gymnasium as gym
env = gym.make('LunarLander-v3') # The Lunar Lander environment was upgraded to v3
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

"""### Initializing the hyperparameters"""

learning_rate = 5e-4
minibatch_size = 100 #num observations that we use in a single training step to ubdate params
discount_factor = 0.99 #gamma
replay_buffer_size = int(1e5) # memory of the agent, so the agent can remember stuff rather than training everytime
interpolation_parameter = 1e-3 #(tao)

"""### Implementing Experience Replay"""

class ReplayMemory(object):
  def __init__(self, capacity):
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if we want to use the gpu instead of cpu
      self.capacity = capacity # max size of memory buffer
      self.memory = []

  def push(self, event): #adds an experience to the memory buffer
      self.memory.append(event)
      if len(self.memory) > self.capacity: # remove oldest memory if we run out of space
          del self.memory[0]

  def sample(self, batch_size):
      experiences = random.sample(self.memory, k = batch_size)
      states = torch.from_numpy(np.vstack([experience[0] for experience in experiences if experience is not None])).float().to(self.device) # e[0] is the state
      actions = torch.from_numpy(np.vstack([experience[1] for experience in experiences if experience is not None])).long().to(self.device) # e[0] is the state
      rewards = torch.from_numpy(np.vstack([experience[2] for experience in experiences if experience is not None])).float().to(self.device) # e[0] is the state
      next_states = torch.from_numpy(np.vstack([experience[3] for experience in experiences if experience is not None])).float().to(self.device) # e[0] is the state
      dones = torch.from_numpy(np.vstack([experience[4] for experience in experiences if experience is not None]).astype(np.uint8)).float().to(self.device) # e[0] is the state
      return states, next_states, actions, rewards, dones

"""### Implementing the DQN class"""

class Agent():
  def __init__(self, state_size, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #if we want to use the gpu instead of cpu
    self.state_size = state_size
    self.action_size = action_size
    self.local_qnetwork = Network(state_size, action_size).to(self.device) #qnetwork for selecting actions
    self.target_qnetwork = Network(state_size, action_size).to(self.device) #for calculating values for different actions
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = ReplayMemory(replay_buffer_size)
    self.time_step = 0

  def step(self, state, action, reward, next_state, done):
    self.memory.push((state, action, reward, next_state, done))
    self.time_step = (self.time_step + 1) % 4

    if self.time_step == 0:
      if len(self.memory.memory) > minibatch_size:
        experiences = self.memory.sample(100) # he says use 100 for some reason
        self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    self.local_qnetwork.eval()
    with torch.no_grad(): # Since we are just passing values the nn, we don't need to do the grad calculations cause that will happen in back-prop
      action_values = self.local_qnetwork(state)
    self.local_qnetwork.train()

    # epsilon greedy
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, next_states, actions, rewards, dones = experiences
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

  def soft_update(self, local_model, target_model, interpolation_parameter):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

"""### Initializing the DQN agent"""

agent = Agent(state_size, number_actions)

"""### Training the DQN agent"""

number_episodes = 2000
max_timesteps = 1000 # per episode
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_rate = 0.995
epsilon = epsilon_start
scores_on_100_episodes = deque(maxlen = 100)

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
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 200.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break

"""## Part 3 - Visualizing the results"""

import glob
import io
import base64
import imageio
from IPython.display import HTML, display

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v3')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()