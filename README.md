# Lunar Lander AI using Gymnasium

This repository contains an implementation of a **Lunar Lander AI** built using the [Gymnasium](https://gymnasium.farama.org/) environment. The AI is designed to safely guide the rocket to land on a platform using reinforcement learning techniques, specifically **Deep Q-Learning**

## Overview

The goal of the **Lunar Lander** environment is to successfully land a spaceship on a designated landing pad. The AI must control the main engine and side thrusters to navigate through a low-gravity environment while avoiding crashes. The agent is rewarded for each successful landing and penalized for crashes or moving far from the landing zone.

### Key Features
- **Reinforcement Learning**: The agent learns through trial and error by interacting with the environment and receiving rewards or penalties based on its actions.
- **Gymnasium Environment**: The project utilizes the `gymnasium[box2d]` environment, particularly the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) task.
- **Custom AI Algorithm**: The implementation uses an AI algorithm (such as DQN or other RL models) to train the agent for optimal landing performance.
  
## Installation

To install and run the environment locally, follow these steps:

1. Install the required dependencies:

    ```bash
    pip install gymnasium
    pip install "gymnasium[box2d]"
    ```

2. (Optional) Install additional dependencies if you're using Atari-based features or other enhancements:

    ```bash
    pip install "gymnasium[atari, accept-rom-license]"
    ```

3. Run the AI:

    After installing the dependencies, you can run the training script or watch a pre-trained model in action by executing:

    ```bash
    python lunar_lander_ai.py
    ```

## How It Works

1. **Environment Setup**: The agent is trained in the Gymnasium Lunar Lander environment. This environment simulates the physics of lunar landings.
   
2. **Reinforcement Learning**: The AI uses reinforcement learning algorithms, such as Deep Q-Learning (DQN), to learn how to land the lunar module safely. It improves over time by optimizing the reward it receives for successful landings.

3. **Training**: The AI undergoes multiple episodes of training, where it interacts with the environment, learns from its mistakes, and adapts its strategy to maximize rewards.

## Requirements

- Python 3.7 or higher
- Gymnasium (`gymnasium[box2d]`)
- NumPy, TensorFlow/PyTorch (optional for training)

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for providing the environment.
- Farama Foundation for maintaining the Box2D environment and related tools.



![image](https://github.com/user-attachments/assets/063432e8-c17d-401a-a581-053a948f2249)
![image](https://github.com/user-attachments/assets/deea1c48-f76c-4559-b830-fbef347f4b48)
![image](https://github.com/user-attachments/assets/c224131b-ea27-4b66-9037-90f373044188)
