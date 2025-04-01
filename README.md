# Deep Reinforcement Learning: Algorithms & Tile Coding

This repository provides implementations of various Deep Reinforcement Learning (DRL) algorithms for solving OpenAI Gym's **CartPole-v1** environment. 

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red)](https://pytorch.org/)  
[![Stable-Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.1.0-blue)](https://stable-baselines3.readthedocs.io/)  
It includes:

- **Deep Q-Network (DQN)** and **Advantage Actor-Critic (A2C)** implemented from scratch using **PyTorch**.
- **Proximal Policy Optimization (PPO)** using **Stable Baselines3**.
- **Q-Learning with Tile Coding** for handling continuous state spaces.  

## 📌 Key Features

## 📌 Key Features  

| **Category**               | **Algorithms**          | **Key Improvements**                          |
|----------------------------|-------------------------|-----------------------------------------------|
| **From Scratch (PyTorch)** | DQN, A2C                | Experience replay, target networks, advantage estimation |
| **Stable-Baselines3**      | PPO                     | Clipped objectives, parallelized training     |
| **Discretization**         | Q-Learning + Tile Coding| 4 tilings, adaptive ε-greedy, state aggregation |

---


### 🔹 **Scratch Implementations (PyTorch)**
- **Deep Q-Network (DQN)**  
  - Uses experience replay and target networks for stability.  
  - Implemented from scratch in PyTorch.  

- **Advantage Actor-Critic (A2C)**  
  - Policy-based method using actor and critic networks.  
  - Trained using advantage estimation.  

### 🔹 **Stable Baselines3 Implementation**
- **Proximal Policy Optimization (PPO)**  
  - Implements policy optimization with clipping and advantage estimation.  
  - Uses Stable Baselines3 for efficient training.  

### 🔹 **Q-Learning with Tile Coding**
- **Tile Coding for State Discretization**  
  - 4 tilings with adaptive granularity.  
  - Custom discretization of position, velocity, angle, and angular velocity.  

- **Optimized Learning Parameters**  
  - **Exploration:** Decaying ε-greedy (1.0 → 0.01).  
  - **Learning Rate:** Adaptive schedule (0.5 → 0.1).  
  - **Discount Factor:** γ = 0.99.  


## 📈 Training Progress

### 🔹 Q-Learning with Tile Coding  
![Tile Coding - Training](DRL-Algorithms-in-Cartpole-Environment/Tile-Coding/cartpole_tilecoding/Training_Progress.png)  
![Tile Coding - Testing](DRL-Algorithms-in-Cartpole-Environment/Tile-Coding/cartpole_tilecoding/Testing_Progress.png)  

### 🔹 Deep Q-Network (DQN)  
![DQN - Training](DRL-Algorithms-in-Cartpole-Environment/DQN/dqn_training.png)  
![DQN - Testing](DRL-Algorithms-in-Cartpole-Environment/DQN/dqn_testing.png)  

### 🔹 Advantage Actor-Critic (A2C)  
![A2C - Training](DRL-Algorithms-in-Cartpole-Environment/A2C/a2c_training.png)  
![A2C - Testing](DRL-Algorithms-in-Cartpole-Environment/A2C/a2c_testing.png)  

### 🔹 Proximal Policy Optimization (PPO)  
![PPO - Training](DRL-Algorithms-in-Cartpole-Environment/PPO/ppo_training.png)  
![PPO - Testing](DRL-Algorithms-in-Cartpole-Environment/PPO/ppo_testing.png)  

## 🛠 Quick Start  
1. **Install dependencies**: 
gymnasium==0.29.1  
numpy==1.26.0  
matplotlib==3.8.0  
pandas==2.1.1  
tqdm==4.66.1   
stable-baselines3==2.1.0   
torch==2.0.0   

## Future Uploads
- Implement Soft Actor-Critic (SAC) for continuous action spaces.
- Extend tile coding to other environments like MountainCar.
- Experiment with different network architectures for A2C and PPO


