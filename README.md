#  Deep Reinforcement Learning on CartPole: DQN, A2C, PPO and Tile Coding
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red)](https://pytorch.org/)  [![Stable-Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.1.0-blue)](https://stable-baselines3.readthedocs.io/)  

##  Introduction:
CartPole is a classical control problem in Reinforcement Learning, also known as the Inverted Pendulum problem.

### Problem Statement:
Imagine a cart that can move left or right along a track. A pole is attached to the cart via a hinge at its base. The objective is to apply forces to the cart — either left or right — in order to keep the pole balanced upright for as long as possible without it falling over.

This environment is widely used as a benchmark to test and compare reinforcement learning algorithms due to its simplicity and dynamic behavior.
In this repository, I share my implementation and analysis of various Deep Reinforcement Learning (DRL) algorithms applied to OpenAI Gym's **CartPole-v1** environment. 


##  Implementations Included:  
- **Deep Q-Network (DQN)** and **Advantage Actor-Critic (A2C)** implemented from scratch using **PyTorch**.  
- **Proximal Policy Optimization (PPO)** using **Stable Baselines3**.  
- **Q-Learning with Tile Coding** for handling continuous state spaces.  

---

<div align="center">
  <img src="Tile-Coding/training.gif" width="40%">
   <p><em>Training visualization of agent on CartPole-v1</em></p>
</div>

---

##  Key Features  

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

---
## 📈 Training Progress  

Below are the training and testing performance graphs for different reinforcement learning algorithms. All plots are scaled uniformly for a clean and professional appearance.

### 🔹 **Q-Learning with Tile Coding**  
<div align="center">
  <img src="Tile-Coding/cartpole_tilecoding/Training_Progress.png" alt="Tile Coding - Training" width="48%">
  <img src="Tile-Coding/cartpole_tilecoding/Testing_Progress.png" alt="Tile Coding - Testing" width="48%">
</div>  

### 🔹 **Deep Q-Network (DQN)**  
<div align="center">
  <img src="DQN/dqn_training.png" alt="DQN - Training" width="48%">
  <img src="DQN/dqn_testing.png" alt="DQN - Testing" width="48%">
</div>  

### 🔹 **Advantage Actor-Critic (A2C)**  
<div align="center">
  <img src="A2C/a2c_train.png" alt="A2C - Training" width="48%">
  <img src="A2C/a2c_testing.png" alt="A2C - Testing" width="48%">
</div>  

### 🔹 **Proximal Policy Optimization (PPO)**  
<div align="center">
  <img src="PPO/ppo_training.png" alt="PPO - Training" width="60%">
</div>
<div align="center">
  <img src="PPO/ppo_testing.png" alt="PPO - Testing" width="60%">
</div>

## 🛠 Installation  

To get started, clone this repository and install dependencies:

```bash
git clone https://github.com/ak-1203/CartPole-Control

pip install gymnasium numpy matplotlib pandas tqdm stable-baselines3 torch
```

##  Planned Updates
- Implement Soft Actor-Critic (SAC) for continuous action spaces.
- Extend tile coding to other environments like MountainCar.
- Experiment with different network architectures for A2C and PPO
