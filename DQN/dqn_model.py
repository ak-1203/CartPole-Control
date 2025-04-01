import numpy as np 
import gymnasium as gym
import random
import torch 
import torch.nn as nn
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.int64).unsqueeze(1))  # Fixed dtype

    def size(self):
        return len(self.buffer)

class DQN_nn(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_nn, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.output(x) 
        return x
                
class DQN_agent():
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.0001, batch_size=64, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)

        self.model = DQN_nn(state_dim, action_dim)
        self.target_model = DQN_nn(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0  
        self.epsilon_decay = 0.995  
        self.epsilon_min = 0.1  

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()

    def update(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        q_values = self.model(states).gather(1, actions) # dim(states) = (batch_size,state_dim) 
        #.gather()to get q values corresponding to particular action in actions

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)  

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

