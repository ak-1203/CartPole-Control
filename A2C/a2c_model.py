import numpy as np 
import gymnasium as gym
import random
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt  

class ActorCritic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(ActorCritic,self).__init__()
        self.shared=nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
        
        self.actor=nn.Linear(128,action_dim)

        self.critic=nn.Linear(128,1)

    def forward(self,state):
        features = self.shared(state)
        action_probs=F.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value
    
class ActorCriticAgent:
    def __init__(self,state_dim,action_dim,lr=2e-3,gamma=0.85):
        self.gamma=gamma
        self.model=ActorCritic(state_dim,action_dim)
        self.optimizer=optim.Adam(self.model.parameters(),lr=lr)

    def select_actions(self,state):
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        action_probs,_=self.model(state)
        action = torch.multinomial(action_probs, 1).item() ##
       
        return action, action_probs[0, action] ##

    def update(self,state,action_prob,reward,next_state,done):
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0)

        _,state_value=self.model(state)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        _, next_state_value = self.model(next_state_tensor)

        target=reward+self.gamma*next_state_value*(1-done)
        advantage=target - state_value


        critic_loss = F.mse_loss(state_value, target) ##

        actor_loss = -torch.log(action_prob + 1e-8) * advantage

        loss=actor_loss+critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()