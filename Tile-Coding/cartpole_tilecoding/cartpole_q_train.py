import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from tilecoding import TileCoder
import pandas as pd
import pickle

lr=0.5
initial_lr = 0.5
min_lr = 0.1
gamma =0.99
max_steps=200
epi_num=5000
max_epsilon=1.0
min_epsilon=0.01
decay_rate = (max_epsilon-min_epsilon)/epi_num

env = gym.make("CartPole-v1")

''' 
##Q = defaultdict(lambda: np.zeros(2))  ##
 ignored using lambda function because : Lambda functions are anonymous and cannot be referenced by name.'
 The pickle module requires functions to be defined at the top level of a module (with a name) to serialize them
 '''

def default_q_value():
    return np.zeros(env.action_space.n)

Q = defaultdict(default_q_value)


def epsilon_greedy(state,epsilon):
    p=np.random.random()
    if p<epsilon:
        action=env.action_space.sample()
    else:
        action=np.argmax(Q[state])
    return action

def state_discretize(state):
    position_step=0.6
    velocity_step=0.75
    angposition_step=0.075
    angvelocity_step=0.5
    state[0] = np.round(state[0] / position_step) * position_step
    state[1] = np.round(state[1] / velocity_step) * velocity_step
    state[2] = np.round(state[2] / angposition_step) * angposition_step
    state[3] = np.round(state[3] / angvelocity_step) * angvelocity_step

    rounded_state= tuple(np.clip(state,-3,3))
    return rounded_state

def state_tiler(state):
    # the number of tiles per dimension
    tiles_per_dim = [8, 4, 10, 4]  # Example: 8 tiles for each of the 4 dimensions
    
    # the value limits for each dimension
    value_limits = [
        (-2.4, 2.4),       # Position limits
        (-3.0, 3.0),       # Velocity limits
        (-0.2095, 0.2095),   # Angle limits (in radians)
        (-2.0, 2.0)        # Angular velocity limits
    ]
    
    # Number of tilings
    tilings = 4  # Example: 8 tilings
    
    # Create a TileCoder instance
    tiler = TileCoder(tiles_per_dim, value_limits, tilings)
    
    # Encode the state into tiles
    active_tiles = tiler[state]
    
    return tuple(active_tiles)


def q_learn_control(lr):
    episodic_rewards = []
    episodic_lengths = []
    epsilon = max_epsilon
    
    for episode in range(epi_num):
        state, _ = env.reset()
        state = state_tiler(state)
        total_reward = 0
        done = False
        steps=0
        while not done:
            action = epsilon_greedy(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = state_tiler(next_state)
            if done :
                reward=-1
            # Q-learning update
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += lr * td_error
            
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
                
       
        epsilon= max(min_epsilon,max_epsilon-(episode*decay_rate))
        lr = max(min_lr, lr * (1 - episode / epi_num))
        
        episodic_rewards.append(total_reward)
        episodic_lengths.append(steps)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episodic_rewards[-100:])
            print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    return episodic_rewards, episodic_lengths


episodic_rewards, episodic_lengths=q_learn_control(lr=0.5)
plt.plot(episodic_rewards)
plt.plot(pd.Series(episodic_rewards).rolling(200).mean())
plt.title("Training Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)